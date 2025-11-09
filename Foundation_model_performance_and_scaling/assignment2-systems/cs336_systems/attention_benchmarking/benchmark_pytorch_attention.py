"""
Benchmark PyTorch attention implementation at different scales.

This script benchmarks the naive attention implementation to identify
memory and compute bottlenecks that FlashAttention-2 aims to solve.

Usage:
    # Benchmark all configurations
    python -m cs336_systems.attention_benchmarking.benchmark_pytorch_attention

    # Benchmark specific configuration
    python -m cs336_systems.attention_benchmarking.benchmark_pytorch_attention \
        --d-model 64 --seq-len 4096

    # Save results to CSV
    python -m cs336_systems.attention_benchmarking.benchmark_pytorch_attention \
        --output results/attention_benchmarking/pytorch_attention_benchmark.csv
"""

import argparse
import timeit
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def naive_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Naive attention implementation following: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V

    Args:
        Q: Query tensor of shape (batch, seq_len, d_model)
        K: Key tensor of shape (batch, seq_len, d_model)
        V: Value tensor of shape (batch, seq_len, d_model)

    Returns:
        Output tensor of shape (batch, seq_len, d_model)
    """
    # Compute attention scores: (batch, seq_len, seq_len)
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)

    # Apply softmax
    attn_weights = F.softmax(scores, dim=-1)

    # Apply attention to values
    output = torch.matmul(attn_weights, V)

    return output


def benchmark_attention(
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_warmup: int = 10,
    num_iterations: int = 100,
    device: str = "cuda",
) -> dict:
    """
    Benchmark attention at a specific configuration.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        d_model: Model/head dimension
        num_warmup: Number of warmup iterations
        num_iterations: Number of measurement iterations
        device: Device to run on

    Returns:
        Dictionary with timing and memory statistics, or error information
    """
    try:
        # Clear GPU memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Create random inputs
        Q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        K = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        V = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

        # Warmup
        for _ in range(num_warmup):
            output = naive_attention(Q, K, V)
            loss = output.sum()
            loss.backward()
            Q.grad = None
            K.grad = None
            V.grad = None
            torch.cuda.synchronize()

        # Clear gradients and reset memory stats
        Q.grad = None
        K.grad = None
        V.grad = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Benchmark forward pass
        forward_times = []
        for _ in range(num_iterations):
            start = timeit.default_timer()
            output = naive_attention(Q, K, V)
            torch.cuda.synchronize()
            end = timeit.default_timer()
            forward_times.append(end - start)

        # Measure memory before backward
        memory_before_backward = torch.cuda.memory_allocated() / (1024 ** 2)  # MB

        # Benchmark backward pass
        backward_times = []
        for _ in range(num_iterations):
            # Forward pass (not timed)
            output = naive_attention(Q, K, V)
            loss = output.sum()
            torch.cuda.synchronize()

            # Clear previous gradients
            Q.grad = None
            K.grad = None
            V.grad = None

            # Time backward pass
            start = timeit.default_timer()
            loss.backward()
            torch.cuda.synchronize()
            end = timeit.default_timer()
            backward_times.append(end - start)

        # Get peak memory
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

        return {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "d_model": d_model,
            "forward_mean_ms": np.mean(forward_times) * 1000,
            "forward_std_ms": np.std(forward_times) * 1000,
            "backward_mean_ms": np.mean(backward_times) * 1000,
            "backward_std_ms": np.std(backward_times) * 1000,
            "total_mean_ms": (np.mean(forward_times) + np.mean(backward_times)) * 1000,
            "memory_before_backward_mb": memory_before_backward,
            "peak_memory_mb": peak_memory,
            "status": "success",
            "error": None,
        }

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "d_model": d_model,
                "forward_mean_ms": None,
                "forward_std_ms": None,
                "backward_mean_ms": None,
                "backward_std_ms": None,
                "total_mean_ms": None,
                "memory_before_backward_mb": None,
                "peak_memory_mb": None,
                "status": "OOM",
                "error": "Out of memory",
            }
        else:
            raise


def benchmark_all_configurations(
    batch_size: int = 8,
    d_model_values: list = None,
    seq_len_values: list = None,
    num_warmup: int = 10,
    num_iterations: int = 100,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Benchmark attention across all configurations.

    Args:
        batch_size: Fixed batch size
        d_model_values: List of d_model values to test
        seq_len_values: List of sequence lengths to test
        num_warmup: Number of warmup iterations
        num_iterations: Number of measurement iterations
        output_path: Optional path to save results CSV

    Returns:
        DataFrame with benchmark results
    """
    if d_model_values is None:
        d_model_values = [16, 32, 64, 128]

    if seq_len_values is None:
        seq_len_values = [256, 1024, 4096, 8192, 16384]

    print("=" * 80)
    print("PyTorch Attention Benchmarking")
    print("=" * 80)
    print(f"Batch size: {batch_size}")
    print(f"d_model values: {d_model_values}")
    print(f"seq_len values: {seq_len_values}")
    print(f"Warmup iterations: {num_warmup}")
    print(f"Measurement iterations: {num_iterations}")
    print()

    results = []

    for d_model in d_model_values:
        for seq_len in seq_len_values:
            print(f"Testing d_model={d_model}, seq_len={seq_len}...", end=" ", flush=True)

            result = benchmark_attention(
                batch_size=batch_size,
                seq_len=seq_len,
                d_model=d_model,
                num_warmup=num_warmup,
                num_iterations=num_iterations,
            )

            results.append(result)

            if result["status"] == "OOM":
                print("OOM")
            else:
                print(f"✓ (fwd: {result['forward_mean_ms']:.2f}ms, "
                      f"bwd: {result['backward_mean_ms']:.2f}ms, "
                      f"mem: {result['peak_memory_mb']:.1f}MB)")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save to CSV if requested
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print()

    # Create pivot table for easy visualization
    for metric in ["forward_mean_ms", "backward_mean_ms", "peak_memory_mb", "status"]:
        print(f"\n{metric}:")
        print("-" * 80)

        if metric == "status":
            pivot = df.pivot_table(
                index="d_model",
                columns="seq_len",
                values=metric,
                aggfunc=lambda x: x.iloc[0] if len(x) > 0 else "N/A"
            )
        else:
            pivot = df.pivot_table(
                index="d_model",
                columns="seq_len",
                values=metric,
                aggfunc="first"
            )

        print(pivot.to_string())
        print()

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch attention implementation"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128],
        help="Head dimensions to test (default: 16 32 64 128)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        nargs="+",
        default=[256, 1024, 4096, 8192, 16384],
        help="Sequence lengths to test (default: 256 1024 4096 8192 16384)",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=100,
        help="Number of measurement iterations (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/attention_benchmarking/pytorch_attention_benchmark.csv",
        help="Output CSV file path",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a GPU.")
        return

    # Run benchmarks
    df = benchmark_all_configurations(
        batch_size=args.batch_size,
        d_model_values=args.d_model,
        seq_len_values=args.seq_len,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
        output_path=args.output,
    )

    # Find OOM threshold
    oom_configs = df[df["status"] == "OOM"]
    if not oom_configs.empty:
        print("\n" + "=" * 80)
        print("OUT OF MEMORY CONFIGURATIONS")
        print("=" * 80)
        print()
        print(oom_configs[["d_model", "seq_len", "status"]].to_string(index=False))
        print()

        # Find smallest OOM configuration
        smallest_oom = oom_configs.nsmallest(1, ["seq_len", "d_model"])
        print(f"Smallest OOM configuration:")
        print(f"  d_model = {smallest_oom.iloc[0]['d_model']}")
        print(f"  seq_len = {smallest_oom.iloc[0]['seq_len']}")
        print()
        print("Run memory accounting analysis for this configuration:")
        print(f"  python -m cs336_systems.attention_benchmarking.memory_accounting \\")
        print(f"    --d-model {smallest_oom.iloc[0]['d_model']} \\")
        print(f"    --seq-len {smallest_oom.iloc[0]['seq_len']}")


if __name__ == "__main__":
    main()
