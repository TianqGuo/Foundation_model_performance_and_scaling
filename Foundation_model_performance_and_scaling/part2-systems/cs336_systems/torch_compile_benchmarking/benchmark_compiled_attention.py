"""
Benchmark torch.compile performance on attention implementation.

This script compares vanilla PyTorch attention with torch.compile optimized version
to measure the impact of JIT compilation on attention kernels.

Usage:
    # Benchmark all configurations (compiled vs uncompiled)
    python -m cs336_systems.torch_compile_benchmarking.benchmark_compiled_attention

    # Benchmark specific configuration
    python -m cs336_systems.torch_compile_benchmarking.benchmark_compiled_attention \
        --d-model 64 --seq-len 4096

    # Save results to CSV
    python -m cs336_systems.torch_compile_benchmarking.benchmark_compiled_attention \
        --output results/torch_compile_benchmarking/compiled_attention_benchmark.csv
"""

import argparse
import timeit
from pathlib import Path
from typing import Optional

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


def benchmark_attention_variant(
    batch_size: int,
    seq_len: int,
    d_model: int,
    compiled: bool = False,
    num_warmup: int = 10,
    num_iterations: int = 100,
    device: str = "cuda",
) -> dict:
    """
    Benchmark attention (compiled or uncompiled) at a specific configuration.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        d_model: Model/head dimension
        compiled: Whether to use torch.compile
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

        # Create attention function
        attention_fn = naive_attention
        if compiled:
            attention_fn = torch.compile(naive_attention)

        # Create random inputs
        Q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        K = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        V = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

        # Warmup (important for compiled version to trigger compilation)
        for _ in range(num_warmup):
            output = attention_fn(Q, K, V)
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
            output = attention_fn(Q, K, V)
            torch.cuda.synchronize()
            end = timeit.default_timer()
            forward_times.append(end - start)

        # Measure memory before backward
        memory_before_backward = torch.cuda.memory_allocated() / (1024 ** 2)  # MB

        # Benchmark backward pass
        backward_times = []
        for _ in range(num_iterations):
            # Forward pass (not timed)
            output = attention_fn(Q, K, V)
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
            "compiled": compiled,
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
                "compiled": compiled,
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


def benchmark_compiled_vs_uncompiled(
    batch_size: int = 8,
    d_model_values: list = None,
    seq_len_values: list = None,
    num_warmup: int = 10,
    num_iterations: int = 100,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Benchmark attention comparing compiled vs uncompiled versions.

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
    print("Torch.compile Attention Benchmarking - Section 1.3.1(a)")
    print("=" * 80)
    print(f"Batch size: {batch_size}")
    print(f"d_model values: {d_model_values}")
    print(f"seq_len values: {seq_len_values}")
    print(f"Warmup iterations: {num_warmup}")
    print(f"Measurement iterations: {num_iterations}")
    print()
    print("Testing both vanilla and torch.compile versions...")
    print()

    results = []

    for d_model in d_model_values:
        for seq_len in seq_len_values:
            # Test uncompiled version
            print(f"Testing d_model={d_model}, seq_len={seq_len} (vanilla)...", end=" ", flush=True)
            result_vanilla = benchmark_attention_variant(
                batch_size=batch_size,
                seq_len=seq_len,
                d_model=d_model,
                compiled=False,
                num_warmup=num_warmup,
                num_iterations=num_iterations,
            )
            results.append(result_vanilla)

            if result_vanilla["status"] == "OOM":
                print("OOM")
                # Skip compiled version if vanilla OOMs
                result_compiled = {
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "d_model": d_model,
                    "compiled": True,
                    "forward_mean_ms": None,
                    "forward_std_ms": None,
                    "backward_mean_ms": None,
                    "backward_std_ms": None,
                    "total_mean_ms": None,
                    "memory_before_backward_mb": None,
                    "peak_memory_mb": None,
                    "status": "SKIPPED",
                    "error": "Vanilla OOM, skipping compiled",
                }
                results.append(result_compiled)
                print(f"Testing d_model={d_model}, seq_len={seq_len} (compiled)... SKIPPED")
                continue
            else:
                print(f"✓ (fwd: {result_vanilla['forward_mean_ms']:.2f}ms, "
                      f"bwd: {result_vanilla['backward_mean_ms']:.2f}ms)")

            # Test compiled version
            print(f"Testing d_model={d_model}, seq_len={seq_len} (compiled)...", end=" ", flush=True)
            result_compiled = benchmark_attention_variant(
                batch_size=batch_size,
                seq_len=seq_len,
                d_model=d_model,
                compiled=True,
                num_warmup=num_warmup,
                num_iterations=num_iterations,
            )
            results.append(result_compiled)

            if result_compiled["status"] == "OOM":
                print("OOM")
            else:
                print(f"✓ (fwd: {result_compiled['forward_mean_ms']:.2f}ms, "
                      f"bwd: {result_compiled['backward_mean_ms']:.2f}ms)")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save to CSV if requested
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

    # Print comparison tables
    print_comparison_tables(df)

    return df


def print_comparison_tables(df: pd.DataFrame) -> None:
    """Print comparison tables for compiled vs uncompiled attention."""
    print("\n" + "=" * 80)
    print("COMPARISON TABLES: COMPILED vs VANILLA")
    print("=" * 80)
    print()

    # Separate vanilla and compiled results
    df_vanilla = df[df["compiled"] == False].copy()
    df_compiled = df[df["compiled"] == True].copy()

    # Forward pass comparison
    print("Forward Pass Timing (ms)")
    print("-" * 80)
    print("\nVanilla:")
    if not df_vanilla.empty:
        pivot_vanilla = df_vanilla.pivot_table(
            index="d_model", columns="seq_len", values="forward_mean_ms", aggfunc="first"
        )
        print(pivot_vanilla.to_string())

    print("\nCompiled:")
    if not df_compiled.empty:
        pivot_compiled = df_compiled.pivot_table(
            index="d_model", columns="seq_len", values="forward_mean_ms", aggfunc="first"
        )
        print(pivot_compiled.to_string())

    # Speedup table
    print("\nSpeedup (Vanilla / Compiled):")
    if not df_vanilla.empty and not df_compiled.empty:
        # Merge dataframes
        merged = pd.merge(
            df_vanilla[["d_model", "seq_len", "forward_mean_ms"]],
            df_compiled[["d_model", "seq_len", "forward_mean_ms"]],
            on=["d_model", "seq_len"],
            suffixes=("_vanilla", "_compiled"),
        )
        merged["speedup"] = merged["forward_mean_ms_vanilla"] / merged["forward_mean_ms_compiled"]
        pivot_speedup = merged.pivot_table(
            index="d_model", columns="seq_len", values="speedup", aggfunc="first"
        )
        print(pivot_speedup.to_string())

    # Backward pass comparison
    print("\n\nBackward Pass Timing (ms)")
    print("-" * 80)
    print("\nVanilla:")
    if not df_vanilla.empty:
        pivot_vanilla_bwd = df_vanilla.pivot_table(
            index="d_model", columns="seq_len", values="backward_mean_ms", aggfunc="first"
        )
        print(pivot_vanilla_bwd.to_string())

    print("\nCompiled:")
    if not df_compiled.empty:
        pivot_compiled_bwd = df_compiled.pivot_table(
            index="d_model", columns="seq_len", values="backward_mean_ms", aggfunc="first"
        )
        print(pivot_compiled_bwd.to_string())

    # Backward speedup
    print("\nSpeedup (Vanilla / Compiled):")
    if not df_vanilla.empty and not df_compiled.empty:
        merged_bwd = pd.merge(
            df_vanilla[["d_model", "seq_len", "backward_mean_ms"]],
            df_compiled[["d_model", "seq_len", "backward_mean_ms"]],
            on=["d_model", "seq_len"],
            suffixes=("_vanilla", "_compiled"),
        )
        merged_bwd["speedup"] = merged_bwd["backward_mean_ms_vanilla"] / merged_bwd["backward_mean_ms_compiled"]
        pivot_speedup_bwd = merged_bwd.pivot_table(
            index="d_model", columns="seq_len", values="speedup", aggfunc="first"
        )
        print(pivot_speedup_bwd.to_string())

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark compiled vs uncompiled attention"
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
        default="results/torch_compile_benchmarking/compiled_attention_benchmark.csv",
        help="Output CSV file path",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a GPU.")
        return

    # Check torch.compile availability
    if not hasattr(torch, "compile"):
        print("ERROR: torch.compile not available. Requires PyTorch 2.0+")
        return

    # Run benchmarks
    df = benchmark_compiled_vs_uncompiled(
        batch_size=args.batch_size,
        d_model_values=args.d_model,
        seq_len_values=args.seq_len,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
        output_path=args.output,
    )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("✓ Benchmarking complete")
    print(f"✓ Results saved to: {args.output}")
    print()
    print("Key observations:")
    print("  - torch.compile may show speedups on forward/backward passes")
    print("  - First iteration after compilation is slower (compilation overhead)")
    print("  - Speedups typically improve for larger workloads")
    print("  - Memory usage should be similar between compiled and vanilla")
    print()


if __name__ == "__main__":
    main()
