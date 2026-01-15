#!/usr/bin/env python3
"""
Benchmarking script for FlashAttention-2 vs PyTorch attention.

This script compares the performance of FlashAttention-2 (Triton implementation)
with standard PyTorch attention across various configurations.

Usage:
    cd cs336_systems/flash_attention
    uv run python benchmark_flash_attention.py [--output OUTPUT_CSV]

Output:
    CSV file with benchmark results in results/flash_attention/
"""

import torch
import triton.testing
import pandas as pd
import argparse
from pathlib import Path
import math

from flash_attention_pytorch import FlashAttentionPyTorchFunc
from flash_attention_triton import FlashAttentionTritonFunc


def pytorch_attention(Q, K, V, is_causal=False):
    """Standard PyTorch attention implementation."""
    scale = 1.0 / math.sqrt(Q.shape[-1])
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

    if is_causal:
        N_q, N_k = scores.shape[-2], scores.shape[-1]
        mask = torch.triu(torch.ones(N_q, N_k, device=Q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))

    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)
    return output


def benchmark_forward(attention_fn, Q, K, V, is_causal=False):
    """Benchmark forward pass only."""
    def fn():
        _ = attention_fn(Q, K, V, is_causal)
        return

    # Use triton.testing.do_bench for accurate benchmarking
    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    return ms


def benchmark_backward(attention_fn, Q, K, V, grad_output, is_causal=False):
    """Benchmark backward pass only."""
    def fn():
        Q.grad = None
        K.grad = None
        V.grad = None
        output = attention_fn(Q, K, V, is_causal)
        output.backward(grad_output)
        return

    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    return ms


def benchmark_forward_backward(attention_fn, Q, K, V, grad_output, is_causal=False):
    """Benchmark end-to-end forward + backward pass."""
    def fn():
        Q.grad = None
        K.grad = None
        V.grad = None
        output = attention_fn(Q, K, V, is_causal)
        output.backward(grad_output)
        return

    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    return ms


def run_benchmark(seq_len, d_model, dtype, batch_size=1, is_causal=True):
    """
    Run benchmarks for a specific configuration.

    Args:
        seq_len: Sequence length
        d_model: Embedding dimension
        dtype: torch.bfloat16 or torch.float32
        batch_size: Batch size (default 1)
        is_causal: Whether to use causal masking (default True)

    Returns:
        Dictionary with benchmark results
    """
    device = torch.device('cuda')

    # Generate random inputs
    Q = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device=device, requires_grad=True)
    K = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device=device, requires_grad=True)
    V = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device=device, requires_grad=True)
    grad_output = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device=device)

    # Make contiguous for Triton
    Q_cont = Q.contiguous()
    K_cont = K.contiguous()
    V_cont = V.contiguous()

    results = {
        'seq_len': seq_len,
        'd_model': d_model,
        'dtype': str(dtype),
        'batch_size': batch_size,
        'is_causal': is_causal,
    }

    try:
        # Benchmark PyTorch attention
        print(f"  Benchmarking PyTorch attention (seq_len={seq_len}, d_model={d_model}, dtype={dtype})...")

        # Forward only
        pytorch_fwd_ms = benchmark_forward(pytorch_attention, Q, K, V, is_causal)
        results['pytorch_forward_ms'] = pytorch_fwd_ms

        # Backward only (need to run forward first to get output)
        Q.grad = None
        K.grad = None
        V.grad = None
        output = pytorch_attention(Q, K, V, is_causal)
        pytorch_bwd_ms = triton.testing.do_bench(
            lambda: output.backward(grad_output, retain_graph=True),
            warmup=25,
            rep=100
        )
        results['pytorch_backward_ms'] = pytorch_bwd_ms

        # Forward + Backward
        pytorch_fwd_bwd_ms = benchmark_forward_backward(pytorch_attention, Q, K, V, grad_output, is_causal)
        results['pytorch_fwd_bwd_ms'] = pytorch_fwd_bwd_ms

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f"  PyTorch OOM for seq_len={seq_len}, d_model={d_model}")
            results['pytorch_forward_ms'] = 'OOM'
            results['pytorch_backward_ms'] = 'OOM'
            results['pytorch_fwd_bwd_ms'] = 'OOM'
            torch.cuda.empty_cache()
        else:
            raise

    try:
        # Benchmark FlashAttention-2 (Triton)
        print(f"  Benchmarking FlashAttention-2 (seq_len={seq_len}, d_model={d_model}, dtype={dtype})...")

        # Create wrapper that applies the function
        def flash_attn_fn(Q, K, V, is_causal):
            return FlashAttentionTritonFunc.apply(Q, K, V, is_causal)

        # Forward only
        flash_fwd_ms = benchmark_forward(flash_attn_fn, Q_cont, K_cont, V_cont, is_causal)
        results['flash_forward_ms'] = flash_fwd_ms

        # Backward only
        Q_cont.grad = None
        K_cont.grad = None
        V_cont.grad = None
        output = flash_attn_fn(Q_cont, K_cont, V_cont, is_causal)
        flash_bwd_ms = triton.testing.do_bench(
            lambda: output.backward(grad_output, retain_graph=True),
            warmup=25,
            rep=100
        )
        results['flash_backward_ms'] = flash_bwd_ms

        # Forward + Backward
        flash_fwd_bwd_ms = benchmark_forward_backward(flash_attn_fn, Q_cont, K_cont, V_cont, grad_output, is_causal)
        results['flash_fwd_bwd_ms'] = flash_fwd_bwd_ms

        # Compute speedups
        if isinstance(pytorch_fwd_ms, (int, float)):
            results['forward_speedup'] = pytorch_fwd_ms / flash_fwd_ms
            results['backward_speedup'] = pytorch_bwd_ms / flash_bwd_ms
            results['fwd_bwd_speedup'] = pytorch_fwd_bwd_ms / flash_fwd_bwd_ms
        else:
            results['forward_speedup'] = 'N/A'
            results['backward_speedup'] = 'N/A'
            results['fwd_bwd_speedup'] = 'N/A'

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f"  FlashAttention OOM for seq_len={seq_len}, d_model={d_model}")
            results['flash_forward_ms'] = 'OOM'
            results['flash_backward_ms'] = 'OOM'
            results['flash_fwd_bwd_ms'] = 'OOM'
            results['forward_speedup'] = 'N/A'
            results['backward_speedup'] = 'N/A'
            results['fwd_bwd_speedup'] = 'N/A'
            torch.cuda.empty_cache()
        else:
            raise

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark FlashAttention-2 vs PyTorch attention'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../../results/flash_attention/flash_benchmarking.csv',
        help='Output CSV file path (default: ../../results/flash_attention/flash_benchmarking.csv)'
    )
    args = parser.parse_args()

    # Check for CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires a GPU.")
        return

    print(f"Running benchmarks on {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")

    # Benchmark configurations
    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    d_models = [16, 32, 64, 128]
    dtypes = [torch.bfloat16, torch.float32]
    batch_size = 1
    is_causal = True

    # Run benchmarks
    all_results = []
    total_configs = len(seq_lengths) * len(d_models) * len(dtypes)
    current = 0

    for seq_len in seq_lengths:
        for d_model in d_models:
            for dtype in dtypes:
                current += 1
                print(f"\n[{current}/{total_configs}] Running benchmark:")
                print(f"  seq_len={seq_len}, d_model={d_model}, dtype={dtype}")

                try:
                    results = run_benchmark(seq_len, d_model, dtype, batch_size, is_causal)
                    all_results.append(results)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    # Still add the config with error markers
                    all_results.append({
                        'seq_len': seq_len,
                        'd_model': d_model,
                        'dtype': str(dtype),
                        'batch_size': batch_size,
                        'is_causal': is_causal,
                        'pytorch_forward_ms': 'ERROR',
                        'pytorch_backward_ms': 'ERROR',
                        'pytorch_fwd_bwd_ms': 'ERROR',
                        'flash_forward_ms': 'ERROR',
                        'flash_backward_ms': 'ERROR',
                        'flash_fwd_bwd_ms': 'ERROR',
                        'forward_speedup': 'N/A',
                        'backward_speedup': 'N/A',
                        'fwd_bwd_speedup': 'N/A',
                    })
                    torch.cuda.empty_cache()

    # Save results to CSV
    df = pd.DataFrame(all_results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n{'='*80}")
    print(f"Benchmark complete! Results saved to {output_path}")
    print(f"{'='*80}")

    # Print summary statistics
    print("\nSummary Statistics (excluding OOM/ERROR cases):")
    print("-" * 80)

    # Filter valid numeric results
    df_valid = df[
        df['forward_speedup'].apply(lambda x: isinstance(x, (int, float)))
    ]

    if len(df_valid) > 0:
        print(f"Forward speedup:  mean={df_valid['forward_speedup'].mean():.2f}x, "
              f"max={df_valid['forward_speedup'].max():.2f}x")
        print(f"Backward speedup: mean={df_valid['backward_speedup'].mean():.2f}x, "
              f"max={df_valid['backward_speedup'].max():.2f}x")
        print(f"Fwd+Bwd speedup:  mean={df_valid['fwd_bwd_speedup'].mean():.2f}x, "
              f"max={df_valid['fwd_bwd_speedup'].max():.2f}x")
    else:
        print("No valid results to summarize.")


if __name__ == '__main__':
    main()