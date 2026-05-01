#!/usr/bin/env python3
"""
Leaderboard-specific benchmark for FlashAttention-2.

Tests the exact configuration used in the leaderboard:
- seq_len = 16384
- d_model = 1024 (16 heads × 64 d_head)
- dtype = bfloat16
- batch_size = 1
- is_causal = True
"""

import torch
import triton.testing
import math

from flash_attention_triton import FlashAttentionTritonFunc
from flash_attention_triton_optimized import FlashAttentionTritonOptimizedFunc


def benchmark_version(version_name, fn_class, Q, K, V, grad_output, is_causal):
    """Benchmark a specific FlashAttention version."""
    print(f"\n{'='*80}")
    print(f"Testing: {version_name}")
    print(f"{'='*80}")

    def flash_attn_fn(Q, K, V, is_causal):
        return fn_class.apply(Q, K, V, is_causal)

    print("Warming up...")
    for _ in range(5):
        Q.grad = None
        K.grad = None
        V.grad = None
        output = flash_attn_fn(Q, K, V, is_causal)
        output.backward(grad_output)

    print("Benchmarking forward pass...")
    def forward_fn():
        return flash_attn_fn(Q, K, V, is_causal)

    forward_ms = triton.testing.do_bench(forward_fn, warmup=25, rep=100)
    print(f"  Forward time: {forward_ms:.3f} ms")

    print("Benchmarking backward pass...")
    Q.grad = None
    K.grad = None
    V.grad = None
    output = flash_attn_fn(Q, K, V, is_causal)

    def backward_fn():
        output.backward(grad_output, retain_graph=True)

    backward_ms = triton.testing.do_bench(backward_fn, warmup=25, rep=100)
    print(f"  Backward time: {backward_ms:.3f} ms")

    print("Benchmarking forward + backward...")
    def fwd_bwd_fn():
        Q.grad = None
        K.grad = None
        V.grad = None
        output = flash_attn_fn(Q, K, V, is_causal)
        output.backward(grad_output)

    fwd_bwd_ms = triton.testing.do_bench(fwd_bwd_fn, warmup=25, rep=100)
    print(f"  Forward + Backward time: {fwd_bwd_ms:.3f} ms")

    return {
        'forward_ms': forward_ms,
        'backward_ms': backward_ms,
        'fwd_bwd_ms': fwd_bwd_ms,
    }


def benchmark_leaderboard():
    """Benchmark the exact leaderboard configuration."""
    # Leaderboard configuration
    batch_size = 1
    seq_len = 16384
    d_model = 1024  # 16 heads × 64 d_head
    dtype = torch.bfloat16
    is_causal = True
    device = 'cuda'

    print("=" * 80)
    print("FlashAttention-2 Leaderboard Benchmark")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  d_model: {d_model}")
    print(f"  dtype: {dtype}")
    print(f"  is_causal: {is_causal}")
    print()

    # Generate random inputs
    torch.manual_seed(42)
    Q = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device=device, requires_grad=True)
    K = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device=device, requires_grad=True)
    V = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device=device, requires_grad=True)
    grad_output = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device=device)

    # Ensure contiguous
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    # Benchmark current version
    results_current = benchmark_version(
        "Current Implementation (with atomics)",
        FlashAttentionTritonFunc,
        Q, K, V, grad_output, is_causal
    )

    # Clear cache
    torch.cuda.empty_cache()

    # Benchmark optimized version
    results_optimized = benchmark_version(
        "Optimized Implementation (two-pass, autotune, early termination)",
        FlashAttentionTritonOptimizedFunc,
        Q, K, V, grad_output, is_causal
    )

    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\n{'Metric':<25} {'Current':<15} {'Optimized':<15} {'Speedup':<10}")
    print("-" * 80)

    fwd_speedup = results_current['forward_ms'] / results_optimized['forward_ms']
    bwd_speedup = results_current['backward_ms'] / results_optimized['backward_ms']
    total_speedup = results_current['fwd_bwd_ms'] / results_optimized['fwd_bwd_ms']

    print(f"{'Forward (ms)':<25} {results_current['forward_ms']:<15.3f} {results_optimized['forward_ms']:<15.3f} {fwd_speedup:<10.3f}x")
    print(f"{'Backward (ms)':<25} {results_current['backward_ms']:<15.3f} {results_optimized['backward_ms']:<15.3f} {bwd_speedup:<10.3f}x")
    print(f"{'Forward+Backward (ms)':<25} {results_current['fwd_bwd_ms']:<15.3f} {results_optimized['fwd_bwd_ms']:<15.3f} {total_speedup:<10.3f}x")
    print("=" * 80)

    return results_current, results_optimized


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires a GPU.")
        exit(1)

    benchmark_leaderboard()