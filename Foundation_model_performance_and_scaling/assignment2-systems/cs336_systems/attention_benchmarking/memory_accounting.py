"""
Memory accounting analysis for attention mechanism.

This script computes the theoretical memory footprint of naive attention
based on the equations from Assignment 1.

Usage:
    # Analyze specific configuration
    python -m cs336_systems.attention_benchmarking.memory_accounting \
        --batch-size 8 --seq-len 8192 --d-model 64

    # Analyze smallest OOM configuration
    python -m cs336_systems.attention_benchmarking.memory_accounting \
        --batch-size 8 --seq-len 8192 --d-model 16
"""

import argparse


def bytes_to_mb(bytes_val: int) -> float:
    """Convert bytes to MB."""
    return bytes_val / (1024 ** 2)


def compute_attention_memory(
    batch_size: int,
    seq_len: int,
    d_model: int,
    dtype_size: int = 4,  # FP32 = 4 bytes
) -> dict:
    """
    Compute memory usage for naive attention implementation.

    Memory breakdown:
    1. Inputs (Q, K, V): 3 × batch × seq_len × d_model
    2. Attention scores (QK^T): batch × seq_len × seq_len
    3. Attention weights (after softmax): batch × seq_len × seq_len
    4. Output: batch × seq_len × d_model

    For backward pass, we need to store:
    - Attention scores for softmax gradient
    - Attention weights for matmul gradient
    - Gradients for Q, K, V

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        d_model: Model/head dimension
        dtype_size: Size of data type in bytes (4 for FP32)

    Returns:
        Dictionary with memory accounting details
    """
    # Forward pass memory
    inputs_memory = 3 * batch_size * seq_len * d_model * dtype_size  # Q, K, V
    scores_memory = batch_size * seq_len * seq_len * dtype_size  # QK^T
    attn_weights_memory = batch_size * seq_len * seq_len * dtype_size  # softmax(scores)
    output_memory = batch_size * seq_len * d_model * dtype_size  # final output

    forward_memory = inputs_memory + scores_memory + attn_weights_memory + output_memory

    # Backward pass additional memory
    # Need to store gradients for Q, K, V (same size as inputs)
    gradients_memory = 3 * batch_size * seq_len * d_model * dtype_size

    # Need to store attention scores and weights for backward computation
    saved_for_backward = scores_memory + attn_weights_memory

    backward_memory = gradients_memory + saved_for_backward

    # Total memory (forward + backward)
    total_memory = forward_memory + backward_memory

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "d_model": d_model,
        "dtype_size": dtype_size,
        # Forward pass components (bytes)
        "inputs_bytes": inputs_memory,
        "scores_bytes": scores_memory,
        "attn_weights_bytes": attn_weights_memory,
        "output_bytes": output_memory,
        "forward_total_bytes": forward_memory,
        # Backward pass components (bytes)
        "gradients_bytes": gradients_memory,
        "saved_for_backward_bytes": saved_for_backward,
        "backward_additional_bytes": backward_memory,
        # Total
        "total_bytes": total_memory,
        # MB conversions
        "inputs_mb": bytes_to_mb(inputs_memory),
        "scores_mb": bytes_to_mb(scores_memory),
        "attn_weights_mb": bytes_to_mb(attn_weights_memory),
        "output_mb": bytes_to_mb(output_memory),
        "forward_total_mb": bytes_to_mb(forward_memory),
        "gradients_mb": bytes_to_mb(gradients_memory),
        "saved_for_backward_mb": bytes_to_mb(saved_for_backward),
        "backward_additional_mb": bytes_to_mb(backward_memory),
        "total_mb": bytes_to_mb(total_memory),
    }


def analyze_seq_len_scaling(
    batch_size: int,
    d_model: int,
    seq_lens: list = None,
) -> None:
    """
    Analyze how memory scales with sequence length.

    Args:
        batch_size: Batch size
        d_model: Model dimension
        seq_lens: List of sequence lengths to analyze
    """
    if seq_lens is None:
        seq_lens = [256, 1024, 4096, 8192, 16384]

    print("\n" + "=" * 80)
    print("MEMORY SCALING WITH SEQUENCE LENGTH")
    print("=" * 80)
    print(f"batch_size={batch_size}, d_model={d_model}")
    print()
    print(f"{'seq_len':>8} | {'Scores (MB)':>12} | {'Total (MB)':>12} | {'Scores / Total':>15}")
    print("-" * 60)

    for seq_len in seq_lens:
        stats = compute_attention_memory(batch_size, seq_len, d_model)
        scores_mb = stats["scores_mb"]
        total_mb = stats["total_mb"]
        ratio = scores_mb / total_mb * 100

        print(f"{seq_len:>8} | {scores_mb:>12.2f} | {total_mb:>12.2f} | {ratio:>14.1f}%")

    print()
    print("Observation: The seq_len² attention scores dominate memory for long sequences.")
    print()


def print_detailed_accounting(stats: dict) -> None:
    """Print detailed memory accounting."""
    print("\n" + "=" * 80)
    print("DETAILED MEMORY ACCOUNTING")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  batch_size = {stats['batch_size']}")
    print(f"  seq_len = {stats['seq_len']}")
    print(f"  d_model = {stats['d_model']}")
    print(f"  dtype = FP32 ({stats['dtype_size']} bytes)")
    print()

    print("Forward Pass Memory:")
    print("-" * 80)
    print(f"  Inputs (Q, K, V):        {stats['inputs_mb']:>10.2f} MB")
    print(f"    = 3 × {stats['batch_size']} × {stats['seq_len']} × {stats['d_model']} × {stats['dtype_size']}")
    print(f"    = {stats['inputs_bytes']:,} bytes")
    print()
    print(f"  Attention Scores (QK^T): {stats['scores_mb']:>10.2f} MB")
    print(f"    = {stats['batch_size']} × {stats['seq_len']} × {stats['seq_len']} × {stats['dtype_size']}")
    print(f"    = {stats['scores_bytes']:,} bytes")
    print()
    print(f"  Attention Weights:       {stats['attn_weights_mb']:>10.2f} MB")
    print(f"    = {stats['batch_size']} × {stats['seq_len']} × {stats['seq_len']} × {stats['dtype_size']}")
    print(f"    = {stats['attn_weights_bytes']:,} bytes")
    print()
    print(f"  Output:                  {stats['output_mb']:>10.2f} MB")
    print(f"    = {stats['batch_size']} × {stats['seq_len']} × {stats['d_model']} × {stats['dtype_size']}")
    print(f"    = {stats['output_bytes']:,} bytes")
    print()
    print(f"  Forward Total:           {stats['forward_total_mb']:>10.2f} MB")
    print()

    print("Backward Pass Additional Memory:")
    print("-" * 80)
    print(f"  Gradients (dQ, dK, dV):  {stats['gradients_mb']:>10.2f} MB")
    print(f"    = 3 × {stats['batch_size']} × {stats['seq_len']} × {stats['d_model']} × {stats['dtype_size']}")
    print(f"    = {stats['gradients_bytes']:,} bytes")
    print()
    print(f"  Saved for backward:      {stats['saved_for_backward_mb']:>10.2f} MB")
    print(f"    (scores + attn_weights for gradient computation)")
    print()
    print(f"  Backward Total:          {stats['backward_additional_mb']:>10.2f} MB")
    print()

    print("Total Memory Requirement:")
    print("-" * 80)
    print(f"  TOTAL:                   {stats['total_mb']:>10.2f} MB")
    print(f"                           {stats['total_bytes']:,} bytes")
    print()

    # Breakdown percentages
    scores_pct = stats['scores_mb'] / stats['total_mb'] * 100
    attn_weights_pct = stats['attn_weights_mb'] / stats['total_mb'] * 100
    seq_len_squared_pct = scores_pct + attn_weights_pct

    print("Memory Breakdown:")
    print("-" * 80)
    print(f"  seq_len² components:     {seq_len_squared_pct:>5.1f}% "
          f"(scores + attn_weights)")
    print(f"  Other components:        {100 - seq_len_squared_pct:>5.1f}% "
          f"(inputs + outputs + gradients)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze memory usage of naive attention"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        required=True,
        help="Sequence length",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        required=True,
        help="Head/model dimension",
    )
    parser.add_argument(
        "--dtype-size",
        type=int,
        default=4,
        help="Size of data type in bytes (default: 4 for FP32)",
    )

    args = parser.parse_args()

    # Compute memory accounting
    stats = compute_attention_memory(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        dtype_size=args.dtype_size,
    )

    # Print detailed accounting
    print_detailed_accounting(stats)

    # Show scaling analysis
    analyze_seq_len_scaling(
        batch_size=args.batch_size,
        d_model=args.d_model,
    )

    # Mitigation strategies
    print("=" * 80)
    print("MITIGATION STRATEGIES TO ELIMINATE seq_len² MEMORY COST")
    print("=" * 80)
    print()
    print("1. FlashAttention-2 (Tiled/Streaming Attention):")
    print("   - Compute attention in tiles, never materializing full seq_len² matrix")
    print("   - Store only O(seq_len) instead of O(seq_len²)")
    print("   - Recompute attention scores during backward (time vs memory tradeoff)")
    print()
    print("2. Memory-Efficient Attention:")
    print("   - Similar tiling approach")
    print("   - Break computation into blocks that fit in SRAM")
    print()
    print("3. Gradient Checkpointing:")
    print("   - Don't store activations, recompute during backward")
    print("   - Reduces memory at cost of 1 extra forward pass")
    print()
    print("4. Sparse Attention Patterns:")
    print("   - Only compute attention for subset of positions")
    print("   - Reduces from O(seq_len²) to O(seq_len × pattern_size)")
    print()

    # Calculate percentage for benefits
    seq_len_squared_pct = (stats['scores_mb'] + stats['attn_weights_mb']) / stats['total_mb'] * 100

    print("Benefits of FlashAttention-2:")
    print(f"  Current memory: {stats['total_mb']:.2f} MB")
    print(f"  With FlashAttention-2: ~{stats['total_mb'] - stats['scores_mb'] - stats['attn_weights_mb']:.2f} MB")
    print(f"  Memory savings: ~{stats['scores_mb'] + stats['attn_weights_mb']:.2f} MB "
          f"({seq_len_squared_pct:.1f}%)")
    print()


if __name__ == "__main__":
    main()
