"""
NVTX-annotated scaled dot product attention for profiling.

This module provides an annotated version of scaled_dot_product_attention
that adds NVTX ranges to track different components in the profiler.
"""

import torch
import torch.cuda.nvtx as nvtx


@nvtx.range("scaled_dot_product_attention")
def annotated_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention with NVTX annotations for profiling.

    This is a drop-in replacement for the original scaled_dot_product_attention
    that adds NVTX ranges to track:
    - Computing attention scores (Q @ K^T)
    - Applying mask (optional)
    - Computing softmax
    - Final matmul (attention @ V)

    Args:
        query: Query tensor of shape (batch, num_heads, seq_len, d_head)
        key: Key tensor of shape (batch, num_heads, seq_len, d_head)
        value: Value tensor of shape (batch, num_heads, seq_len, d_head)
        mask: Optional attention mask

    Returns:
        Output tensor of shape (batch, num_heads, seq_len, d_head)
    """
    # Get d_k for scaling
    d_k = query.size(-1)

    # Compute attention scores: Q @ K^T / sqrt(d_k)
    with nvtx.range("computing_attention_scores"):
        # Shape: (batch, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)

    # Apply mask if provided
    if mask is not None:
        with nvtx.range("applying_mask"):
            scores = scores.masked_fill(mask == 0, float('-inf'))

    # Compute softmax
    with nvtx.range("computing_softmax"):
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)

    # Final matmul: attention @ V
    with nvtx.range("final_matmul"):
        output = torch.matmul(attention_weights, value)

    return output
