"""
Scaled Dot-Product Attention implementation.
"""

import torch
from torch import Tensor
from jaxtyping import Float

from .softmax import softmax


def scaled_dot_product_attention(
    Q: Float[Tensor, "... n_queries d_k"],
    K: Float[Tensor, "... n_keys d_k"],
    V: Float[Tensor, "... n_keys d_v"],
    mask: Float[Tensor, "... n_queries n_keys"] | None = None,
) -> Float[Tensor, "... n_queries d_v"]:
    """
    Scaled Dot-Product Attention as described in "Attention Is All You Need".

    Computes: Attention(Q, K, V) = softmax(QK^T / √d_k) V

    Args:
        Q: Query tensor of shape (..., n_queries, d_k)
        K: Key tensor of shape (..., n_keys, d_k)
        V: Value tensor of shape (..., n_keys, d_v)
        mask: Optional boolean mask of shape (..., n_queries, n_keys).
              True means attend, False means don't attend.

    Returns:
        Output tensor of shape (..., n_queries, d_v)
    """
    # Get the dimension of keys for scaling
    d_k = Q.shape[-1]

    # Compute attention scores: QK^T / √d_k
    # Q: (..., n_queries, d_k), K: (..., n_keys, d_k)
    # We want: (..., n_queries, n_keys)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    # Apply mask if provided
    if mask is not None:
        # Add a large negative value where mask is False to make softmax output ~0
        # We use -1e9 instead of -inf to avoid NaN issues
        scores = scores.masked_fill(~mask, -1e9)

    # Apply softmax to get attention weights
    # Shape: (..., n_queries, n_keys)
    attention_weights = softmax(scores, dim=-1)

    # Apply attention weights to values
    # attention_weights: (..., n_queries, n_keys)
    # V: (..., n_keys, d_v)
    # Result: (..., n_queries, d_v)
    output = torch.matmul(attention_weights, V)

    return output


class ScaledDotProductAttention(torch.nn.Module):
    """
    Scaled Dot-Product Attention module.

    This is a module wrapper around the scaled_dot_product_attention function
    for cases where you need a stateful module.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        Q: Float[Tensor, "... n_queries d_k"],
        K: Float[Tensor, "... n_keys d_k"],
        V: Float[Tensor, "... n_keys d_v"],
        mask: Float[Tensor, "... n_queries n_keys"] | None = None,
    ) -> Float[Tensor, "... n_queries d_v"]:
        """
        Forward pass of scaled dot-product attention.

        Args:
            Q: Query tensor of shape (..., n_queries, d_k)
            K: Key tensor of shape (..., n_keys, d_k)
            V: Value tensor of shape (..., n_keys, d_v)
            mask: Optional boolean mask of shape (..., n_queries, n_keys)

        Returns:
            Output tensor of shape (..., n_queries, d_v)
        """
        return scaled_dot_product_attention(Q, K, V, mask)