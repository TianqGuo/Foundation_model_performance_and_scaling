"""
Transformer Block implementation with pre-norm architecture.
"""

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int

from .multihead_attention import MultiHeadSelfAttention
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block implementation.

    Following the pre-norm architecture, each sublayer follows the pattern:
    output = x + sublayer(LayerNorm(x))

    The block consists of:
    1. Multi-head self-attention sublayer with causal masking and RoPE
    2. Position-wise feed-forward network (SwiGLU) sublayer

    Args:
        d_model: Dimensionality of the model (input/output dimension)
        num_heads: Number of attention heads
        d_ff: Dimensionality of the feed-forward inner layer
        max_seq_len: Maximum sequence length for RoPE precomputation
        rope_theta: RoPE theta parameter (default: 10000.0)
        eps: Epsilon for RMSNorm (default: 1e-5)
        device: Device to place the module on
        dtype: Data type for the module parameters
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 4096,
        rope_theta: float = 10000.0,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # First sublayer: Multi-head self-attention
        self.ln1 = RMSNorm(d_model=d_model, eps=eps, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            device=device,
            dtype=dtype,
        )

        # Second sublayer: Feed-forward network
        self.ln2 = RMSNorm(d_model=d_model, eps=eps, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
    ) -> Float[Tensor, "... seq_len d_model"]:
        """
        Forward pass of the Transformer block.

        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            token_positions: Token positions for RoPE, shape (..., seq_len).
                           If None, uses range(seq_len)

        Returns:
            Output tensor of shape (..., seq_len, d_model)
        """
        # First sublayer: Multi-head self-attention with residual connection
        # y = x + MultiHeadSelfAttention(RMSNorm(x))
        norm1_output = self.ln1(x)
        attn_output = self.attn(norm1_output, token_positions=token_positions)
        x = x + attn_output

        # Second sublayer: Feed-forward network with residual connection
        # z = y + SwiGLU(RMSNorm(y))
        norm2_output = self.ln2(x)
        ffn_output = self.ffn(norm2_output)
        x = x + ffn_output

        return x