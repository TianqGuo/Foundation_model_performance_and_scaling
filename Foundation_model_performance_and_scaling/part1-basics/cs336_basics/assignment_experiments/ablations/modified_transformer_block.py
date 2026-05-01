"""
Modified Transformer Block for ablation experiments.

This module provides configurable Transformer blocks that support various ablations:
- No layer normalization (layer_norm_ablation)
- Post-norm architecture (pre_norm_ablation)
- No position embeddings (no_pos_emb)
- SiLU FFN instead of SwiGLU (swiglu_ablation)
"""

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int

from cs336_basics.transformer_training.model.multihead_attention import MultiHeadSelfAttention
from cs336_basics.transformer_training.model.rmsnorm import RMSNorm
from cs336_basics.transformer_training.model.swiglu import SwiGLU
from cs336_basics.transformer_training.model.linear import Linear
from cs336_basics.transformer_training.model.activations import SiLU


class SiLUFFN(nn.Module):
    """
    Feed-forward network with SiLU activation (no gating).

    FFN_SiLU(x) = W2 * SiLU(W1 * x)

    Args:
        d_model: Input/output dimensionality
        d_ff: Inner layer dimensionality (typically 4 * d_model for parameter matching)
        device: Device to place the module on
        dtype: Data type for the module parameters
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.silu = SiLU()
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """Apply SiLU feed-forward network."""
        return self.w2(self.silu(self.w1(x)))


class FlexibleTransformerBlock(nn.Module):
    """
    Configurable Transformer block for ablation experiments.

    Supports:
    - Pre-norm vs. Post-norm architecture
    - With or without layer normalization
    - SwiGLU vs. SiLU feed-forward networks
    - With or without position embeddings

    Args:
        d_model: Dimensionality of the model
        num_heads: Number of attention heads
        d_ff: Dimensionality of the feed-forward inner layer
        max_seq_len: Maximum sequence length for RoPE precomputation
        rope_theta: RoPE theta parameter
        eps: Epsilon for RMSNorm
        use_layer_norm: Whether to use layer normalization (default: True)
        post_norm: Whether to use post-norm instead of pre-norm (default: False)
        use_swiglu: Whether to use SwiGLU or SiLU FFN (default: True)
        use_rope: Whether to use RoPE position embeddings (default: True)
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
        use_layer_norm: bool = True,
        post_norm: bool = False,
        use_swiglu: bool = True,
        use_rope: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.use_layer_norm = use_layer_norm
        self.post_norm = post_norm
        self.use_rope = use_rope

        # Layer normalization (may be disabled for ablation)
        if use_layer_norm:
            self.ln1 = RMSNorm(d_model=d_model, eps=eps, device=device, dtype=dtype)
            self.ln2 = RMSNorm(d_model=d_model, eps=eps, device=device, dtype=dtype)
        else:
            self.ln1 = None
            self.ln2 = None

        # Multi-head self-attention
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            device=device,
            dtype=dtype,
        )

        # Feed-forward network (SwiGLU or SiLU)
        if use_swiglu:
            self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        else:
            self.ffn = SiLUFFN(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
    ) -> Float[Tensor, "... seq_len d_model"]:
        """
        Forward pass of the Transformer block.

        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            token_positions: Token positions for RoPE (ignored if use_rope=False)

        Returns:
            Output tensor of shape (..., seq_len, d_model)
        """
        # Disable position embeddings if requested
        pos = token_positions if self.use_rope else None

        if self.post_norm:
            # Post-norm: z = RMSNorm(x + Attention(x))
            attn_output = self.attn(x, token_positions=pos)
            x = x + attn_output
            if self.ln1 is not None:
                x = self.ln1(x)

            # y = RMSNorm(z + FFN(z))
            ffn_output = self.ffn(x)
            x = x + ffn_output
            if self.ln2 is not None:
                x = self.ln2(x)
        else:
            # Pre-norm: x = x + Attention(RMSNorm(x))
            if self.ln1 is not None:
                norm1_output = self.ln1(x)
            else:
                norm1_output = x

            attn_output = self.attn(norm1_output, token_positions=pos)
            x = x + attn_output

            # x = x + FFN(RMSNorm(x))
            if self.ln2 is not None:
                norm2_output = self.ln2(x)
            else:
                norm2_output = x

            ffn_output = self.ffn(norm2_output)
            x = x + ffn_output

        return x
