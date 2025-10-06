"""
Multi-Head Self-Attention implementation with causal masking and RoPE.
"""

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int

from .linear import Linear
from .attention import scaled_dot_product_attention
from .rope import RotaryPositionalEmbedding


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with causal masking and RoPE.

    Implements the multi-head self-attention mechanism from "Attention Is All You Need"
    with causal masking for autoregressive language modeling and RoPE for positional encoding.

    Args:
        d_model: Dimensionality of the model (input/output dimension)
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length for RoPE precomputation
        rope_theta: RoPE theta parameter (default: 10000.0)
        device: Device to place the module on
        dtype: Data type for the module parameters
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 4096,
        rope_theta: float = 10000.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads  # dk = dv = d_model / num_heads
        self.max_seq_len = max_seq_len

        # Query, Key, Value projection matrices
        self.q_proj = Linear(
            in_features=d_model,
            out_features=d_model,  # num_heads * d_head = d_model
            device=device,
            dtype=dtype,
        )
        self.k_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )
        self.v_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

        # Output projection
        self.output_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

        # RoPE for positional encoding (applied to Q and K, not V)
        self.rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_k=self.d_head,
            max_seq_len=max_seq_len,
            device=device,
        )

    def _create_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Float[Tensor, "seq_len seq_len"]:
        """
        Create a causal (lower triangular) mask for autoregressive attention.

        Args:
            seq_len: Sequence length
            device: Device to create the mask on
            dtype: Data type for the mask

        Returns:
            Boolean mask of shape (seq_len, seq_len) where True means attend
        """
        # Create lower triangular mask (True for positions that can attend)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        return mask

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
    ) -> Float[Tensor, "... seq_len d_model"]:
        """
        Forward pass of multi-head self-attention.

        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            token_positions: Token positions for RoPE, shape (..., seq_len).
                           If None, uses range(seq_len)

        Returns:
            Output tensor of shape (..., seq_len, d_model)
        """
        # Get batch dimensions and sequence length
        *batch_dims, seq_len, d_model = x.shape
        batch_shape = batch_dims

        # Generate token positions if not provided
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
            # Expand to match batch dimensions
            for _ in batch_dims:
                token_positions = token_positions.unsqueeze(0)
            token_positions = token_positions.expand(*batch_shape, seq_len)

        # Linear projections: Q, K, V
        Q = self.q_proj(x)  # (..., seq_len, d_model)
        K = self.k_proj(x)  # (..., seq_len, d_model)
        V = self.v_proj(x)  # (..., seq_len, d_model)

        # Reshape for multi-head attention
        # From (..., seq_len, d_model) to (..., seq_len, num_heads, d_head)
        Q = Q.view(*batch_shape, seq_len, self.num_heads, self.d_head)
        K = K.view(*batch_shape, seq_len, self.num_heads, self.d_head)
        V = V.view(*batch_shape, seq_len, self.num_heads, self.d_head)

        # Apply RoPE to Q and K before transposing
        # Apply RoPE to each head separately
        Q_rope_list = []
        K_rope_list = []

        for head_idx in range(self.num_heads):
            # Extract Q and K for this head: (..., seq_len, d_head)
            Q_head = Q[..., head_idx, :]
            K_head = K[..., head_idx, :]

            # Apply RoPE to this head
            Q_head_rope = self.rope(Q_head, token_positions)
            K_head_rope = self.rope(K_head, token_positions)

            Q_rope_list.append(Q_head_rope)
            K_rope_list.append(K_head_rope)

        # Stack the heads back together
        Q = torch.stack(Q_rope_list, dim=-2)  # (..., seq_len, num_heads, d_head)
        K = torch.stack(K_rope_list, dim=-2)  # (..., seq_len, num_heads, d_head)

        # Transpose to (..., num_heads, seq_len, d_head) for efficient attention computation
        Q = Q.transpose(-3, -2)  # (..., num_heads, seq_len, d_head)
        K = K.transpose(-3, -2)  # (..., num_heads, seq_len, d_head)
        V = V.transpose(-3, -2)  # (..., num_heads, seq_len, d_head)

        # Create causal mask
        causal_mask = self._create_causal_mask(seq_len, x.device, x.dtype)

        # Expand causal mask to match batch and head dimensions
        # (seq_len, seq_len) -> (..., num_heads, seq_len, seq_len)
        expanded_mask = causal_mask
        for _ in range(len(batch_shape) + 1):  # +1 for num_heads dimension
            expanded_mask = expanded_mask.unsqueeze(0)
        expanded_mask = expanded_mask.expand(*batch_shape, self.num_heads, seq_len, seq_len)

        # Apply scaled dot-product attention
        attn_output = scaled_dot_product_attention(Q, K, V, mask=expanded_mask)
        # Shape: (..., num_heads, seq_len, d_head)

        # Transpose back to (..., seq_len, num_heads, d_head)
        attn_output = attn_output.transpose(-3, -2)

        # Concatenate heads: (..., seq_len, num_heads, d_head) -> (..., seq_len, d_model)
        attn_output = attn_output.contiguous().view(*batch_shape, seq_len, d_model)

        # Apply output projection
        output = self.output_proj(attn_output)

        return output