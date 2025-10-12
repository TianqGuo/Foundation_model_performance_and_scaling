"""
Rotary Position Embeddings (RoPE) implementation.

ALTERNATIVE: You can use torchtune.modules.RotaryPositionalEmbeddings
==================================================================

TorchTune provides a production-ready RoPE implementation that you can use instead:

    from torchtune.modules import RotaryPositionalEmbeddings

    # TorchTune interface:
    rope = RotaryPositionalEmbeddings(
        dim=d_k,            # embedding dimension per head
        max_seq_len=4096,   # maximum sequence length (default: 4096)
        base=10000          # base for geometric progression (default: 10000)
    )

    # Forward pass (slightly different interface):
    output = rope(x, input_pos=token_positions)

Key differences between our implementation and TorchTune's:

Our implementation:
- Parameters: theta, d_k, max_seq_len, device
- Forward: rope(x, token_positions)
- theta parameter directly controls the base frequency

TorchTune implementation:
- Parameters: dim, max_seq_len=4096, base=10000
- Forward: rope(x, input_pos=token_positions)
- base parameter corresponds to our theta parameter
- More optimized for production use
- Follows LLaMA reference implementation

Interface mapping:
    # Our version:
    rope = RotaryPositionalEmbedding(theta=10000.0, d_k=64, max_seq_len=512)
    output = rope(x, token_positions)

    # TorchTune equivalent:
    rope = torchtune.modules.RotaryPositionalEmbeddings(dim=64, max_seq_len=512, base=10000)
    output = rope(x, input_pos=token_positions)

Reasons to use our custom implementation:
1. Educational purposes - understanding RoPE mechanics
2. Explicit control over implementation details
3. Custom theta parameter handling
4. Assignment requirements

Reasons to use TorchTune's:
1. Production-tested and optimized
2. Follows LLaMA reference implementation
3. Better memory efficiency
4. Integrated with other TorchTune modules
5. Regular updates and bug fixes

Note: Both implementations should give mathematically equivalent results,
but TorchTune's may have better performance optimizations.
"""

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) implementation.

    RoPE applies rotations to query and key embeddings based on their position,
    allowing the model to understand relative positions between tokens.

    The rotation is applied pairwise to embedding dimensions, rotating pairs
    of elements by an angle θ_i,k = i / Θ^(2k-1)/d for position i and dimension pair k.

    Args:
        theta: Θ value for RoPE (typically 10000.0)
        d_k: Dimension of query and key vectors
        max_seq_len: Maximum sequence length that will be processed
        device: Device to store the buffers on
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | str | None = None,
    ):
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Ensure d_k is even (required for pairwise rotations)
        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even, got {d_k}")

        # Pre-compute the frequency values for each dimension pair
        # θ_k = 1 / Θ^(2k/d) for k ∈ {0, 1, ..., d/2-1}
        dim_pairs = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)
        freqs = 1.0 / (theta ** (dim_pairs / d_k))  # Shape: (d_k/2,)

        # Pre-compute angles for all positions up to max_seq_len
        # positions: (max_seq_len,), freqs: (d_k/2,) -> angles: (max_seq_len, d_k/2)
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        angles = torch.outer(positions, freqs)  # Shape: (max_seq_len, d_k/2)

        # Pre-compute cos and sin values
        cos_vals = torch.cos(angles)  # Shape: (max_seq_len, d_k/2)
        sin_vals = torch.sin(angles)  # Shape: (max_seq_len, d_k/2)

        # Register as buffers (not parameters, since these are fixed values)
        self.register_buffer("cos_vals", cos_vals, persistent=False)
        self.register_buffer("sin_vals", sin_vals, persistent=False)

    def _apply_rotary_embedding(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        cos: Float[Tensor, "seq_len d_k_half"],
        sin: Float[Tensor, "seq_len d_k_half"],
    ) -> Float[Tensor, "... seq_len d_k"]:
        """
        Apply rotary embedding to input tensor x using precomputed cos and sin values.

        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            cos: Cosine values of shape (seq_len, d_k/2)
            sin: Sine values of shape (seq_len, d_k/2)

        Returns:
            Tensor with rotary embeddings applied, same shape as input
        """
        # Split x into pairs: (x0, x1), (x2, x3), ..., (x_{d-2}, x_{d-1})
        x_even = x[..., ::2]  # Shape: (..., seq_len, d_k/2) - x0, x2, x4, ...
        x_odd = x[..., 1::2]  # Shape: (..., seq_len, d_k/2) - x1, x3, x5, ...

        # Apply rotation:
        # x'_even = cos * x_even - sin * x_odd
        # x'_odd = sin * x_even + cos * x_odd
        x_even_rot = cos * x_even - sin * x_odd
        x_odd_rot = sin * x_even + cos * x_odd

        # Interleave the rotated values back
        # Create output tensor with same shape as input
        x_rot = torch.empty_like(x)
        x_rot[..., ::2] = x_even_rot   # Place even indices
        x_rot[..., 1::2] = x_odd_rot   # Place odd indices

        return x_rot

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"],
    ) -> Float[Tensor, "... seq_len d_k"]:
        """
        Apply rotary position embeddings to input tensor.

        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            token_positions: Token positions of shape (..., seq_len)

        Returns:
            Tensor with RoPE applied, same shape as input
        """
        # Get the sequence length
        seq_len = x.shape[-2]

        # Select cos and sin values based on token positions
        # token_positions might have arbitrary batch dimensions, so we need to handle this carefully
        cos = self.cos_vals[token_positions]  # Shape: (..., seq_len, d_k/2)
        sin = self.sin_vals[token_positions]  # Shape: (..., seq_len, d_k/2)

        # Apply rotary embedding
        return self._apply_rotary_embedding(x, cos, sin)