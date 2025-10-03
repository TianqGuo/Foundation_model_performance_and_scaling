"""
Linear transformation module for the CS336 Transformer implementation.

This module implements a linear transformation layer without bias, following
the requirements specified in the assignment.
"""

import torch
import torch.nn as nn
from einops import einsum


class Linear(nn.Module):
    """
    Linear transformation module without bias.

    Performs the transformation y = Wx where W is a learnable weight matrix.
    This follows most modern LLM architectures that omit bias terms.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        """
        Initialize the Linear module.

        Args:
            in_features (int): Size of each input sample (final dimension)
            out_features (int): Size of each output sample (final dimension)
            device (torch.device | None): Device to store parameters on
            dtype (torch.dtype | None): Data type of the parameters
        """
        super().__init__()

        # Store dimensions
        self.in_features = in_features
        self.out_features = out_features

        # Create weight parameter with proper shape (out_features, in_features)
        # This is W (not W^T) for memory ordering reasons as specified
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        # Initialize weights using truncated normal distribution
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using truncated normal distribution."""
        # Calculate standard deviation: σ² = 2/(d_in + d_out)
        fan_in = self.in_features
        fan_out = self.out_features
        std = (2.0 / (fan_in + fan_out)) ** 0.5

        # Initialize with truncated normal: N(μ=0, σ²) truncated at [-3σ, 3σ]
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply linear transformation to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., in_features)

        Returns:
            torch.Tensor: Output tensor of shape (..., out_features)
        """
        # Use einsum for self-documenting matrix multiplication
        # This handles arbitrary leading batch dimensions
        print(x.shape)
        print(self.weight.shape)
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

    def extra_repr(self) -> str:
        """Return extra representation string for debugging."""
        return f"in_features={self.in_features}, out_features={self.out_features}, bias=False"