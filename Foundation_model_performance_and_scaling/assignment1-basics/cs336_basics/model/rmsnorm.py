"""
Root Mean Square Layer Normalization (RMSNorm) module for the CS336 Transformer implementation.

This module implements RMSNorm as described in Zhang and Sennrich (2019), which is used
in modern language models like LLaMA instead of traditional LayerNorm.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm normalizes activations by scaling them by the root mean square of the vector,
    followed by an elementwise affine transformation using learnable gain parameters.
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        """
        Initialize the RMSNorm module.

        Args:
            d_model (int): Hidden dimension of the model
            eps (float): Epsilon value for numerical stability (default: 1e-5)
            device (torch.device | None): Device to store parameters on
            dtype (torch.dtype | None): Data type of the parameters
        """
        super().__init__()

        # Store dimensions and hyperparameters
        self.d_model = d_model
        self.eps = eps

        # Create learnable gain parameters (one for each dimension)
        # Initialize to 1 as specified in the requirements
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model)

        Returns:
            torch.Tensor: Normalized tensor of the same shape as input
        """
        # Store original dtype for later conversion back
        in_dtype = x.dtype

        # Upcast to float32 to prevent overflow when squaring
        x = x.to(torch.float32)

        # Compute RMS: sqrt(mean(x^2) + eps)
        # We compute this over the last dimension (d_model)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

        # Normalize by RMS and apply learnable gain
        # x / RMS(x) * g_i
        result = (x / rms) * self.weight

        # Return in the original dtype
        return result.to(in_dtype)

    def extra_repr(self) -> str:
        """Return extra representation string for debugging."""
        return f"d_model={self.d_model}, eps={self.eps}"