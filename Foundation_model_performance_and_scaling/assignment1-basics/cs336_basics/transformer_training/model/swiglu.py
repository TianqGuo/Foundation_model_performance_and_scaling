"""
SwiGLU feed-forward network implementation.
"""

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float

from .linear import Linear
from .activations import SiLU


class SwiGLU(nn.Module):
    """
    SwiGLU (Swish Gated Linear Unit) feed-forward network.

    Implementation of the SwiGLU activation function used in modern LLMs like LLaMA.
    SwiGLU combines the SiLU (Swish) activation with a gating mechanism (GLU).

    The computation is: FFN(x) = W2(SiLU(W1*x) ⊙ W3*x)
    where ⊙ represents element-wise multiplication.

    Args:
        d_model: Input and output dimension
        d_ff: Hidden dimension (inner feed-forward layer size)
        device: Device to place the module on
        dtype: Data type for the module parameters
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        # Set d_ff to approximately (8/3) * d_model, rounded to nearest multiple of 64
        if d_ff is None:
            d_ff_approx = int((8 / 3) * d_model)
            d_ff = ((d_ff_approx + 63) // 64) * 64  # Round up to nearest multiple of 64

        self.d_model = d_model
        self.d_ff = d_ff

        # Three linear transformations (no bias by default in our Linear implementation)
        self.w1 = Linear(
            in_features=d_model,
            out_features=d_ff,
            device=device,
            dtype=dtype,
        )
        self.w2 = Linear(
            in_features=d_ff,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )
        self.w3 = Linear(
            in_features=d_model,
            out_features=d_ff,
            device=device,
            dtype=dtype,
        )

        # SiLU activation function
        self.silu = SiLU()

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """
        Forward pass of SwiGLU.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Output tensor of shape (..., d_model)
        """
        # SwiGLU(x) = W2(SiLU(W1*x) ⊙ W3*x)
        gate = self.silu(self.w1(x))  # SiLU(W1*x)
        up = self.w3(x)  # W3*x
        gated = gate * up  # Element-wise multiplication (⊙)
        output = self.w2(gated)  # W2(...)
        return output