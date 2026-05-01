"""
Activation functions for neural networks.
"""

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float


class SiLU(nn.Module):
    """
    SiLU (Sigmoid Linear Unit) activation function, also known as Swish.

    SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

    This is a smooth activation function that is similar to ReLU but smooth at zero.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        """
        Apply SiLU activation function.

        Args:
            x: Input tensor of any shape

        Returns:
            Output tensor with SiLU applied element-wise
        """
        return x * torch.sigmoid(x)