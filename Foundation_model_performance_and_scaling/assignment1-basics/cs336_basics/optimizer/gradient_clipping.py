"""
Gradient clipping utilities.

Implements gradient clipping to prevent gradient explosion during training.
"""

import torch
from typing import Iterable


def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    Clip gradients by L2 norm to prevent gradient explosion.

    Given the gradient g for all parameters, compute its L2-norm ||g||2.
    If this norm is less than max_l2_norm, leave g unchanged.
    Otherwise, scale g down by a factor of max_l2_norm / (||g||2 + ε)
    where ε = 1e-6 for numerical stability.

    Args:
        parameters: Iterable of trainable parameters with .grad attributes
        max_l2_norm: Maximum allowed L2 norm for gradients

    The gradients of the parameters (parameter.grad) are modified in-place.
    """
    # Convert to list to allow multiple iterations
    param_list = list(parameters)

    # Filter parameters that have gradients
    params_with_grad = [p for p in param_list if p.grad is not None]

    if not params_with_grad:
        return  # No gradients to clip

    # Compute the L2 norm of all gradients combined
    # ||g||2 = sqrt(sum(||g_i||2^2)) where g_i is the gradient for parameter i
    total_norm_squared = 0.0
    for param in params_with_grad:
        param_norm_squared = param.grad.data.norm(dtype=torch.float32) ** 2
        total_norm_squared += param_norm_squared

    total_norm = total_norm_squared ** 0.5

    # Numerical stability epsilon (PyTorch default)
    eps = 1e-6

    # Compute clipping factor
    clip_factor = max_l2_norm / (total_norm + eps)

    # Only clip if the norm exceeds the maximum
    if clip_factor < 1.0:
        # Scale down all gradients by the clipping factor
        for param in params_with_grad:
            param.grad.data.mul_(clip_factor)