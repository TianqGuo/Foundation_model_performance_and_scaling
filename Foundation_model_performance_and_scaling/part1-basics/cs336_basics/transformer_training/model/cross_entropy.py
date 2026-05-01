"""
Cross-entropy loss implementation with numerical stability.
"""

import torch
from torch import Tensor
from jaxtyping import Float, Int


def cross_entropy(
    inputs: Float[Tensor, "... vocab_size"],
    targets: Int[Tensor, "..."],
) -> Float[Tensor, ""]:
    """
    Compute cross-entropy loss with numerical stability.

    Implements: ℓ = -log(softmax(inputs)[targets])

    Uses numerical stability techniques:
    1. Subtract max for numerical stability in softmax
    2. Use log-sum-exp trick to cancel out log and exp
    3. Simplify to: -inputs[targets] + log_sum_exp(inputs)

    Args:
        inputs: Logits tensor of shape (..., vocab_size)
        targets: Target indices of shape (...), values in [0, vocab_size-1]

    Returns:
        Scalar tensor with average cross-entropy loss across all batch dimensions
    """
    # Get the shape information
    *batch_dims, vocab_size = inputs.shape

    # Flatten batch dimensions for easier processing
    # inputs: (..., vocab_size) -> (batch_total, vocab_size)
    inputs_flat = inputs.view(-1, vocab_size)
    targets_flat = targets.view(-1)

    # Numerical stability: subtract max along vocab dimension
    # This prevents overflow in exp() while keeping softmax unchanged
    max_vals = torch.max(inputs_flat, dim=1, keepdim=True)[0]  # (batch_total, 1)
    inputs_shifted = inputs_flat - max_vals  # (batch_total, vocab_size)

    # Compute log_sum_exp for the denominator of softmax
    # log(sum(exp(inputs_shifted))) = log(sum(exp(inputs - max)))
    log_sum_exp = torch.log(torch.sum(torch.exp(inputs_shifted), dim=1))  # (batch_total,)

    # Get the logits for the target indices
    # This is the numerator of softmax: inputs[targets] - max_vals
    target_logits = inputs_flat.gather(1, targets_flat.unsqueeze(1)).squeeze(1)  # (batch_total,)
    target_logits_shifted = target_logits - max_vals.squeeze(1)  # (batch_total,)

    # Cross-entropy formula with cancellation:
    # -log(softmax(inputs)[targets]) = -log(exp(inputs[targets] - max) / sum(exp(inputs - max)))
    #                                = -(inputs[targets] - max) + log(sum(exp(inputs - max)))
    #                                = -target_logits_shifted + log_sum_exp
    cross_entropy_losses = -target_logits_shifted + log_sum_exp  # (batch_total,)

    # Return average loss across all batch dimensions
    return cross_entropy_losses.mean()


def perplexity(cross_entropy_loss: Float[Tensor, ""]) -> Float[Tensor, ""]:
    """
    Compute perplexity from cross-entropy loss.

    Args:
        cross_entropy_loss: Average cross-entropy loss over sequence

    Returns:
        Perplexity = exp(cross_entropy_loss)
    """
    return torch.exp(cross_entropy_loss)