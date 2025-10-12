"""
AdamW optimizer implementation following Algorithm 1 from Loshchilov and Hutter [2019].
"""

import math
import torch
from torch.optim.optimizer import Optimizer
from typing import Any, Dict, Optional


class AdamW(Optimizer):
    """
    AdamW optimizer implementation.

    Implements AdamW as described in Algorithm 1 of Loshchilov and Hutter [2019]:
    "Decoupled Weight Decay Regularization"

    AdamW modifies Adam by decoupling weight decay from the gradient update,
    which improves regularization and generalization.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (α in the paper, default: 1e-3)
        betas: Tuple of (β1, β2) coefficients for moment estimates (default: (0.9, 0.999))
        eps: Small value for numerical stability (ϵ in the paper, default: 1e-8)
        weight_decay: Weight decay rate (λ in the paper, default: 0.01)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss

        Returns:
            The loss value if closure is provided, otherwise None
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Get gradient
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')

                # Initialize state if needed
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # Initialize first moment vector (m)
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Initialize second moment vector (v)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # Increment step counter
                state['step'] += 1
                step = state['step']

                # Update first moment estimate: m ← β1*m + (1 - β1)*g
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update second moment estimate: v ← β2*v + (1 - β2)*g²
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias correction terms
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # Compute adjusted learning rate: αt ← α * sqrt(1 - β2^t) / (1 - β1^t)
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Update parameters: θ ← θ - αt * m / (sqrt(v) + ϵ)
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Apply weight decay: θ ← θ - α*λ*θ
                if group['weight_decay'] > 0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])

        return loss