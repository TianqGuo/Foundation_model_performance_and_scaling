"""
Learning rate scheduling functions.

Implements cosine annealing learning rate schedule with warmup as used in LLaMA.
"""

import math


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Cosine annealing learning rate schedule with warmup.

    Implements the learning rate schedule used in LLaMA training:
    - Warmup phase: Linear increase from 0 to max_learning_rate
    - Cosine annealing: Cosine decay from max to min learning rate
    - Post-annealing: Constant at min_learning_rate

    Args:
        it: Current iteration (starting from 0)
        max_learning_rate: Maximum learning rate (αmax)
        min_learning_rate: Minimum learning rate (αmin)
        warmup_iters: Number of warmup iterations (Tw)
        cosine_cycle_iters: Number of cosine annealing iterations (Tc)

    Returns:
        Learning rate for the current iteration (αt)

    The schedule is defined as:
    - Warmup (t < Tw): αt = (t / Tw) * αmax
    - Cosine annealing (Tw ≤ t ≤ Tc): αt = αmin + 0.5 * (1 + cos((t - Tw) / (Tc - Tw) * π)) * (αmax - αmin)
    - Post-annealing (t > Tc): αt = αmin
    """
    if it < warmup_iters:
        # Warmup phase: Linear increase from 0 to max_learning_rate
        # αt = (t / Tw) * αmax
        return (it / warmup_iters) * max_learning_rate

    elif it <= cosine_cycle_iters:
        # Cosine annealing phase
        # αt = αmin + 0.5 * (1 + cos((t - Tw) / (Tc - Tw) * π)) * (αmax - αmin)
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine_factor = 0.5 * (1 + math.cos(progress * math.pi))
        return min_learning_rate + cosine_factor * (max_learning_rate - min_learning_rate)

    else:
        # Post-annealing phase: Constant at minimum learning rate
        # αt = αmin
        return min_learning_rate