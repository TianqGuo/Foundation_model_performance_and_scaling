from __future__ import annotations

from typing import Callable, Literal

import torch


def compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute per-rollout rewards and normalize within groups.

    rollout_responses and repeated_ground_truths are both length
    n_prompts * group_size, laid out as [q1_r1, q1_r2, ..., q1_rG, q2_r1, ...].
    """
    n = len(rollout_responses)
    assert n % group_size == 0

    rewards_data = [
        reward_fn(resp, gt)
        for resp, gt in zip(rollout_responses, repeated_ground_truths)
    ]
    raw_rewards = torch.tensor([d["reward"] for d in rewards_data], dtype=torch.float32)

    n_groups = n // group_size
    rewards_grouped = raw_rewards.view(n_groups, group_size)

    group_means = rewards_grouped.mean(dim=1, keepdim=True)
    centered = rewards_grouped - group_means

    if normalize_by_std:
        group_stds = rewards_grouped.std(dim=1, keepdim=True, unbiased=True)
        advantages_grouped = centered / (group_stds + advantage_eps)
    else:
        advantages_grouped = centered

    advantages = advantages_grouped.view(-1)

    metadata: dict[str, float] = {
        "mean_reward": raw_rewards.mean().item(),
        "std_reward": raw_rewards.std().item() if n > 1 else 0.0,
        "max_reward": raw_rewards.max().item(),
        "min_reward": raw_rewards.min().item(),
        "fraction_correct": (raw_rewards == 1.0).float().mean().item(),
        "mean_format_reward": sum(d["format_reward"] for d in rewards_data) / n,
        "mean_answer_reward": sum(d["answer_reward"] for d in rewards_data) / n,
    }
    return advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Per-token policy-gradient loss: -A * log π_θ(a_t | s_t).

    Args:
        raw_rewards_or_advantages: (batch_size, 1)
        policy_log_probs:          (batch_size, sequence_length)
    Returns:
        (batch_size, sequence_length)
    """
    return -(raw_rewards_or_advantages * policy_log_probs)


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """GRPO-Clip per-token loss with importance weighting and PPO-style clipping.

    Args:
        advantages:       (batch_size, 1)
        policy_log_probs: (batch_size, sequence_length)  — current policy
        old_log_probs:    (batch_size, sequence_length)  — rollout policy (detached)
        cliprange:        ε, e.g. 0.2
    Returns:
        loss     (batch_size, sequence_length)
        metadata dict with "is_clipped" mask
    """
    log_ratio = policy_log_probs - old_log_probs.detach()
    ratio = log_ratio.exp()
    clipped_ratio = ratio.clamp(1.0 - cliprange, 1.0 + cliprange)

    # -min(ratio * A, clipped_ratio * A); advantages broadcasts over seq_len
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

    is_clipped = (ratio < 1.0 - cliprange) | (ratio > 1.0 + cliprange)
    return loss, {"is_clipped": is_clipped.float()}


def compute_grpo_no_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Off-policy importance-weighted loss without PPO-style clipping.

    Args:
        advantages:       (batch_size, 1)
        policy_log_probs: (batch_size, sequence_length) — current policy
        old_log_probs:    (batch_size, sequence_length) — rollout policy (detached)
    Returns:
        loss (batch_size, sequence_length), empty metadata dict
    """
    log_ratio = policy_log_probs - old_log_probs.detach()
    ratio = log_ratio.exp()
    loss = -(ratio * advantages)
    return loss, {}


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dispatch to the correct policy-gradient loss function."""
    if loss_type == "no_baseline":
        assert raw_rewards is not None
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    elif loss_type == "grpo_clip":
        assert advantages is not None and old_log_probs is not None and cliprange is not None
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    elif loss_type == "grpo_no_clip":
        assert advantages is not None and old_log_probs is not None
        return compute_grpo_no_clip_loss(advantages, policy_log_probs, old_log_probs)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}")


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """Mean of tensor over positions where mask == 1.

    Args:
        tensor: any shape
        mask:   same shape as tensor
        dim:    dimension to reduce; None reduces over all masked elements
    Returns:
        scalar (dim=None) or tensor with that dimension removed.
        Positions where the mask is entirely zero produce nan (dim != None)
        or 0 (dim=None, which clamps the denominator to avoid division by zero).
    """
    mask = mask.float()
    if dim is None:
        # clamp so a fully-masked tensor returns 0 rather than nan
        return (tensor * mask).sum() / mask.sum().clamp(min=1)
    # Let 0/0 → nan naturally: signals that no elements contributed
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    length_norm: Literal["masked_mean", "masked_normalize"] = "masked_mean",
    max_response_tokens: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Forward + backward pass for one GRPO microbatch.

    Computes per-token loss, reduces over response tokens (masked_mean or
    masked_normalize), averages over the batch, scales by
    1/gradient_accumulation_steps, and calls loss.backward().
    Returns the detached scalar loss.
    """
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    # (batch_size, seq_len) → (batch_size,) → scalar
    if length_norm == "masked_normalize":
        from cs336_alignment.section4_sft.helpers import masked_normalize
        assert max_response_tokens is not None, "max_response_tokens required for masked_normalize"
        per_example_loss = masked_normalize(
            per_token_loss, response_mask, normalize_constant=max_response_tokens, dim=1
        )
    else:
        per_example_loss = masked_mean(per_token_loss, response_mask, dim=1)

    loss = per_example_loss.mean() / gradient_accumulation_steps
    loss.backward()

    return loss.detach(), metadata
