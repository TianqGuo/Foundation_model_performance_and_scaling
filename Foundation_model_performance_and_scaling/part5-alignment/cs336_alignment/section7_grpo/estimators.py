"""Composable policy-gradient estimators; GSPO uses a response sequence ratio.

GSPO: exp(mean_response(log pi - log pi_old)), clipped once per sequence.
Reference: https://arxiv.org/abs/2507.18071 . No zero-advantage pruning here.
"""
from dataclasses import dataclass, asdict
import torch


@dataclass(frozen=True)
class EstimatorConfig:
    baseline: str = 'mean'
    advantage_normalizer: str = 'std'
    importance_reweighting: str = 'none'
    loss_normalization: str = 'sequence'

    def __post_init__(self):
        for name, values in {'baseline': ('mean','none'),
            'advantage_normalizer': ('std','none','mean'),
            'importance_reweighting': ('none','noclip','grpo','gspo'),
            'loss_normalization': ('sequence','constant')}.items():
            if getattr(self, name) not in values:
                raise ValueError(f'Invalid {name}: {getattr(self, name)}')

    def to_dict(self): return asdict(self)


def resolve_estimator(args):
    """Explicit independent knobs override the old CLI aliases independently."""
    aliases = {'no_baseline': ('none','none','none'),
               'reinforce_with_baseline': ('mean','std' if args.use_std_normalization else 'none','none'),
               'grpo_clip': ('mean','std' if args.use_std_normalization else 'none','grpo'),
               'grpo_no_clip': ('mean','std' if args.use_std_normalization else 'none','noclip')}
    baseline, normalizer, weighting = aliases[args.loss_type]
    return EstimatorConfig(
        getattr(args, 'baseline', None) or baseline,
        getattr(args, 'advantage_normalizer', None) or normalizer,
        getattr(args, 'importance_reweighting', None) or weighting,
        getattr(args, 'loss_normalization', None) or ('constant' if args.length_norm == 'masked_normalize' else 'sequence'),
    )


def score_rewards(reward_fn, responses, ground_truths):
    if not responses or len(responses) != len(ground_truths):
        raise ValueError('Reward batches must have equal nonzero sizes')
    data = [reward_fn(r,g) for r,g in zip(responses,ground_truths)]
    rewards = torch.tensor([d['reward'] for d in data], dtype=torch.float32)
    if not torch.isfinite(rewards).all(): raise ValueError('Rewards must be finite')
    meta = {'mean_reward': rewards.mean().item(),
        'std_reward': rewards.std().item() if len(rewards)>1 else 0.,
        'max_reward': rewards.max().item(), 'min_reward': rewards.min().item(),
        'fraction_correct': (rewards == 1.).float().mean().item(),
        'mean_format_reward': sum(d['format_reward'] for d in data)/len(data),
        'mean_answer_reward': sum(d['answer_reward'] for d in data)/len(data)}
    return rewards, meta


def build_advantages(rewards, group_size, baseline='mean', normalizer='std', eps=1e-6):
    EstimatorConfig(baseline=baseline, advantage_normalizer=normalizer)
    if rewards.ndim != 1 or not rewards.numel() or group_size <= 0 or rewards.numel()%group_size:
        raise ValueError('Rewards must form complete nonempty groups')
    if eps <= 0 or not torch.isfinite(rewards).all(): raise ValueError('Positive epsilon and finite rewards required')
    groups = rewards.detach().reshape(-1,group_size)
    mean = groups.mean(1,keepdim=True)
    advantages = groups - mean if baseline == 'mean' else groups
    if normalizer == 'std':
        std = groups.std(1,keepdim=True,unbiased=True) if group_size>1 else torch.zeros_like(mean)
        advantages = advantages/(std+eps)
    elif normalizer == 'mean':
        if (groups < 0).any(): raise ValueError('Mean reward normalization requires nonnegative rewards')
        advantages = advantages/(mean+eps)
    return advantages.reshape(-1)


def estimator_loss(advantages, policy_log_probs, response_mask, method='none', old_log_probs=None, cliprange=.2):
    EstimatorConfig(importance_reweighting=method)
    if policy_log_probs.ndim != 2 or response_mask.shape != policy_log_probs.shape:
        raise ValueError('Log probabilities/mask must have matching batch x token shapes')
    mask = response_mask.bool()
    if not mask.any(1).all(): raise ValueError('Every response must have at least one token')
    a = advantages.detach().reshape(-1,1)
    if a.shape[0] != mask.shape[0] or not torch.isfinite(a).all(): raise ValueError('Invalid advantages')
    # Keep full precision for CPU references; use FP32 ratios with BF16/FP16 training.
    lp = policy_log_probs.float() if policy_log_probs.dtype in (torch.float16,torch.bfloat16) else policy_log_probs
    lp = torch.where(mask, lp, 0.)
    if not torch.isfinite(lp).all(): raise ValueError('Nonfinite response log probabilities')
    if method == 'none': return -a*lp, {}
    if old_log_probs is None or old_log_probs.shape != lp.shape: raise ValueError('Matching rollout-policy log probabilities required')
    old = torch.where(mask, old_log_probs.detach().to(lp.dtype), 0.)
    if not torch.isfinite(old).all(): raise ValueError('Nonfinite old response log probabilities')
    delta = lp-old
    if method == 'gspo': delta = delta.sum(1,keepdim=True)/mask.sum(1,keepdim=True)
    ratio = delta.exp()
    if not torch.isfinite(ratio).all(): raise ValueError('Importance ratio overflow; abort instead of silently clipping log ratios')
    if method == 'noclip': return -a*ratio, {}
    if cliprange is None or not 0 < cliprange < 1: raise ValueError('Clipping epsilon must be between 0 and 1')
    clipped = ratio.clamp(1-cliprange,1+cliprange)
    loss = -torch.minimum(ratio*a,clipped*a)
    outside = (ratio<1-cliprange)|(ratio>1+cliprange)
    active = (ratio*a > clipped*a)
    meta = {'is_clipped': outside.expand_as(lp).float(),
            'clip_active': active.expand_as(lp).float()}
    if method == 'gspo': meta['sequence_is_clipped'] = outside.squeeze(1).float()
    return loss.expand_as(lp), meta


def aggregate_loss(per_token_loss, response_mask, normalization='sequence', constant=None):
    EstimatorConfig(loss_normalization=normalization)
    mask=response_mask.bool()
    counts=mask.sum(1)
    if per_token_loss.shape!=mask.shape or not counts.numel() or (counts==0).any():
        raise ValueError('Loss/mask shapes must match and every sequence must have tokens')
    totals=torch.where(mask, per_token_loss, 0.).sum(1)
    if normalization=='constant':
        if constant is None or constant<=0: raise ValueError('Positive per-sequence normalization constant required')
        return (totals/constant).mean()
    return (totals/counts).mean()
