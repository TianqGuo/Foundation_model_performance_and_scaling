"""Frozen HuggingFace policy agreement; does not establish vLLM scoring parity."""
import torch


@torch.no_grad()
def policy_agreement(current, old, mask):
    if current.shape != old.shape or mask.shape != current.shape:
        raise ValueError("Matching log-probability and mask shapes required")
    mask = mask.bool()
    counts = mask.sum(1)
    if (counts == 0).any():
        raise ValueError("Every response requires tokens")
    delta = torch.where(mask, current.float() - old.float(), 0.)
    if not torch.isfinite(delta).all():
        raise ValueError("Nonfinite policy agreement")
    tokens = delta[mask]
    sequences = delta.sum(1) / counts
    return {
        "responses": len(counts), "tokens": int(counts.sum()),
        "current_dtype": str(current.dtype), "old_dtype": str(old.dtype),
        "mean_abs_token_log_ratio": float(tokens.abs().mean()),
        "max_abs_token_log_ratio": float(tokens.abs().max()),
        "max_abs_sequence_log_ratio": float(sequences.abs().max()),
        "max_abs_sequence_ratio_minus_one": float(torch.expm1(sequences).abs().max()),
    }
