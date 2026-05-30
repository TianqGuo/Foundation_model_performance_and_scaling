from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """Tokenize prompt+output pairs and construct a causal-LM training batch.

    Returns input_ids and labels (standard left/right shift of the concatenated
    sequence) and response_mask = True only for output tokens in labels.
    """
    prompt_ids = [tokenizer.encode(p, add_special_tokens=True) for p in prompt_strs]
    output_ids = [tokenizer.encode(o, add_special_tokens=False) for o in output_strs]

    sequences = [p + o for p, o in zip(prompt_ids, output_ids)]
    prompt_lens = [len(p) for p in prompt_ids]
    seq_lens = [len(s) for s in sequences]
    max_len = max(seq_lens)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    padded = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

    input_ids = padded[:, :-1]
    labels = padded[:, 1:]

    # labels[i][j] = tokens[i][j+1], so first output token (tokens[p_len]) appears
    # at labels[p_len-1]; last output token (tokens[s_len-1]) at labels[s_len-2].
    response_mask = torch.zeros(len(sequences), max_len - 1, dtype=torch.bool)
    for i, (p_len, s_len) in enumerate(zip(prompt_lens, seq_lens)):
        start = max(p_len - 1, 0)
        end = s_len - 1
        if end > start:
            response_mask[i, start:end] = True

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Per-token entropy of next-token predictions.

    Args:
        logits: (batch_size, sequence_length, vocab_size)
    Returns:
        (batch_size, sequence_length)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).nan_to_num(0.0).sum(dim=-1)


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Per-token conditional log-probs from a causal LM, optionally with token entropy.

    Args:
        model:                HuggingFace causal LM
        input_ids:            (batch_size, sequence_length)
        labels:               (batch_size, sequence_length) — shifted input_ids
        return_token_entropy: if True, also return per-token entropy
    Returns:
        dict with "log_probs" and optionally "token_entropy", both (batch, seq_len)
    """
    logits = model(input_ids).logits
    log_probs_all = F.log_softmax(logits, dim=-1)
    log_probs = log_probs_all.gather(
        dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)

    result: dict[str, torch.Tensor] = {"log_probs": log_probs}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """Masked sum divided by normalize_constant.

    Positions where mask == 0 (or False) do not contribute to the sum.
    """
    summed = (tensor * mask).sum(dim=dim) if dim is not None else (tensor * mask).sum()
    return summed / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute NLL loss over response tokens, scale for gradient accumulation, backprop.

    Normalizes by normalize_constant × batch_size × gradient_accumulation_steps so that
    the effective update is independent of both batch size and accumulation depth.
    """
    batch_size = policy_log_probs.shape[0]
    loss = (
        -masked_normalize(policy_log_probs, response_mask, normalize_constant)
        / (batch_size * gradient_accumulation_steps)
    )
    loss.backward()
    return loss.detach(), {}
