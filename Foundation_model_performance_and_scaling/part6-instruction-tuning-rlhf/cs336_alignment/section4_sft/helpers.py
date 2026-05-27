from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """Tokenize prompt+output pairs and construct a causal-LM training batch.

    Returns input_ids and labels as the standard causal LM left/right shift of
    the concatenated sequence, plus response_mask = True only for output tokens
    in labels (prompt and padding positions are False).
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

    input_ids = padded[:, :-1]   # (batch, max_len - 1)
    labels = padded[:, 1:]        # (batch, max_len - 1)

    # Output tokens occupy labels[i, p_len-1 : s_len-1].
    # Derivation: labels[i][j] = tokens[i][j+1], so tokens[p_len] (first output)
    # appears at labels[p_len-1]; tokens[s_len-1] (last output) at labels[s_len-2].
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
    # nan_to_num handles 0 * (-inf) that arises when a probability underflows to 0
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
    logits = model(input_ids).logits                          # (batch, seq_len, vocab)
    log_probs_all = F.log_softmax(logits, dim=-1)
    log_probs = log_probs_all.gather(
        dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)                                             # (batch, seq_len)

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
    Calls loss.backward() internally; returns the detached scalar loss for logging.
    """
    batch_size = policy_log_probs.shape[0]
    loss = (
        -masked_normalize(policy_log_probs, response_mask, normalize_constant)
        / (batch_size * gradient_accumulation_steps)
    )
    loss.backward()
    return loss.detach(), {}


def log_generations(
    vllm_model: LLM,
    policy_model: PreTrainedModel | None,
    tokenizer: PreTrainedTokenizerBase,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    sampling_params: SamplingParams,
    device: str = "cuda",
) -> dict[str, Any]:
    """Generate responses with vLLM, compute rewards and diagnostics.

    Logs per-example: prompt, response, ground truth, rewards.
    Logs aggregate: accuracy, format rate, avg token entropy, avg response length
    (overall, correct-only, incorrect-only).

    Args:
        vllm_model:    vLLM LLM instance for fast generation.
        policy_model:  HuggingFace model for token entropy (None to skip entropy).
        tokenizer:     Tokenizer shared by both models.
        reward_fn:     Callable(response, ground_truth) → {reward, format_reward, answer_reward}.
        prompts:       List of formatted prompt strings.
        ground_truths: Ground-truth answer strings aligned with prompts.
        sampling_params: vLLM SamplingParams for generation.
        device:        Device for the policy model forward pass.

    Returns:
        dict with aggregate metrics and an "examples" list of per-example dicts.
    """
    outputs = vllm_model.generate(prompts, sampling_params)
    responses = [o.outputs[0].text for o in outputs]

    rewards_list = [reward_fn(resp, gt) for resp, gt in zip(responses, ground_truths)]

    # Token-level response lengths
    response_lengths = [
        len(tokenizer.encode(resp, add_special_tokens=False)) for resp in responses
    ]
    correct_idx = [i for i, r in enumerate(rewards_list) if r["answer_reward"] == 1.0]
    incorrect_idx = [i for i, r in enumerate(rewards_list) if r["answer_reward"] == 0.0]

    avg_len = sum(response_lengths) / max(len(response_lengths), 1)
    avg_len_correct = (
        sum(response_lengths[i] for i in correct_idx) / len(correct_idx)
        if correct_idx else 0.0
    )
    avg_len_incorrect = (
        sum(response_lengths[i] for i in incorrect_idx) / len(incorrect_idx)
        if incorrect_idx else 0.0
    )

    # Per-token entropy via HuggingFace model (only on first few examples to keep cost low)
    avg_entropy = 0.0
    if policy_model is not None:
        total_entropy, n_counted = 0.0, 0
        policy_model.eval()
        with torch.no_grad():
            for prompt, resp in zip(prompts[:5], responses[:5]):
                tok = tokenize_prompt_and_output([prompt], [resp], tokenizer)
                ids = tok["input_ids"].to(device)
                labs = tok["labels"].to(device)
                mask = tok["response_mask"].to(device).float()
                out = get_response_log_probs(policy_model, ids, labs, return_token_entropy=True)
                entropy = out["token_entropy"]          # (1, seq_len)
                n_resp = mask.sum()
                if n_resp > 0:
                    total_entropy += ((entropy * mask).sum() / n_resp).item()
                    n_counted += 1
        avg_entropy = total_entropy / max(n_counted, 1)

    examples = [
        {
            "prompt": prompts[i],
            "response": responses[i],
            "ground_truth": ground_truths[i],
            "rewards": rewards_list[i],
            "response_length": response_lengths[i],
        }
        for i in range(len(prompts))
    ]

    return {
        "accuracy": sum(r["answer_reward"] for r in rewards_list) / max(len(rewards_list), 1),
        "format_rate": sum(r["format_reward"] for r in rewards_list) / max(len(rewards_list), 1),
        "avg_reward": sum(r["reward"] for r in rewards_list) / max(len(rewards_list), 1),
        "avg_token_entropy": avg_entropy,
        "avg_response_length": avg_len,
        "avg_response_length_correct": avg_len_correct,
        "avg_response_length_incorrect": avg_len_incorrect,
        "examples": examples,
    }