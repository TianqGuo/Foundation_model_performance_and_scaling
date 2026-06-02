from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

_ALPACA_TEMPLATE = (
    Path(__file__).parent.parent / "prompts" / "alpaca_sft.prompt"
).read_text().rstrip("\n")


def _sequence_log_prob(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    response_start: int = 0,
) -> torch.Tensor:
    """Sum of response token log-probabilities (paper-correct DPO formulation).

    Returns sum_{t=response_start}^{T-2} log p(token_{t+1} | token_0 ... token_t).
    input_ids: 1-D tensor of token IDs, shape (T,).
    response_start: first index in the per-token array belonging to the response;
        prompt tokens are excluded. Defaults to 0 (full sequence).
    Logits cast to float32 before log_softmax to avoid bfloat16 underflow.
    """
    inputs = input_ids[:-1].unsqueeze(0)              # (1, T-1)
    targets = input_ids[1:]                            # (T-1,)
    logits = model(inputs).logits[0].float()           # (T-1, vocab), float32
    log_probs = F.log_softmax(logits, dim=-1)
    per_token = log_probs[torch.arange(len(targets), device=input_ids.device), targets]
    return per_token[response_start:].sum()


def per_instance_dpo_loss(
    lm: PreTrainedModel,
    lm_ref: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """Per-instance DPO loss (Rafailov et al., 2023).

    Formats (prompt, response) pairs with the Alpaca template + EOS, then computes:
        loss = -log σ(β * [(log π_θ(yw) - log π_ref(yw)) - (log π_θ(yl) - log π_ref(yl))])

    where log-probs are unconditional sequence log-probs (prompt contribution cancels
    in the difference, so conditioning is implicit).

    Returns the loss on the same device as lm.
    """
    policy_device = next(lm.parameters()).device
    ref_device = next(lm_ref.parameters()).device
    eos_id = tokenizer.eos_token_id

    def _tokenize(response: str) -> list[int]:
        text = _ALPACA_TEMPLATE.format(instruction=prompt, response=response)
        ids = tokenizer.encode(text, add_special_tokens=True)
        ids.append(eos_id)
        return ids

    # Compute where the response starts in the per-token log-prob array so we
    # only sum response tokens.  Prompt log-probs are identical for chosen and
    # rejected and would cancel anyway, but including them can cause catastrophic
    # cancellation in bfloat16 (both values become -inf → difference is NaN).
    prompt_text = _ALPACA_TEMPLATE.format(instruction=prompt, response="")
    prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=True))
    # per_token[t] = log p(token[t+1] | ...); first response token is at
    # index prompt_len in input_ids, so response_start = prompt_len - 1.
    response_start = max(0, prompt_len - 1)

    chosen_ids_list = _tokenize(response_chosen)
    rejected_ids_list = _tokenize(response_rejected)

    chosen_policy = torch.tensor(chosen_ids_list, dtype=torch.long, device=policy_device)
    rejected_policy = torch.tensor(rejected_ids_list, dtype=torch.long, device=policy_device)

    # Policy log-probs (gradients flow through these)
    lp_chosen_policy = _sequence_log_prob(lm, chosen_policy, response_start)
    lp_rejected_policy = _sequence_log_prob(lm, rejected_policy, response_start)

    # Reference log-probs (no gradients needed)
    with torch.no_grad():
        chosen_ref = torch.tensor(chosen_ids_list, dtype=torch.long, device=ref_device)
        rejected_ref = torch.tensor(rejected_ids_list, dtype=torch.long, device=ref_device)
        lp_chosen_ref = _sequence_log_prob(lm_ref, chosen_ref, response_start).to(policy_device)
        lp_rejected_ref = _sequence_log_prob(lm_ref, rejected_ref, response_start).to(policy_device)

    # Implicit reward margin and DPO loss
    reward_margin = beta * (
        (lp_chosen_policy - lp_chosen_ref) - (lp_rejected_policy - lp_rejected_ref)
    )
    return -F.logsigmoid(reward_margin)
