"""Autoregressive decoding helpers for Transformer language models."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cs336_basics.transformer_training.model.transformer_lm import TransformerLM


@dataclass
class SamplingConfig:
    """Configuration options for sampling tokens from the model."""

    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    eos_token_id: int | None = None


def _validate_sampling_args(temperature: float, top_k: int | None, top_p: float | None) -> None:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if top_p is not None and not (0.0 < top_p <= 1.0):
        raise ValueError("top_p must lie in the interval (0, 1]")


def _apply_top_k(logits: Tensor, top_k: int) -> Tensor:
    if top_k >= logits.shape[-1]:
        return logits

    values, indices = torch.topk(logits, top_k, dim=-1)
    filtered = torch.full_like(logits, -torch.inf)
    filtered.scatter_(-1, indices, values)
    return filtered


def _apply_top_p(probs: Tensor, top_p: float) -> Tensor:
    if top_p >= 1.0:
        return probs

    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    keep = (cumulative - sorted_probs) < top_p
    keep[..., 0] = True  # always keep at least one token

    trimmed = sorted_probs * keep
    normalizer = torch.clamp(trimmed.sum(dim=-1, keepdim=True), min=1e-12)
    trimmed = trimmed / normalizer

    filtered = torch.zeros_like(probs)
    filtered.scatter_(-1, sorted_indices, trimmed)
    return filtered


def sample_next_token(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample token indices from the final-position logits."""
    _validate_sampling_args(temperature, top_k, top_p)

    scaled_logits = logits / temperature
    if top_k is not None:
        scaled_logits = _apply_top_k(scaled_logits, top_k)

    probs = torch.softmax(scaled_logits, dim=-1)
    if top_p is not None:
        probs = _apply_top_p(probs, top_p)

    return torch.multinomial(probs, num_samples=1, generator=generator)


def decode(
    model: TransformerLM,
    input_ids: Tensor,
    *,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Autoregressively extend ``input_ids`` using ``model``."""
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    _validate_sampling_args(temperature, top_k, top_p)

    was_training = model.training
    if was_training:
        model.eval()

    try:
        generated = input_ids.clone()
        finished = torch.zeros(generated.size(0), dtype=torch.bool, device=generated.device)

        for _ in range(max_new_tokens):
            current_len = generated.shape[1]
            if current_len > model.context_length:
                model_input = generated[:, -model.context_length:]
            else:
                model_input = generated

            with torch.no_grad():
                logits = model(model_input)
                next_logits = logits[:, -1, :]

            next_token = sample_next_token(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=generator,
            )

            if eos_token_id is not None:
                eos_tensor = torch.full_like(next_token, eos_token_id)
                next_token = torch.where(finished.unsqueeze(1), eos_tensor, next_token)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None:
                finished |= next_token.squeeze(-1) == eos_token_id
                if torch.all(finished):
                    break

        return generated
    finally:
        if was_training:
            model.train()


def generate_text(
    model: TransformerLM,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token: str = "<|endoftext|>",
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> str:
    """Decode text from ``model`` using ``tokenizer`` for convenience."""
    if device is None:
        device = next(model.parameters()).device

    prompt_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    eos_token_id = None
    if eos_token is not None:
        token_bytes = eos_token.encode("utf-8")
        eos_token_id = tokenizer.byte_to_id.get(token_bytes)

    generated_ids = decode(
        model,
        input_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=eos_token_id,
        generator=generator,
    )

    return tokenizer.decode(generated_ids[0].tolist())
