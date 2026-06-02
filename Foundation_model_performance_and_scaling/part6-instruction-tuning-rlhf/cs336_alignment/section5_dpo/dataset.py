"""Anthropic HH preference dataset loader for DPO training."""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path


_HH_FILES = [
    "harmless-base.jsonl.gz",
    "helpful-base.jsonl.gz",
    "helpful-online.jsonl.gz",
    "helpful-rejection-sampled.jsonl.gz",
]


def _parse_conversation(text: str) -> tuple[str, str]:
    """Extract (instruction, response) from a single-turn HH conversation string.

    Format: '\n\nHuman: {msg}\n\nAssistant: {response}'
    Returns (instruction, response) stripped of whitespace.
    """
    # Split on first '\n\nAssistant:' to separate human message from response
    parts = text.split("\n\nAssistant:", 1)
    instruction = parts[0].replace("\n\nHuman:", "").strip()
    response = parts[1].strip() if len(parts) > 1 else ""
    return instruction, response


def load_hh_dataset(
    data_dir: str | Path,
    balance: bool = False,
    seed: int = 42,
) -> list[dict]:
    """Load and preprocess the Anthropic HH preference dataset.

    Combines all four HH files, filters to single-turn conversations only
    (human sent exactly one message), and returns a list of dicts with keys:
        instruction  — the human's message
        chosen       — the preferred assistant response
        rejected     — the rejected assistant response
        source       — which file the example came from

    Args:
        data_dir: directory containing the four .jsonl.gz files
        balance: if True, keep all harmless-base examples and downsample the
            three helpful files combined to the same total, giving 50% harmless /
            50% helpful. Default False loads all examples as-is (~25% harmless).
        seed: random seed for subsampling when balance=True

    Returns:
        List of single-turn preference examples across all four files.
    """
    data_dir = Path(data_dir)
    by_source: dict[str, list[dict]] = {}

    for filename in _HH_FILES:
        path = data_dir / filename
        if not path.exists():
            continue
        source = filename.replace(".jsonl.gz", "")
        source_examples: list[dict] = []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                chosen_text = ex.get("chosen", "")
                rejected_text = ex.get("rejected", "")

                # Filter to single-turn: exactly one Human turn in the conversation
                if chosen_text.count("\n\nHuman:") != 1:
                    continue

                instruction, chosen_response = _parse_conversation(chosen_text)
                _, rejected_response = _parse_conversation(rejected_text)

                if not instruction or not chosen_response or not rejected_response:
                    continue

                source_examples.append({
                    "instruction": instruction,
                    "chosen": chosen_response,
                    "rejected": rejected_response,
                    "source": source,
                })
        by_source[source] = source_examples

    if balance and by_source:
        # Give harmless-base 50% of total, helpful files share the other 50%.
        # Keeps all harmless examples; downsamples helpful proportionally.
        harmless_key = "harmless-base"
        harmless = by_source.get(harmless_key, [])
        helpful = {k: v for k, v in by_source.items() if k != harmless_key}
        n_harmless = len(harmless)
        n_helpful_total = sum(len(v) for v in helpful.values())
        rng = random.Random(seed)
        if n_helpful_total > n_harmless and helpful:
            # Sample n_harmless examples from helpful proportionally across files
            balanced_helpful: dict[str, list] = {}
            for k, v in helpful.items():
                n_keep = round(n_harmless * len(v) / n_helpful_total)
                balanced_helpful[k] = rng.sample(v, min(n_keep, len(v)))
            by_source = {harmless_key: harmless, **balanced_helpful}
        counts_str = ", ".join(f"{k}={len(v)}" for k, v in by_source.items())
        total = sum(len(v) for v in by_source.values())
        print(f"  Balanced to 50/50 harmless/helpful: {counts_str} (total={total})")

    examples = [ex for src in by_source.values() for ex in src]
    return examples


def split_train_val(
    examples: list[dict],
    n_val: int = 200,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split examples into train and validation sets."""
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    return shuffled[n_val:], shuffled[:n_val]
