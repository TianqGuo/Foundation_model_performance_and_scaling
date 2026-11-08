"""Dependency-free SFT pair selection and response-only label construction.

    No model/tokenizer is imported. Token IDs must be produced on cloud using the
    explicitly chosen tokenizer, with prompt and response encoded separately and
    add_special_tokens=False. No implicit chat template or BOS is inserted.
"""
import hashlib
import json
from pathlib import Path

PART5_SFT_SHA256 = "55bf19e43fba511b5e90fc256c5e415f68049de25caf1204cae25475a93f241f"


def select_pairs(source, *, expected_sha256=PART5_SFT_SHA256, train_count=128,
                 eval_count=16, seed=12345):
    """Read hash-pinned JSONL and return disjoint deterministic pairs plus provenance.

    Inputs: existing Part 5 prompt/response file, expected digest and selection
    counts/seed. Keeps exact text and the first response for each exact prompt;
    assigns unique prompt groups by SHA256(seed,prompt), then splits. Raises on
    malformed/empty records, hash mismatch or insufficient unique prompts. No
    reward filtering, model loading, tokenization or writes occur here.
    """
    for value in (train_count, eval_count):
        if type(value) is not int or value < 1:
            raise ValueError("Split counts must be positive integers")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an unsigned 32-bit integer")
    raw = Path(source).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError("SFT source hash differs from the recorded reference")
    groups = {}
    rows = raw.decode("utf-8").splitlines()
    for index, line in enumerate(rows):
        try:
            row = json.loads(line)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid SFT JSON at source index {index}") from exc
        if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key].strip()
                                            for key in ("prompt", "response")):
            raise ValueError(f"Nonempty prompt/response required at source index {index}")
        groups.setdefault(row["prompt"], dict(source_index=index, prompt=row["prompt"], response=row["response"]))
    ordered = sorted(groups.values(), key=lambda row: hashlib.sha256(
        json.dumps([seed, row["prompt"]], ensure_ascii=False).encode()).hexdigest())
    if len(ordered) < train_count + eval_count:
        raise ValueError("Not enough unique prompts for disjoint train/evaluation splits")
    train, evaluation = ordered[:train_count], ordered[train_count:train_count+eval_count]
    receipt = dict(source_name=Path(source).name, source_sha256=digest, source_rows=len(rows),
        unique_prompts=len(groups), duplicate_prompt_rows=len(rows)-len(groups),
        selection="SHA256([seed, exact prompt]); first response per prompt; no regrading",
        seed=seed, train_indices=[row["source_index"] for row in train],
        evaluation_indices=[row["source_index"] for row in evaluation],
        evaluation_scope="held-out SFT loss; not MATH answer accuracy")
    return train, evaluation, receipt


def response_example(prompt_ids, response_ids, *, eos_token_id, max_length):
    """Return lists for causal-LM inputs/mask/labels from separately encoded segments.

    Prompt labels are -100; response and one terminal EOS are supervised. Avoids
    adding a second EOS when already present. Rejects empty/invalid segments and
    overlength examples rather than silently truncating away response targets.
    The future cloud collator must pad labels with -100 and masks with zero.
    """
    if type(eos_token_id) is not int or eos_token_id < 0:
        raise ValueError("A valid tokenizer EOS ID is required")
    if type(max_length) is not int or max_length < 2:
        raise ValueError("Maximum length must be an integer >=2")
    prompt, response = list(prompt_ids), list(response_ids)
    if not prompt or not response or any(type(x) is not int or x < 0 for x in prompt+response):
        raise ValueError("Prompt/response require nonempty nonnegative integer token IDs")
    if response[-1] != eos_token_id:
        response.append(eos_token_id)
    ids = prompt + response
    if len(ids) > max_length:
        raise ValueError("Overlength SFT example; record exclusion before training")
    return dict(input_ids=ids, attention_mask=[1]*len(ids), labels=[-100]*len(prompt)+response)
