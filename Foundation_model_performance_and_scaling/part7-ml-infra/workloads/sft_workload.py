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


def tokenize_pairs(rows, tokenizer, *, max_length):
    """Encode exact segments with a supplied tokenizer; return examples and exclusions.

    Tokenizer construction/model access belongs on cloud. This adapter itself
    imports no libraries. Records overlength source indices without truncation;
    malformed token IDs or missing EOS fail. Returned digest covers actual IDs,
    labels and source order, so recovery can detect changed tokenizer behavior.
    """
    if type(max_length) is not int or max_length < 2:
        raise ValueError("Maximum length must be an integer >=2")
    kept, excluded = [], []
    for row in rows:
        prompt = tokenizer.encode(row["prompt"], add_special_tokens=False)
        response = tokenizer.encode(row["response"], add_special_tokens=False)
        # Validate IDs and EOS even for rows that will be excluded for length.
        example = response_example(prompt, response, eos_token_id=tokenizer.eos_token_id,
                                   max_length=max(max_length, len(prompt)+len(response)+1))
        if len(example["input_ids"]) > max_length:
            excluded.append(dict(source_index=row["source_index"], tokens=len(example["input_ids"])))
            continue
        kept.append(dict(source_index=row["source_index"], **example))
    raw = json.dumps(kept, sort_keys=True, separators=(",", ":")).encode()
    return kept, dict(kept=len(kept), excluded=excluded,
                     tokens_sha256=hashlib.sha256(raw).hexdigest())


def update_indices(step, rank, *, world_size, micro_batch_size, accumulation_steps):
    """Return ordered per-microbatch sample positions for a zero-based optimizer step.

    Each update consumes one contiguous global batch with disjoint rank slices.
    No shuffle, wraparound or hidden sampler cursor is used. Caller checks dataset
    capacity; negative steps/ranks and invalid sizes raise ValueError.
    """
    for value in (step, rank, world_size, micro_batch_size, accumulation_steps):
        if type(value) is not int:
            raise ValueError("Batch indices and sizes must be integers")
    if step < 0 or world_size < 1 or not 0 <= rank < world_size or min(micro_batch_size, accumulation_steps) < 1:
        raise ValueError("Invalid step, rank or batch dimensions")
    width = world_size * micro_batch_size
    start = step * width * accumulation_steps
    return [list(range(start + micro*width + rank*micro_batch_size,
                       start + micro*width + (rank+1)*micro_batch_size))
            for micro in range(accumulation_steps)]


def collate_lists(examples, *, pad_token_id):
    """Right-pad tokenized rows as Python lists; mask padding in attention and labels.

    Returns input_ids/attention_mask/labels only; tensor creation is cloud-only in
    the worker. Explicit pad identity is required; padding may use EOS but is never
    supervised. Reject empty batches and malformed shapes/masks/targets.
    """
    if not examples or type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("Nonempty batch and valid padding ID required")
    width = max(len(row["input_ids"]) for row in examples)
    batch = dict(input_ids=[], attention_mask=[], labels=[])
    for row in examples:
        ids, mask, labels = (row[key] for key in batch)
        if not ids or len(ids) != len(mask) or len(ids) != len(labels) or any(x != 1 for x in mask):
            raise ValueError("Collator expects aligned unpadded examples")
        if not any(x != -100 for x in labels[1:]):
            raise ValueError("Batch example has no causal response targets")
        padding = width-len(ids)
        batch["input_ids"].append(list(ids)+[pad_token_id]*padding)
        batch["attention_mask"].append(list(mask)+[0]*padding)
        batch["labels"].append(list(labels)+[-100]*padding)
    return batch


def loss_scale(local_tokens, global_tokens, world_size):
    """Return mean-loss weight for a global response-token mean under FSDP averaging.

    Use loss.mean * local_tokens * world_size / global_tokens on each microbatch,
    with one global denominator for the whole accumulated optimizer update. Do
    not additionally divide by accumulation steps. Reject zero/invalid counts.
    """
    if any(type(x) is not int or x <= 0 for x in (local_tokens, global_tokens, world_size)):
        raise ValueError("Positive integer token/rank counts required")
    if local_tokens > global_tokens:
        raise ValueError("Local token count exceeds the complete update")
    return local_tokens * world_size / global_tokens


def evaluation_slots(count, rank, world_size):
    """Return (index, include_metric) slots with equal forward counts across ranks.

    A final uneven group uses index zero as a zero-weight duplicate on trailing
    ranks. This keeps FSDP collectives aligned without biasing held-out loss.
    Reject invalid count/rank inputs; no model or distributed imports occur.
    """
    if any(type(value) is not int for value in (count, rank, world_size)):
        raise ValueError("Evaluation dimensions must be integers")
    if count < 1 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Nonempty evaluation data and a valid rank are required")
    return [(offset+rank, True) if offset+rank < count else (0, False)
            for offset in range(0, count, world_size)]
