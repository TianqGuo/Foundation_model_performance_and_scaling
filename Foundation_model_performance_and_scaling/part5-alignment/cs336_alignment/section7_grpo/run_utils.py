"""Lightweight run bookkeeping; safe to use without model or CUDA imports."""

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import platform
from pathlib import Path
import random
import uuid


def file_fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def evaluation_indices(size: int, count: int, seed: int) -> list[int]:
    if size < 0 or count <= 0:
        raise ValueError("Evaluation requires a nonnegative dataset size and positive count")
    return random.Random(seed).sample(range(size), min(size, count))


def create_run_directory(base: Path, run_name: str) -> Path:
    if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("run_name must be a single nonempty filename component")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = base / f"{run_name}_{stamp}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def environment_versions() -> dict:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "vllm", "transformers", "flash-attn", "accelerate", "wandb", "math-verify"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def resolve_rollout_device(n_gpus: int, train_device: str, vllm_device: str) -> tuple[str, bool]:
    device = vllm_device if n_gpus >= 2 else train_device
    return device, device == train_device


def should_evaluate(skip_eval: bool, size: int, step: int, interval: int) -> bool:
    return not skip_eval and size > 0 and step % interval == 0


def accumulation_weight(microbatch_size: int, update_size: int) -> float:
    """Weight a microbatch mean by its share of the actual optimizer batch."""
    if not 0 < microbatch_size <= update_size:
        raise ValueError("Expected 0 < microbatch_size <= update_size")
    return microbatch_size / update_size


def training_microbatches(size: int, micro_size: int, accumulation_steps: int,
                          layout: str = 'stable', rng=None) -> list[tuple[int, list[int], int]]:
    """Return (example offset, row indices, actual optimizer-batch size).

    Stable layout shuffles intact original microbatches, including a partial one.
    Reshuffle reproduces the legacy per-example permutation. Each optimizer
    update collects accumulation_steps microbatches and uses their actual count.
    """
    if min(size, micro_size, accumulation_steps) <= 0 or layout not in ('stable', 'reshuffle'):
        raise ValueError('Positive batch sizes and stable/reshuffle layout required')
    rng = random if rng is None else rng
    indices = list(range(size))
    if layout == 'reshuffle':
        rng.shuffle(indices)
    batches = [indices[start:start + micro_size] for start in range(0, size, micro_size)]
    if layout == 'stable':
        rng.shuffle(batches)
    result = []
    offset = 0
    for i, batch in enumerate(batches):
        start = (i // accumulation_steps) * accumulation_steps
        actual_size = sum(len(b) for b in batches[start:start + accumulation_steps])
        result.append((offset, batch, actual_size))
        offset += len(batch)
    return result


def validate_frozen_policy_agreement(diagnostics: list[dict]) -> None:
    """Guard stable scoring before updates; tolerances are below tight GSPO bounds."""
    import math
    if not diagnostics or any(
        not math.isfinite(d['max_abs_token_log_ratio']) or
        not math.isfinite(d['max_abs_sequence_ratio_minus_one']) or
        d['max_abs_token_log_ratio'] > 1e-4 or
        d['max_abs_sequence_ratio_minus_one'] > 1e-5 for d in diagnostics
    ):
        raise ValueError(f'Stable frozen-policy scoring agreement failed; defer tighter clipping. Diagnostics: {diagnostics}')
