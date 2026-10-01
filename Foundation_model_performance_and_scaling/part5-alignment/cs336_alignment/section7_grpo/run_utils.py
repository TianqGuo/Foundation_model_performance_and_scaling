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
