"""Dependency-light serving settings; model execution belongs on cloud GPUs."""
from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import re

DEFAULT_REPO = "sclion/part7-7a-smoke-execution-20261005T045405Z-dwaxLH"
DEFAULT_REVISION = "40c33b6ad0a7a2879e7b5a8f5f6a52baaa94a584"
# Candidate only: Serve LLM imports/dependency resolution must be checked separately.
RAY_VERSION = "2.49.2"


@dataclass(frozen=True)
class ServingConfig:
    """Hold immutable model/resource settings; validate before rendering or deployment.

    Inputs are configurable fields below. Defaults select the reviewed 7A BF16
    export, one replica/whole GPU and bounded contexts. No model imports or I/O.
    """
    model_repo: str = DEFAULT_REPO
    model_revision: str = DEFAULT_REVISION
    model_id: str = "part7-math-grpo"
    max_model_len: int = 4096
    max_num_seqs: int = 16
    gpu_memory_utilization: float = 0.5
    tensor_parallel_size: int = 1

    def validate(self):
        """Return self after identity/range checks; reject mutable revisions and unsafe IDs."""
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", self.model_repo):
            raise ValueError("Use owner/repository for the exported model")
        if not re.fullmatch(r"[0-9a-f]{40}", self.model_revision):
            raise ValueError("Pin the model to a full immutable Hub revision")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", self.model_id):
            raise ValueError("Use a short safe served model ID")
        if type(self.max_model_len) is not int or not 512 <= self.max_model_len <= 32768:
            raise ValueError("Context limit must be an integer in [512,32768]")
        if type(self.max_num_seqs) is not int or not 1 <= self.max_num_seqs <= 64:
            raise ValueError("Sequence capacity must be an integer in [1,64]")
        if not 0.1 <= self.gpu_memory_utilization <= 0.9:
            raise ValueError("GPU memory fraction must be in [0.1,0.9]")
        if type(self.tensor_parallel_size) is not int or self.tensor_parallel_size != 1:
            raise ValueError("Initial acceptance supports TP=1; larger topology needs separate validation")
        return self

    def snapshot_path(self):
        """Return deterministic PVC path derived from repo/revision, avoiding name collisions."""
        self.validate()
        identity = hashlib.sha256((self.model_repo + "@" + self.model_revision).encode()).hexdigest()[:24]
        return "/workspace/models/7d/" + identity

    def llm_config(self):
        """Return native Ray Serve LLM settings for the staged export, without loading it.

        One fixed replica; engine owns one whole GPU. Use raw completions for the
        MATH benchmark so no implicit chat template changes Part 5 prompt semantics.
        """
        self.validate()
        return dict(model_loading_config=dict(model_id=self.model_id, model_source=self.snapshot_path()),
                    deployment_config=dict(num_replicas=1, max_ongoing_requests=16),
                    engine_kwargs=dict(dtype="bfloat16", tensor_parallel_size=1,
                        max_model_len=self.max_model_len, max_num_seqs=self.max_num_seqs,
                        gpu_memory_utilization=self.gpu_memory_utilization, enforce_eager=True,
                        enable_prefix_caching=False))


def safe_file(root, relative):
    """Resolve a model inventory entry, rejecting escapes/symlinks/missing files.

    Inputs: snapshot root and relative POSIX name. Returns regular-file Path;
    no tensor parsing occurs. Used by snapshot validation and hash checks.
    """
    name = PurePosixPath(relative)
    path = Path(root) / relative
    if name.is_absolute() or ".." in name.parts or "\\" in relative or path.is_symlink() or not path.is_file():
        raise ValueError("Unsafe or missing snapshot file")
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Snapshot file escapes its directory")
    return path
