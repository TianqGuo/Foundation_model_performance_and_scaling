"""CPU-only settings for the bounded Ray Train/FSDP2 SFT experiment."""
from dataclasses import asdict, dataclass
import math
import re


@dataclass(frozen=True)
class SFTConfig:
    """Hold model, workload and topology inputs; no imports or model execution.

    Defaults propose a four-step two-rank experiment. A full model revision is
    mandatory for future cloud execution, but may remain unresolved in previews.
    """
    model_repo: str = "Qwen/Qwen2.5-Math-1.5B"
    model_revision: str | None = None
    train_count: int = 128
    eval_count: int = 16
    seed: int = 12345
    world_size: int = 2
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    max_sequence_length: int = 2048
    total_steps: int = 4
    checkpoint_step: int = 2
    learning_rate: float = 1e-5

    def validate(self, *, cloud=False):
        """Return self after strict bounds/identity checks; reject unsupported topology.

        cloud=True also rejects unresolved model revisions. This validates only
        configuration, never dependency compatibility, memory or GPU availability.
        """
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", self.model_repo):
            raise ValueError("Use a Hugging Face owner/repository")
        if self.model_revision is not None and not re.fullmatch(r"[0-9a-f]{40}", self.model_revision):
            raise ValueError("Model revision must be a full immutable commit")
        if cloud and self.model_revision is None:
            raise ValueError("Resolve and pin the model revision before cloud execution")
        for name, lower, upper in (("train_count", 1, 4096), ("eval_count", 1, 512),
                ("seed", 0, 2**32-1), ("micro_batch_size", 1, 8),
                ("gradient_accumulation_steps", 1, 16), ("max_sequence_length", 128, 8192),
                ("total_steps", 2, 32), ("checkpoint_step", 1, 31)):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in [{lower},{upper}]")
        if type(self.world_size) is not int or self.world_size != 2:
            raise ValueError("Initial acceptance requires exactly two training ranks")
        if self.checkpoint_step >= self.total_steps:
            raise ValueError("Checkpoint step must precede the final step")
        if isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, (int, float)) or not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1e-3:
            raise ValueError("Learning rate must be finite and in (0,1e-3]")
        if self.total_steps * self.global_batch_size() > self.train_count:
            raise ValueError("Bounded experiment must fit in the selected training prefix")
        return self

    def global_batch_size(self):
        """Return samples per optimizer update from ranks, microbatch and accumulation."""
        return self.world_size * self.micro_batch_size * self.gradient_accumulation_steps

    def preview(self):
        """Return validated serializable settings and explicit unverified runtime state."""
        self.validate()
        return dict(settings=asdict(self), global_batch_size=self.global_batch_size(),
                    status="CPU preparation only; trainer/image/recovery pending",
                    model_revision_resolved=self.model_revision is not None,
                    model_loaded=False, gpu_execution_verified=False)
