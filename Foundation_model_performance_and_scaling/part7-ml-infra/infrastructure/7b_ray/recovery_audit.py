"""State fingerprints and thin, opt-in hooks around pinned verl checkpoint methods.

Importing this module is CPU-safe. Tensor access/imports occur only inside the
cloud hooks. Fingerprints include all local tensor bytes, not sampled statistics.
No training loop, optimizer or distributed synchronization is replaced.
"""
from contextlib import contextmanager
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import struct
import time


class PlannedInterruption(RuntimeError):
    """Raised only after a requested checkpoint boundary has completed."""


def fingerprint(value, tensor_encoder=None):
    """Hash nested state deterministically, preserving types and container order.

    Inputs: nested scalar/dict/list/tuple/bytes state; optional encoder returning
    (metadata, bytes-like payload) for tensor/array leaves, or None otherwise.
    Returns: SHA-256 hex digest. Unsupported leaves raise TypeError rather than
    silently hashing repr or pickle. No state is changed and no RNG is consumed.
    """
    digest = hashlib.sha256()

    def visit(item):
        """Feed one typed state subtree into the enclosing incremental digest."""
        encoded = tensor_encoder(item) if tensor_encoder else None
        if encoded is not None:
            metadata, payload = encoded
            visit(("array", metadata))
            digest.update(struct.pack("!Q", len(payload)))
            digest.update(payload)
        elif isinstance(item, dict):
            digest.update(b"dict")
            # Sorting typed key hashes avoids Python's mixed-type ordering rules.
            for key in sorted(item, key=lambda key: fingerprint(key, tensor_encoder)):
                visit(key)
                visit(item[key])
            digest.update(b"end")
        elif isinstance(item, (tuple, list)):
            digest.update(b"tuple" if isinstance(item, tuple) else b"list")
            for child in item:
                visit(child)
            digest.update(b"end")
        elif isinstance(item, bytes):
            digest.update(b"bytes" + struct.pack("!Q", len(item)) + item)
        elif item is None or type(item) in (str, bool, int, float):
            payload = json.dumps(item, ensure_ascii=True, allow_nan=False).encode()
            digest.update(type(item).__name__.encode() + struct.pack("!Q", len(payload)) + payload)
        else:
            raise TypeError(f"Unsupported recovery state type: {type(item).__name__}")

    visit(value)
    return digest.hexdigest()


def tensor_encoder(value):
    """Encode cloud tensor/NumPy leaves as metadata and contiguous local bytes.

    Input: any state leaf. Returns None for ordinary scalars, otherwise a metadata
    tuple and byte view. DTensor metadata records global shape, placements and
    mesh; payload covers this rank's entire shard. Copies to CPU may synchronize
    the GPU and consume host memory. This function imports Torch only when called.
    """
    import numpy as np
    import torch
    from torch.distributed.tensor import DTensor
    if isinstance(value, torch.Tensor):
        local = value.to_local() if isinstance(value, DTensor) else value
        metadata = ("torch", str(value.dtype), tuple(value.shape), tuple(local.shape),
                    tuple(str(p) for p in value.placements) if isinstance(value, DTensor) else ())
        if isinstance(value, DTensor):
            metadata += (tuple(value.device_mesh.mesh.flatten().tolist()),)
        # Byte views support bfloat16 without NumPy dtype conversion.
        payload = local.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        return metadata, memoryview(payload).cast("B")
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays are not supported in checkpoint state")
        return ("numpy", str(value.dtype), tuple(value.shape)), memoryview(np.ascontiguousarray(value)).cast("B")
    if isinstance(value, np.generic):
        return ("numpy_scalar", str(value.dtype)), memoryview(value.tobytes())
    return None


def compare_states(expected, actual, encoder=None):
    """Return per-component digests/matches for two equally keyed state mappings.

    Inputs: expected and actual component dictionaries plus optional leaf encoder.
    Returns: JSON-compatible component results. Different component keys raise
    ValueError. Callers persist results before failing on any false match.
    """
    if expected.keys() != actual.keys():
        raise ValueError("Recovery audit component inventory differs")
    return {key: dict(expected=fingerprint(expected[key], encoder),
                      actual=fingerprint(actual[key], encoder)) for key in expected}


def require_matches(results):
    """Validate recorded component digests; raise on any restoration mismatch.

    Input: compare_states result. Returns None on success; RuntimeError lists
    mismatched components. Separate persistence from validation to retain failures.
    """
    changed = [key for key, value in results.items() if value["expected"] != value["actual"]]
    if changed:
        raise RuntimeError(f"Restored state differs: {changed}")


@contextmanager
def preserve_rng(manager):
    """Keep audit work from advancing trainer RNG even if an audit raises.

    Input: native checkpoint manager exposing get_rng_state/load_rng_state.
    Yields None, then restores Python/NumPy/Torch/device RNG in finally. This
    covers the state supported by verl, not vLLM server RNG or every CUDA device.
    """
    state = manager.get_rng_state()
    try:
        yield
    finally:
        manager.load_rng_state(state)


def write_event(event, **fields):
    """Append a process-specific audit event below PART7_RUN_DIR/recovery.

    Inputs: event name and JSON-compatible fields. Returns the written record.
    Adds PID/time; creates directories. No credentials/environment are dumped.
    Missing run directory or file/serialization errors propagate to the caller.
    """
    destination = Path(os.environ["PART7_RUN_DIR"]) / "recovery"
    destination.mkdir(parents=True, exist_ok=True)
    record = dict(event=event, pid=os.getpid(), time=time.time(), **fields)
    with (destination / f"events_pid_{os.getpid()}.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    return record


def audit_rank(manager, actor_path):
    """Compare one rank's checkpoint payloads with live FSDP2 training state.

    Inputs: native FSDPCheckpointManager and actor checkpoint directory. Returns
    component digests after exact model/Adam/scheduler/RNG comparisons. Runs only
    on cloud; trusted full checkpoints are deserialized. All ranks enter the same
    native sharded state context. Persist evidence before rejecting mismatches.
    Requires populated Adam moments and a scheduler, and preserves trainer RNG.
    """
    import torch
    from torch.distributed.fsdp import StateDictType, ShardedStateDictConfig, ShardedOptimStateDictConfig
    from verl.utils.fsdp_utils import get_fsdp_state_ctx
    if manager.world_size != 2 or not torch.cuda.is_available():
        raise RuntimeError("Recovery audit requires two cloud FSDP2 ranks")
    if manager.optimizer is None or manager.lr_scheduler is None:
        raise RuntimeError("Recovery audit requires optimizer and scheduler")
    results = {}
    with preserve_rng(manager):
        with get_fsdp_state_ctx(manager.model, StateDictType.SHARDED_STATE_DICT,
                                ShardedStateDictConfig(offload_to_cpu=True),
                                ShardedOptimStateDictConfig(offload_to_cpu=True)):
            for component, filename, capture in (
                ("model", "model", manager.model.state_dict),
                ("optimizer", "optim", manager.optimizer.state_dict),
            ):
                path = Path(actor_path) / f"{filename}_world_size_2_rank_{manager.rank}.pt"
                saved = torch.load(path, weights_only=False, map_location="cpu")
                expected = fingerprint(saved, tensor_encoder)
                del saved
                current = capture()
                if component == "optimizer" and not any(
                    "exp_avg" in value and "exp_avg_sq" in value and "step" in value
                    for value in current["state"].values()
                ):
                    raise RuntimeError("Restored Adam moments/steps are absent")
                results[component] = dict(expected=expected, actual=fingerprint(current, tensor_encoder))
                del current
        saved_extra = torch.load(Path(actor_path) / f"extra_state_world_size_2_rank_{manager.rank}.pt",
                                 weights_only=False, map_location="cpu")
        results.update(compare_states(
            dict(scheduler=saved_extra["lr_scheduler"], rng=saved_extra["rng"]),
            dict(scheduler=manager.lr_scheduler.state_dict(), rng=manager.get_rng_state()), tensor_encoder))
        write_event("rank_restore", rank=manager.rank, world_size=manager.world_size,
                    checkpoint=str(actor_path), components=results)
        # Fail both ranks together; one rank must not continue into training alone.
        bad = torch.tensor(int(any(v["expected"] != v["actual"] for v in results.values())),
                           device=torch.device("cuda", torch.cuda.current_device()))
        torch.distributed.all_reduce(bad, op=torch.distributed.ReduceOp.MAX)
        if bad.item():
            raise RuntimeError("At least one rank failed the full training-state audit")
    return results


def install_worker_hooks(manager_class=None):
    """Wrap native FSDP load to audit restored state before returning to verl.

    Input: optional manager class for lightweight tests; default imports pinned
    verl. Returns None. Idempotent per process; preserves the original signature
    and delegates loading exactly once. No model is loaded during installation.
    """
    if manager_class is None:
        from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
        manager_class = FSDPCheckpointManager
    if getattr(manager_class, "_part7_recovery_audit", False):
        return
    original = manager_class.load_checkpoint

    @wraps(original)
    def audited_load(self, local_path, *args, **kwargs):
        """Delegate native loading, then audit a nonempty local resume path."""
        if kwargs.get("del_local_after_load", False) or (len(args) > 1 and args[1]):
            raise ValueError("Recovery audit requires keeping local checkpoint files")
        started = time.monotonic()
        result = original(self, local_path, *args, **kwargs)
        native_seconds = time.monotonic() - started
        if local_path is not None:
            audit_rank(self, local_path)
            write_event("rank_load_complete", rank=self.rank, native_load_seconds=native_seconds,
                        audit_seconds=time.monotonic() - started - native_seconds,
                        seconds=time.monotonic() - started)
        return result

    manager_class.load_checkpoint = audited_load
    manager_class._part7_recovery_audit = True


def install_trainer_hooks(trainer_class=None):
    """Wrap native trainer save/load and record training batch source indices.

    Input: optional trainer class for tests; default imports pinned verl. Returns
    None. Save-boundary interruption is enabled by PART7_INTERRUPT_STEP; it writes
    a receipt only after the completion marker and all required files exist.
    Restore audits compare global step/data before the next generation. Tensor
    deserialization and Ray inspection execute only inside the cloud wrappers.
    """
    if trainer_class is None:
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        trainer_class = RayPPOTrainer
    if getattr(trainer_class, "_part7_recovery_audit", False):
        return
    original_save = trainer_class._save_checkpoint
    original_load = trainer_class._load_checkpoint
    original_batch = trainer_class._get_gen_batch

    @wraps(original_save)
    def saved(self, *args, **kwargs):
        """Save natively, validate completion, then optionally interrupt the job."""
        from importlib import import_module
        bundle = import_module("infrastructure.7b_ray.checkpoint_bundle")
        started = time.monotonic()
        result = original_save(self, *args, **kwargs)
        root = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
        bundle.layout(root, 2)
        marker = root.parent / "latest_checkpointed_iteration.txt"
        if int(marker.read_text()) != self.global_steps:
            raise RuntimeError("Checkpoint completion marker does not match saved step")
        write_event("checkpoint_complete", step=self.global_steps, checkpoint=str(root),
                    seconds=time.monotonic() - started)
        if int(os.environ.get("PART7_INTERRUPT_STEP", "0")) == self.global_steps:
            # Driver accepts a planned stop only with this validated receipt.
            receipt = dict(step=self.global_steps, checkpoint=str(root), completed=True)
            (Path(os.environ["PART7_RUN_DIR"]) / "planned_interruption.json").write_text(json.dumps(receipt))
            raise PlannedInterruption(f"7B planned interruption after complete step {self.global_steps}")
        return result

    @wraps(original_load)
    def loaded(self, *args, **kwargs):
        """Load natively and compare data-loader progress and saved global step."""
        import torch
        started = time.monotonic()
        result = original_load(self, *args, **kwargs)
        if self.config.trainer.resume_mode == "resume_path":
            root = Path(self.config.trainer.resume_from_path)
            expected_step = int(root.name.removeprefix("global_step_"))
            saved_state = torch.load(root / "data.pt", weights_only=False, map_location="cpu")
            comparison = compare_states(dict(data=saved_state, step=expected_step),
                                        dict(data=self.train_dataloader.state_dict(), step=self.global_steps),
                                        tensor_encoder)
            write_event("trainer_restore", step=self.global_steps, components=comparison,
                        seconds=time.monotonic() - started)
            require_matches(comparison)
        import ray
        from dataclasses import asdict
        from ray.util.state import list_actors
        from ray.util import placement_group_table
        write_event("ray_placement", nodes=ray.nodes(), resources=ray.cluster_resources(),
                    available_resources=ray.available_resources(), placement_groups=placement_group_table(),
                    actors=[{key: value for key, value in asdict(actor).items()
                             if key in {"actor_id", "class_name", "pid", "node_id", "state", "name",
                                        "num_restarts", "required_resources"}}
                            for actor in list_actors(detail=True, limit=1000)])
        return result

    @wraps(original_batch)
    def batch(self, value, *args, **kwargs):
        """Record source indices before native prompt-batch preparation."""
        extras = value.non_tensor_batch.get("extra_info")
        # Validation also calls this method; only training indices form the audit.
        if extras is not None and all(entry.get("split") == "train" for entry in extras):
            indices = [entry["index"] for entry in extras]
            write_event("training_batch", step=self.global_steps, source_indices=indices)
        return original_batch(self, value, *args, **kwargs)

    trainer_class._save_checkpoint = saved
    trainer_class._load_checkpoint = loaded
    trainer_class._get_gen_batch = batch
    trainer_class._part7_recovery_audit = True
