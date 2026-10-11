"""Preview native recovery or run one bounded cloud phase; see documents/7B_RUNBOOK.md."""
import argparse
from importlib import import_module
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
reference = import_module("infrastructure.7a_verl.run_7a")
bundle = import_module("infrastructure.7b_ray.checkpoint_bundle")


def apply_recovery_config(config, manifest, checkpoint, total_steps):
    """Apply native resume settings to a composed 7A configuration.

    Validate the single-node FSDP2 topology, then set the resume path, final
    step limit, per-step checkpointing and final-step evaluation. Disable
    initial evaluation so restore audits inspect RNG before generation.

    Args:
        config (dict): Mutable, resolved 7A configuration.
        manifest (dict): Verified checkpoint manifest containing world_size/step.
        checkpoint (Path | str): Restored checkpoint directory.
        total_steps (int): Final global-step limit, greater than the saved step.
    Returns:
        dict: The same configuration, modified in place.
    Raises:
        ValueError: Training topology differs or the step limit is invalid.
    """
    if config["trainer"]["nnodes"] != 1 or config["trainer"]["n_gpus_per_node"] != manifest["world_size"]:
        raise ValueError("Recovery topology differs from the initial single-node checkpoint")
    actor = config["actor_rollout_ref"]["actor"]
    if actor["strategy"] != "fsdp2" or actor["fsdp_config"]["fsdp_size"] != manifest["world_size"]:
        raise ValueError("Recovery requires matching FSDP2 sharding")
    config["trainer"].update(bundle.resume_settings(manifest, checkpoint, total_steps))
    config["trainer"].update(save_freq=1, test_freq=total_steps, val_before_train=False)
    return config


def main():
    """Parse paths/phase, verify input metadata and compose the existing 7A YAML.

    Input: sys.argv selects CPU --check-config or explicit --cloud execution.
    Returns None; preview prints YAML, cloud mode delegates to execute_phase.
    Invalid topology, hashes or step limits fail before model preparation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-config", action="store_true")
    mode.add_argument("--cloud", action="store_true")
    parser.add_argument("--phase", choices=["baseline", "interrupt", "resume"], default="resume")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--interrupt-step", type=int, default=2)
    parser.add_argument("--model-id", default=reference.DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision")
    parser.add_argument("--total-steps", type=int, default=3)
    parser.add_argument("--verl-source", type=Path, default=ROOT / ".venv-7a/src/verl")
    parser.add_argument("--data", type=Path, default=ROOT / "results/7a/step2_math_final")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results/7b/resume_preview")
    args = parser.parse_args()
    if args.phase == "resume" and args.checkpoint is None:
        parser.error("resume requires --checkpoint")
    manifest = bundle.verify_manifest(args.checkpoint) if args.phase == "resume" else None
    current = reference.verify_bundle(args.data)
    if manifest:
        saved_outputs = manifest["identity"]["data_manifest"]["outputs"]
        if {k: v["sha256"] for k, v in current["outputs"].items()} != {k: v["sha256"] for k, v in saved_outputs.items()}:
            raise ValueError("Recovery data/reward/prompt bundle differs from checkpoint identity")
    model = args.model_path or (reference.discover_model_path() if args.model_id == reference.DEFAULT_MODEL_ID
                               else ROOT / "assets/models" / args.model_id.replace("/", "--"))
    from omegaconf import OmegaConf
    config = reference.compose_config(args.verl_source.resolve(), args.data.resolve(),
                                      model.resolve(), args.output.resolve())
    result = OmegaConf.to_container(config, resolve=True)
    if manifest:
        apply_recovery_config(result, manifest, args.checkpoint, args.total_steps)
    else:
        if args.total_steps <= args.interrupt_step or args.interrupt_step < 1:
            parser.error("total steps must exceed positive interruption step")
        result["trainer"].update(total_training_steps=args.total_steps, val_before_train=False,
                                  test_freq=args.total_steps,
                                  save_freq=args.interrupt_step if args.phase == "interrupt" else args.total_steps)
    result["trainer"].update(project_name="part7_7b", experiment_name=args.phase)
    result["actor_rollout_ref"]["model"]["external_lib"] = "infrastructure.7b_ray.worker_extension"
    result["ray_kwargs"]["ray_init"]["runtime_env"]["env_vars"].update(
        PART7_7B_AUDIT="1", PART7_INTERRUPT_STEP=str(args.interrupt_step if args.phase == "interrupt" else 0))
    config = OmegaConf.create(result)
    if args.check_config:
        print(OmegaConf.to_yaml(config))
        return
    execute_phase(args, config, current, model.resolve(), manifest)


def model_inventory(model):
    """Return sizes/hashes of model snapshot files, excluding Hub cache metadata.

    Input: model directory. Returns a relative-name dictionary; reads bytes only.
    Used on cloud to reject a different initialization snapshot during recovery.
    """
    return {path.relative_to(model).as_posix(): dict(bytes=path.stat().st_size,
                                                  sha256=reference.sha256_file(path))
            for path in sorted(model.rglob("*")) if path.is_file() and ".cache" not in path.relative_to(model).parts}


def execute_phase(args, config, data_identity, model, manifest):
    """Run one cloud phase through a custom TaskRunner and native verl workers.

    Inputs: parsed arguments, composed configuration, bundle identity, model path
    and optional verified manifest. Returns None; writes config/identities/status,
    Ray logs and recovery events. Checks two GPUs/pinned source before preparation.
    Expected interruption succeeds only with the exact validated receipt and
    PlannedInterruption cause; ordinary exceptions propagate. No provisioning.
    """
    import importlib.metadata
    import json
    import shutil
    import subprocess
    from omegaconf import OmegaConf
    source, output = args.verl_source.resolve(), args.output.resolve()
    bundle.cloud_guard()
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != reference.VERL_COMMIT:
        raise ValueError("Unexpected verl commit")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Expose exactly two cloud GPUs")
    if output.exists():
        raise FileExistsError(f"Use a fresh run directory: {output}")
    reference.ensure_model_snapshot(model, args.model_id, args.model_revision, cloud=True,
                                    explicit_path=args.model_path is not None)
    inventory = model_inventory(model)
    if manifest and inventory != manifest["identity"]["model_files"]:
        raise ValueError("Recovery initialization model differs from checkpoint identity")
    output.mkdir(parents=True)
    (output / "resolved_config.yaml").write_text(OmegaConf.to_yaml(config))
    identity = dict(verl_commit=commit, data_manifest=data_identity, model_files=inventory,
                    model_path=str(model), requested_model_id=args.model_id,
                    requested_model_revision=args.model_revision, python=sys.version,
                    packages={dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()},
                    gpu_inventory=subprocess.check_output(["nvidia-smi", "-L"], text=True),
                    source_hashes={path.relative_to(ROOT).as_posix(): reference.sha256_file(path)
                                   for folder in (ROOT / "workloads", ROOT / "infrastructure/7a_verl", ROOT / "infrastructure/7b_ray")
                                   for path in sorted(folder.rglob("*"))
                                   if path.is_file() and "__pycache__" not in path.parts})
    (output / "identities.json").write_text(json.dumps(identity, indent=2))
    if manifest and any(identity[key] != manifest["identity"].get(key)
                        for key in ("python", "packages", "source_hashes")):
        raise ValueError("Recovery requires the same Python/packages/workload and infrastructure source")
    sys.path.insert(0, str(source))
    import ray
    from verl.trainer.main_ppo import TaskRunner, run_ppo
    PlannedInterruption = import_module("infrastructure.7b_ray.recovery_audit").PlannedInterruption

    class RecoveryTaskRunner(TaskRunner):
        """Install driver-side hooks inside the Ray actor before native execution."""
        def run(self, config):
            """Input: verl config. Return native run result with audit hooks active."""
            import_module("infrastructure.7b_ray.recovery_audit").install_trainer_hooks()
            return super().run(config)

    status = dict(success=False, phase=args.phase, planned_interruption=False)
    try:
        run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(RecoveryTaskRunner))
        if args.phase == "interrupt":
            raise RuntimeError("Interruption phase finished without planned interruption")
        status["success"] = True
    except Exception as error:
        receipt_path = output / "planned_interruption.json"
        receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
        cause = getattr(error, "cause", error)
        if (args.phase == "interrupt" and isinstance(cause, PlannedInterruption)
                and receipt.get("completed") and receipt.get("step") == args.interrupt_step):
            status.update(success=True, planned_interruption=True)
        else:
            status["error_type"] = type(error).__name__
            raise
    finally:
        (output / "job_status.json").write_text(json.dumps(status, indent=2))
        ray.shutdown()
        logs = Path(config.ray_kwargs.ray_init._temp_dir) / "session_latest/logs"
        if logs.is_dir():
            shutil.copytree(logs, output / "ray_logs")


if __name__ == "__main__":
    main()
