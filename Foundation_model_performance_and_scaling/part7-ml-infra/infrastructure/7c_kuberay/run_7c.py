"""Run one native verl phase on existing KubeRay; no cluster creation or training loop."""
import argparse
from importlib import import_module
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
reference = import_module("infrastructure.7a_verl.run_7a")
recovery = import_module("infrastructure.7b_ray.run_7b")
bundle = import_module("infrastructure.7b_ray.checkpoint_bundle")
transfer = import_module("infrastructure.7c_kuberay.checkpoint_transfer")


def configure_phase(config, phase, output, total_steps=3, interrupt_step=2, checkpoint=None, manifest=None):
    """Modify resolved config for existing-cluster execution and native recovery.

    Inputs: plain config, phase, shared output/checkpoint paths and step bounds.
    Returns the mutated config. Reject invalid phases/limits; remove local Ray
    startup arguments. Enable existing 7B audits without changing GRPO semantics.
    """
    if phase not in {"baseline", "interrupt", "resume"}:
        raise ValueError("Unknown phase")
    if not 0 < interrupt_step < total_steps:
        raise ValueError("Require 0 < interruption step < total steps")
    if phase == "resume":
        if manifest is None or checkpoint is None:
            raise ValueError("Resume requires a verified manifest and checkpoint")
        recovery.apply_recovery_config(config, manifest, checkpoint, total_steps)
    else:
        config["trainer"].update(total_training_steps=total_steps, val_before_train=False,
            test_freq=total_steps, save_freq=interrupt_step if phase == "interrupt" else total_steps,
            resume_mode="disable", resume_from_path=None)
    config["trainer"].update(project_name="part7_7c", experiment_name=phase)
    config["actor_rollout_ref"]["model"]["external_lib"] = "infrastructure.7b_ray.worker_extension"
    env = config["ray_kwargs"]["ray_init"]["runtime_env"]["env_vars"]
    env.update(PART7_RUN_DIR=str(output), PART7_7B_AUDIT="1",
               PART7_INTERRUPT_STEP=str(interrupt_step if phase == "interrupt" else 0))
    config["ray_kwargs"]["ray_init"] = dict(address="auto", runtime_env=dict(env_vars=env))
    return config


def validate_identity(current, manifest):
    """Compare recovery identities without tensor loading or model computation.

    Inputs: newly recorded identity and optional verified manifest. Returns None;
    reject changed source, interpreter, packages, model bytes or workload hashes.
    Pod/runtime IDs may change and are deliberately outside these comparisons.
    """
    if manifest is None:
        return
    saved = manifest["identity"]
    for key in ("verl_commit", "python", "packages", "source_hashes", "model_files"):
        if current[key] != saved.get(key):
            raise ValueError(f"Recovery identity differs: {key}")
    for identity in (current, saved):
        if not identity.get("platform", {}).get("milestone") == "7c":
            raise ValueError("Use a checkpoint produced by this 7C launcher")
    if current["platform"].get("image") != saved["platform"].get("image"):
        raise ValueError("Recovery image digest differs")
    if {k: v["sha256"] for k, v in current["data_manifest"]["outputs"].items()} != {
            k: v["sha256"] for k, v in saved["data_manifest"]["outputs"].items()}:
        raise ValueError("Recovery data/reward/prompt bundle differs")


def validate_shared_paths(paths):
    """Require persistent /workspace paths before cloud preparation or training.

    Input: iterable of paths. Returns None; reject paths resolving outside the
    mounted workspace, including symlink escapes. CPU previews need not use PVCs.
    """
    for path in paths:
        if not Path(path).resolve().is_relative_to(Path("/workspace")):
            raise ValueError("Cloud data/model/output/checkpoint paths must be on /workspace")


def prepare_worker(model, model_id, revision, explicit_path):
    """Run on a two-GPU cloud actor: validate hardware and prepare shared model.

    Inputs: snapshot path and Hub model identity. Returns inventory and placement;
    imports Torch only remotely, downloads missing snapshots to PVC using a Secret
    file, and rejects wrong GPU visibility/pinned source. Never prints credentials.
    """
    import subprocess
    import torch
    from huggingface_hub import snapshot_download
    if not os.environ.get("POD_UID") or not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Preparation actor requires a Kubernetes worker with two GPUs")
    commit = subprocess.check_output(["git", "-C", str(ROOT / ".venv-7a/src/verl"), "rev-parse", "HEAD"], text=True).strip()
    if commit != reference.VERL_COMMIT:
        raise ValueError("Unexpected verl source commit")
    model = Path(model)
    if explicit_path:
        if revision or not (model / "config.json").is_file():
            raise ValueError("Explicit model path requires complete snapshot and no revision override")
    elif revision or not (model / "config.json").is_file():
        snapshot_download(repo_id=model_id, revision=revision, local_dir=str(model),
            token=Path("/run/secrets/huggingface/token").read_text().strip())
    if not (model / "config.json").is_file():
        raise FileNotFoundError("Model snapshot config is missing")
    import ray
    return dict(model_files=recovery.model_inventory(model),
                node_id=ray.get_runtime_context().get_node_id(),
                pod_uid=os.environ["POD_UID"], pod_name=os.environ.get("POD_NAME"),
                physical_node=os.environ.get("NODE_NAME"),
                gpu_inventory=subprocess.check_output(["nvidia-smi", "-L"], text=True))


def execute(args, config, data, model, manifest):
    """Attach to KubeRay and delegate one phase to native verl with 7B hooks.

    Inputs: CLI/config/data/model/verified manifest. Returns None; writes config,
    identities, status and a completed interruption manifest on shared storage.
    Does not provision or initialize a local Ray runtime; optional Hub retention
    uploads verified state before reporting phase success.
    Only a completed receipt plus PlannedInterruption counts as expected success.
    """
    import importlib.metadata
    import ray
    from omegaconf import OmegaConf
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
    from verl.trainer.main_ppo import TaskRunner, run_ppo
    if not os.environ.get("POD_UID") or not os.environ.get("NODE_NAME"):
        raise RuntimeError("Cloud execution requires the prepared Kubernetes pod")
    validate_shared_paths([args.output, args.data, model] + ([args.checkpoint] if args.checkpoint else []))
    if not os.environ.get("PART7_IMAGE"):
        raise RuntimeError("Prepared RayJob must record its image digest")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = dict(success=False, phase=args.phase, planned_interruption=False)
    preparer = None
    try:
        # Supply verl's defaults explicitly: native run_ppo skips initialization
        # (including its default environment merge) once Ray is already attached.
        runtime = OmegaConf.to_container(OmegaConf.merge(get_ppo_ray_runtime_env(),
                         config.ray_kwargs.ray_init.runtime_env), resolve=True)
        ray.init(address="auto", runtime_env=runtime)
        if ray.__version__ != "2.49.2":
            raise RuntimeError("Unexpected Ray version")
        nodes = [node for node in ray.nodes() if node["Alive"]]
        workers = [node for node in nodes if node["Resources"].get("GPU", 0) == 2]
        if len(workers) != 1 or sum(node["Resources"].get("GPU", 0) for node in nodes) != 2:
            raise RuntimeError("Require one live two-GPU worker")
        worker_id = workers[0]["NodeID"]

        @ray.remote(num_cpus=1, num_gpus=2)
        class Preparation:
            """Reserve both worker GPUs only while validating/preparing the snapshot."""
            def run(self):
                """Return shared snapshot hashes and worker placement, or propagate failure."""
                return prepare_worker(str(model), args.model_id, args.model_revision, args.model_path is not None)

        preparer = Preparation.options(scheduling_strategy=NodeAffinitySchedulingStrategy(worker_id, soft=False)).remote()
        prepared = ray.get(preparer.run.remote(), timeout=900)
        ray.kill(preparer)
        preparer = None
        identity = dict(verl_commit=reference.VERL_COMMIT, data_manifest=data,
            model_files=prepared["model_files"], model_path=str(model),
            requested_model_id=args.model_id, requested_model_revision=args.model_revision,
            python=sys.version, packages={dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()},
            source_hashes={path.relative_to(ROOT).as_posix(): reference.sha256_file(path)
                for folder in (ROOT / "workloads", ROOT / "infrastructure/7a_verl", ROOT / "infrastructure/7b_ray", ROOT / "infrastructure/7c_kuberay")
                for path in sorted(folder.rglob("*")) if path.is_file() and "__pycache__" not in path.parts},
            platform=dict(milestone="7c", image=os.environ["PART7_IMAGE"], head_pod_uid=os.environ["POD_UID"], worker=prepared))
        validate_identity(identity, manifest)
        (output / "identities.json").write_text(json.dumps(identity, indent=2))
        (output / "resolved_config.yaml").write_text(OmegaConf.to_yaml(config))

        class RecoveryTaskRunner(TaskRunner):
            """Install proven trainer audits inside native verl's coordinating actor."""
            def run(self, config):
                """Run native training with checkpoint interruption and restoration checks."""
                import_module("infrastructure.7b_ray.recovery_audit").install_trainer_hooks()
                return super().run(config)

        runner = ray.remote(num_cpus=1, num_gpus=0)(RecoveryTaskRunner).options(
            runtime_env=runtime, scheduling_strategy=NodeAffinitySchedulingStrategy(worker_id, soft=False))
        try:
            run_ppo(config, task_runner_class=runner)
            if args.phase == "interrupt":
                raise RuntimeError("Expected checkpoint-boundary interruption did not occur")
        except Exception as error:
            receipt_path = output / "planned_interruption.json"
            receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
            planned = import_module("infrastructure.7b_ray.recovery_audit").PlannedInterruption
            if not (args.phase == "interrupt" and isinstance(getattr(error, "cause", error), planned)
                    and receipt.get("completed") and receipt.get("step") == args.interrupt_step):
                raise
            bundle.create_manifest(Path(receipt["checkpoint"]), identity)
            status["planned_interruption"] = True
        if args.upload_checkpoint_repo:
            checkpoint = output / "checkpoints" / f"global_step_{args.interrupt_step if args.phase == 'interrupt' else args.total_steps}"
            if not (checkpoint / bundle.MANIFEST).exists():
                bundle.create_manifest(checkpoint, identity)
            receipt = transfer.upload(checkpoint, args.upload_checkpoint_repo, output.name)
            (output / "hf_checkpoint.json").write_text(json.dumps(receipt, indent=2))
            status["checkpoint_uploaded"] = True
        status["success"] = True
    except Exception as error:
        status["error_type"] = type(error).__name__
        raise
    finally:
        try:
            if preparer is not None:
                ray.kill(preparer)
        finally:
            ray.shutdown()  # Disconnect; KubeRay owns the cluster lifecycle.
            (output / "job_status.json").write_text(json.dumps(status, indent=2))


def main():
    """Parse shared paths, verify byte manifests and preview or execute one phase.

    Input: CLI with --check-config (CPU) or --cloud (Kubernetes). Returns None;
    preview prints YAML without Ray/Torch imports or model loading. Resume requires
    a verified completed checkpoint. Shared inputs must already exist on the PVC.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check-config", action="store_true")
    modes.add_argument("--cloud", action="store_true")
    parser.add_argument("--phase", choices=["baseline", "interrupt", "resume"], default="baseline")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--upload-checkpoint-repo")
    parser.add_argument("--restore-repo")
    parser.add_argument("--restore-revision")
    parser.add_argument("--data", type=Path, default=Path("/workspace/data/step2_math_final"))
    parser.add_argument("--model-id", default=reference.DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=3)
    parser.add_argument("--interrupt-step", type=int, default=2)
    parser.add_argument("--verl-source", type=Path, default=ROOT / ".venv-7a/src/verl")
    args = parser.parse_args()
    if args.cloud and not os.environ.get("POD_UID"):
        parser.error("Cloud mode requires Kubernetes pod identity")
    if args.phase == "resume" and args.checkpoint is None:
        parser.error("Resume requires --checkpoint")
    transfer.validate_options(args.upload_checkpoint_repo, args.restore_repo, args.restore_revision)
    if args.restore_repo:
        if args.phase != "resume" or not args.cloud:
            parser.error("Hub restore requires --cloud --phase resume")
        if args.output.exists():
            raise FileExistsError("Use a fresh output directory before downloading")
        transfer.restore(args.restore_repo, args.restore_revision, args.checkpoint)
    manifest = bundle.verify_manifest(args.checkpoint) if args.phase == "resume" else None
    data = reference.verify_bundle(args.data)
    model = (args.model_path or Path("/workspace/models") / args.model_id.replace("/", "--")).resolve()
    from omegaconf import OmegaConf
    config = reference.compose_config(args.verl_source.resolve(), args.data.resolve(), model, args.output.resolve())
    config = OmegaConf.create(configure_phase(OmegaConf.to_container(config, resolve=True), args.phase,
        args.output.resolve(), args.total_steps, args.interrupt_step, args.checkpoint, manifest))
    if args.check_config:
        print(OmegaConf.to_yaml(config))
    else:
        execute(args, config, data, model, manifest)


if __name__ == "__main__":
    main()
