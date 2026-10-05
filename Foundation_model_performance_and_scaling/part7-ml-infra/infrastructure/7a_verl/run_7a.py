"""Prepare/check config locally; launch only on an explicitly selected cloud host.

python infrastructure/7a_verl/run_7a.py --check-config --verl-source /path/to/verl
python infrastructure/7a_verl/run_7a.py --cloud --model-path /path/to/model
See documents/7A_RUNBOOK.md. No cloud provisioning happens in this runner.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
VERL_COMMIT = "bec9ef74768dd201881cd4e54cd0385e87caae27"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"
sys.path.insert(0, str(ROOT))


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_model_path():
    candidates = [ROOT / "assets/models/Qwen2.5-Math-1.5B",
                  ROOT / "assets/Qwen2.5-Math-1.5B",
                  Path("/data/a5-alignment/models/Qwen2.5-Math-1.5B")]
    return next((path for path in candidates if (path / "config.json").is_file()), candidates[0])


def ensure_model_snapshot(model, model_id, revision, *, cloud, explicit_path=False):
    """Reuse local snapshots; download missing defaults only after cloud checks."""
    if not cloud:
        raise RuntimeError("Model preparation is cloud-only")
    if (model / "config.json").is_file() and revision is None:
        return
    if explicit_path:
        if not (model / "config.json").is_file():
            raise FileNotFoundError(f"Explicit --model-path must contain an existing snapshot: {model}")
        raise ValueError("--model-revision applies to automatic downloads; omit --model-path to use it")
    from huggingface_hub import snapshot_download
    print(f"Preparing {model_id} on cloud at {model}", flush=True)
    snapshot_download(repo_id=model_id, revision=revision, local_dir=str(model))
    if not (model / "config.json").is_file():
        raise FileNotFoundError(f"Downloaded snapshot has no config.json: {model}")


def compose_config(source, data, model, output):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(ROOT / "infrastructure/7a_verl/config"), version_base="1.3"):
        config = compose(config_name="grpo_smoke", overrides=[
            f"hydra.searchpath=[file://{source / 'verl/trainer/config'}]"])
    OmegaConf.set_struct(config, False)
    config.data.train_files = [str(data / "train.parquet")]
    config.data.val_files = [str(data / "validation.parquet")]
    config.actor_rollout_ref.model.path = str(model)
    config.actor_rollout_ref.model.custom_chat_template = (data / "raw_prompt.jinja").read_text()
    config.actor_rollout_ref.rollout.agent.agent_loop_config_path = str(ROOT / "infrastructure/7a_verl/config/r1_agents.yaml")
    config.reward.custom_reward_function = dict(path=str(ROOT / "workloads/math_workload.py"),
        name="compute_score", reward_kwargs=dict(grader_path=str(data / "reference/drgrpo_grader.py")))
    config.trainer.default_local_dir = str(output / "checkpoints")
    config.trainer.rollout_data_dir = str(output / "rollouts")
    config.trainer.validation_data_dir = str(output / "validation")
    config.ray_kwargs.ray_init.runtime_env = dict(env_vars=dict(
        PYTHONPATH=os.pathsep.join([str(ROOT), str(source)]), PART7_RUN_DIR=str(output),
        PART7_FROZEN_LOGPROB_TOLERANCE="0.02", TOKENIZERS_PARALLELISM="false"))
    # Short path avoids Ray's Unix-domain socket length limit.
    config.ray_kwargs.ray_init._temp_dir = "/tmp/r7_" + hashlib.sha256(str(output).encode()).hexdigest()[:12]
    OmegaConf.resolve(config)
    return config


def verify_bundle(data):
    manifest = json.loads((data / "manifest.json").read_text())
    # Check each transferred artifact, rather than relying on source paths on cloud.
    for relative, entry in manifest["outputs"].items():
        path = data / relative
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Prepared artifact changed: {path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-config", action="store_true")
    mode.add_argument("--cloud", action="store_true", help="explicit cloud-host assertion; loads models")
    parser.add_argument("--verl-source", type=Path, default=ROOT / ".venv-7a/src/verl")
    parser.add_argument("--data", type=Path, default=ROOT / "results/7a/step2_math_final")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face repository for automatic cloud download")
    parser.add_argument("--model-revision", help="optional repository commit/tag for automatic cloud download")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    default_model = (discover_model_path() if args.model_id == DEFAULT_MODEL_ID else
                     ROOT / "assets/models" / args.model_id.replace("/", "--"))
    source, data, model = (path.resolve() for path in
                           (args.verl_source, args.data, args.model_path or default_model))
    output = (args.output or ROOT / "results/7a" / datetime.now(timezone.utc).strftime("smoke_%Y%m%dT%H%M%S%fZ")).resolve()
    manifest = verify_bundle(data)
    config = compose_config(source, data, model, output)
    from omegaconf import OmegaConf
    if args.check_config:
        print(OmegaConf.to_yaml(config))
        return
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != VERL_COMMIT:
        raise ValueError("Unexpected verl commit")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Expose exactly two cloud GPUs for this smoke")
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    ensure_model_snapshot(model, args.model_id, args.model_revision,
                          cloud=args.cloud, explicit_path=args.model_path is not None)
    output.mkdir(parents=True, exist_ok=False)
    (output / "resolved_config.yaml").write_text(OmegaConf.to_yaml(config))
    evidence = dict(verl_commit=commit, python=sys.version, data_manifest=manifest,
        model_path=str(model), download_model_id=args.model_id if args.model_path is None else None,
        requested_model_revision=args.model_revision,
        packages={dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()},
        gpu_inventory=subprocess.check_output(["nvidia-smi", "-L"], text=True),
        source_hashes={str(path.relative_to(ROOT)): sha256_file(path)
                      for folder in (ROOT / "workloads", ROOT / "infrastructure/7a_verl")
                      for path in sorted(folder.rglob("*")) if path.is_file() and "__pycache__" not in path.parts},
        model_files={str(path.relative_to(model)): dict(bytes=path.stat().st_size,
                     sha256=sha256_file(path))
                     for path in sorted(model.rglob("*")) if path.is_file()})
    (output / "identities.json").write_text(json.dumps(evidence, indent=2))
    sys.path.insert(0, str(source))
    from verl.trainer.main_ppo import run_ppo
    import ray
    status = dict(success=False)
    try:
        run_ppo(config)
        status["success"] = True
    finally:
        (output / "job_status.json").write_text(json.dumps(status))
        ray.shutdown()
        logs = Path(config.ray_kwargs.ray_init._temp_dir) / "session_latest/logs"
        if logs.is_dir():
            shutil.copytree(logs, output / "ray_logs")


if __name__ == "__main__":
    main()
