"""Compose 7B resume configuration without model loading. Cloud launch is not exposed yet."""
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
    initial evaluation so future restore audits can inspect RNG first.

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
    """Compose and print a recovery configuration without launching training.

    Inputs:
        sys.argv supplies --check-config, checkpoint and optional data/model,
        source/output paths and final step limit.
    Returns:
        None; prints resolved YAML to stdout.
    How it works:
        Verify checkpoint/data hashes, compose the existing 7A YAML against
        upstream verl, then apply native resume settings. Only metadata and
        configuration are read; no tokenizer/model is loaded.
    Raises:
        ValueError: Integrity, data identity or recovery topology differs.
        ImportError: Hydra/OmegaConf required for composition are unavailable.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-config", action="store_true", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=4)
    parser.add_argument("--verl-source", type=Path, default=ROOT / ".venv-7a/src/verl")
    parser.add_argument("--data", type=Path, default=ROOT / "results/7a/step2_math_final")
    parser.add_argument("--model-path", type=Path, default=reference.discover_model_path())
    parser.add_argument("--output", type=Path, default=ROOT / "results/7b/resume_preview")
    args = parser.parse_args()
    manifest = bundle.verify_manifest(args.checkpoint)
    current = reference.verify_bundle(args.data)
    saved_outputs = manifest["identity"]["data_manifest"]["outputs"]
    if {k: v["sha256"] for k, v in current["outputs"].items()} != {k: v["sha256"] for k, v in saved_outputs.items()}:
        raise ValueError("Recovery data/reward/prompt bundle differs from checkpoint identity")
    from omegaconf import OmegaConf
    config = reference.compose_config(args.verl_source.resolve(), args.data.resolve(),
                                      args.model_path.resolve(), args.output.resolve())
    result = apply_recovery_config(OmegaConf.to_container(config, resolve=True), manifest,
                                   args.checkpoint, args.total_steps)
    print(OmegaConf.to_yaml(OmegaConf.create(result)))


if __name__ == "__main__":
    main()
