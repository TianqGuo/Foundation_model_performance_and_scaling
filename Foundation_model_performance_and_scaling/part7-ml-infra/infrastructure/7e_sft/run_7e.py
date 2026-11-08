"""Internal RayJob launcher for native 7E SFT; cloud packaging/E2E still pending.

CPU preview: part7E.sh train --check --bundle PATH
Future pod entry: run_7e.py --cloud --bundle /workspace/... --output /workspace/...
"""
from dataclasses import asdict, replace
from importlib import import_module
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
settings = import_module("infrastructure.7e_sft.config")
workload = import_module("workloads.sft_workload")
SNAPSHOT_RECEIPT = "part7_model_snapshot.json"


def read_bundle(root):
    """Verify a prepared bundle's bytes/config/order; return data, config and identity.

    Inputs: existing directory written by prepare.py. Rejects symlinks, changed
    adapter/files, split overlap and inconsistent provenance/counts. No model or
    tokenizer loading. Identity is the manifest byte hash for later checkpointing.
    """
    root = Path(root)
    manifest_path = root/"manifest.json"
    if root.is_symlink() or manifest_path.is_symlink():
        raise ValueError("Bundle symlinks are not accepted")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if manifest["schema_version"] != 1 or set(manifest["files"]) != {"train.jsonl","evaluation.jsonl"}:
        raise ValueError("Unsupported SFT bundle schema/files")
    adapter = hashlib.sha256(Path(workload.__file__).read_bytes()).hexdigest()
    if manifest["adapter_sha256"] != adapter:
        raise ValueError("SFT adapter changed; prepare a fresh data bundle")
    config = settings.SFTConfig(**manifest["config"]).validate()
    data = {}
    for split, count, indices_key in (("train",config.train_count,"train_indices"),
                                     ("evaluation",config.eval_count,"evaluation_indices")):
        path = root/(split+".jsonl")
        if path.is_symlink():
            raise ValueError("Bundle files must not be symlinks")
        content = path.read_bytes()
        identity = manifest["files"][path.name]
        if hashlib.sha256(content).hexdigest() != identity["sha256"]:
            raise ValueError("Prepared SFT file hash changed")
        rows = [json.loads(line) for line in content.splitlines()]
        if len(rows) != count or identity["rows"] != count or [row["source_index"] for row in rows] != manifest[indices_key]:
            raise ValueError("SFT split counts/order differ from provenance")
        if any(not isinstance(row.get(key),str) or not row[key].strip() for row in rows for key in ("prompt","response")):
            raise ValueError("Invalid SFT text")
        data[split] = rows
    prompts = [row["prompt"] for rows in data.values() for row in rows]
    source_indices = [row["source_index"] for rows in data.values() for row in rows]
    if len(set(prompts)) != len(prompts) or len(set(source_indices)) != len(source_indices):
        raise ValueError("Prepared split has duplicate/overlapping prompt groups")
    return data, config, hashlib.sha256(raw).hexdigest()


def snapshot_inventory(root):
    """Hash local model/tokenizer assets and validate complete safetensors references.

    Ignores Hub download-cache metadata and this module's receipt. Rejects symlinks,
    missing config/tokenizer/weights and shard paths escaping the model directory.
    Reads bytes only; never deserializes tensors or constructs a tokenizer/model.
    """
    root = Path(root)
    if root.is_symlink():
        raise ValueError("Model snapshot must not be a symlink")
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ".cache" in relative.parts or relative.as_posix() == SNAPSHOT_RECEIPT:
            continue
        if path.is_symlink():
            raise ValueError("Model snapshot contains a symlink")
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8*1024*1024), b""):
                    digest.update(chunk)
            files[relative.as_posix()] = digest.hexdigest()
    if not {"config.json","tokenizer_config.json","tokenizer.json"}.issubset(files):
        raise ValueError("Complete model config/tokenizer assets required")
    weights = [name for name in files if name.endswith(".safetensors")]
    if not weights:
        raise ValueError("Safetensors model weights required")
    index = root/"model.safetensors.index.json"
    if index.exists():
        references = set(json.loads(index.read_text())["weight_map"].values())
        if not references or any(not isinstance(name,str) or not name.endswith(".safetensors") or name not in files for name in references):
            raise ValueError("Missing or unsafe model shard reference")
    return files


def verify_snapshot(root, config):
    """Verify pinned repo/commit and every local snapshot asset; return receipt."""
    config.validate(cloud=True)
    receipt_path = Path(root)/SNAPSHOT_RECEIPT
    if receipt_path.is_symlink():
        raise ValueError("Snapshot receipt must not be a symlink")
    receipt = json.loads(receipt_path.read_text())
    if (receipt["repo_id"],receipt["revision"]) != (config.model_repo,config.model_revision):
        raise ValueError("Snapshot identity differs from requested model")
    if receipt["files"] != snapshot_inventory(root):
        raise ValueError("Model snapshot integrity check failed")
    return receipt


def stage_model(root, config, api):
    """Download an immutable Hub snapshot on cloud, hash it and publish atomically.

    Inputs: fresh or previously completed PVC snapshot directory, pinned model
    config and authenticated Hub API. Reuses same-identity partial downloads;
    records exact assets without loading model tensors. Returns snapshot receipt.
    Caller must enforce cloud/pod context before supplying the API/network client.
    """
    config.validate(cloud=True)
    root = Path(root)
    if root.exists():
        return verify_snapshot(root,config)
    if api.model_info(config.model_repo,revision=config.model_revision).sha != config.model_revision:
        raise ValueError("Hub did not resolve requested model commit")
    partial = root.with_name(root.name+".download")
    identity = dict(repo_id=config.model_repo,revision=config.model_revision)
    source = partial/"part7_download_source.json"
    if partial.exists():
        if json.loads(source.read_text()) != identity:
            raise ValueError("Partial snapshot belongs to another model")
    else:
        partial.mkdir(parents=True)
        source.write_text(json.dumps(identity))
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=config.model_repo, revision=config.model_revision, token=api.token,
        local_dir=str(partial), allow_patterns=["*.json","*.safetensors","*.model","*.txt","*.jinja","*.tiktoken"])
    receipt = dict(**identity, files=snapshot_inventory(partial))
    (partial/SNAPSHOT_RECEIPT).write_text(json.dumps(receipt,indent=2)+"\n")
    partial.rename(root)
    return receipt


def require_cloud_pod():
    """Reject local/host training; require explicit Kubernetes pod identity and image pin."""
    if not os.environ.get("POD_UID") or not os.environ.get("NODE_NAME"):
        raise RuntimeError("7E training runs only inside the prepared cloud RayJob")
    if os.environ.get("RAY_TRAIN_V2_ENABLED", "1") != "1":
        raise ValueError("7E requires the reviewed Ray Train V2 API")
    image = os.environ.get("PART7_IMAGE", "")
    if not re.fullmatch(r"[a-z0-9][a-z0-9./:_-]*@sha256:[0-9a-f]{64}",image):
        raise ValueError("Cloud training requires the actual immutable image identity")
    return image


def launch(bundle, output, *, model_revision=None):
    """Run native TorchTrainer on the existing cloud Ray cluster; retain diagnostics.

    Resolves/pins model revision on cloud, stages model automatically and enforces
    STRICT_PACK for both GPU ranks. No Ray Train auto-retry hides missing recovery.
    Returns final metrics after worker success; writes receipts and preserves
    failure artifacts. No checkpoint/recovery claim or automatic cleanup is made.
    """
    image = require_cloud_pod()
    output, bundle = Path(output).resolve(), Path(bundle).resolve()
    if not output.is_relative_to(Path("/workspace")) or not bundle.is_relative_to(Path("/workspace")):
        raise ValueError("SFT data and results must use the shared PVC")
    _, config, bundle_hash = read_bundle(bundle)
    if config.model_revision is not None and model_revision is not None and config.model_revision != model_revision:
        raise ValueError("CLI revision differs from prepared configuration")
    runtime = import_module("infrastructure.7e_sft.runtime_check").check_runtime()
    api = import_module("infrastructure.7c_kuberay.checkpoint_transfer").credentials()
    requested_revision = model_revision or config.model_revision
    if requested_revision is not None:
        replace(config,model_revision=requested_revision).validate(cloud=True)
    info = api.model_info(config.model_repo,revision=requested_revision)
    config = replace(config,model_revision=info.sha).validate(cloud=True)
    output.mkdir(parents=True,exist_ok=False)
    identity = hashlib.sha256((config.model_repo+"@"+config.model_revision).encode()).hexdigest()[:24]
    model_path = Path("/workspace/models/7e")/identity
    snapshot = stage_model(model_path,config,api)
    (output/"launch.json").write_text(json.dumps(dict(config=asdict(config), bundle_sha256=bundle_hash,
        image=image, runtime=runtime, model=snapshot, recovery_verified=False),indent=2)+"\n")
    import ray
    from ray.train import ScalingConfig, RunConfig, FailureConfig
    from ray.train.torch import TorchTrainer, TorchConfig
    ray.init(address="auto")
    try:
        trainer = TorchTrainer(
            train_loop_per_worker=import_module("infrastructure.7e_sft.worker").train_loop_per_worker,
            train_loop_config=dict(config=asdict(config), bundle=str(bundle), bundle_sha256=bundle_hash,
                model_path=str(model_path), output=str(output), image=image),
            scaling_config=ScalingConfig(num_workers=2,use_gpu=True,resources_per_worker={"CPU":4,"GPU":1},
                                         placement_strategy="STRICT_PACK"),
            torch_config=TorchConfig(backend="nccl",timeout_s=300),
            run_config=RunConfig(name="native_sft",storage_path=str(output/"ray_train"),
                                 failure_config=FailureConfig(max_failures=0, controller_failure_limit=0, max_preemption_failures=0)))
        result = trainer.fit()
        metrics = dict(step=result.metrics.get("step"), held_out_loss=result.metrics.get("held_out_loss"),
                       recovery_verified=False, scope="native SFT worker; recovery/E2E acceptance pending")
        (output/"training_result.json").write_text(json.dumps(metrics,indent=2)+"\n")
        return metrics
    finally:
        ray.shutdown()


def main():
    """Parse a CPU preview or explicit internal cloud-pod training invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check",action="store_true")
    mode.add_argument("--cloud",action="store_true")
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--model-revision")
    args = parser.parse_args()
    if args.check:
        _, config, identity = read_bundle(args.bundle)
        print(json.dumps(dict(**config.preview(), bundle_sha256=identity,
            placement="STRICT_PACK; two whole GPUs on one Ray node", checkpoint_implemented=False),indent=2))
    else:
        if args.output is None:
            parser.error("Cloud invocation requires a fresh --output path")
        print(json.dumps(launch(args.bundle,args.output,model_revision=args.model_revision),indent=2))


if __name__ == "__main__":
    main()
