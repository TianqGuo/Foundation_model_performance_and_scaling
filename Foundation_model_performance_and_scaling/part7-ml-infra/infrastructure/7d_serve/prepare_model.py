"""Cloud init-container download of a pinned exported model; no model execution."""
import argparse
import fcntl
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
settings = import_module("infrastructure.7d_serve.config")
RECEIPT = "part7_snapshot.json"


def file_digest(path):
    """Stream a regular file into SHA256; return hex digest without parsing tensors."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root):
    """Validate minimal HF export layout and return content hashes of snapshot files.

    Input: downloaded directory. Checks config/tokenizer and complete safetensors
    index or monolithic weights; rejects links/escapes, no tensor/model loading.
    """
    json.loads(settings.safe_file(root,"config.json").read_text())
    json.loads(settings.safe_file(root,"tokenizer_config.json").read_text())
    if not any((root / name).is_file() for name in ("tokenizer.json","tokenizer.model")):
        raise ValueError("Export lacks tokenizer assets")
    index = root / "model.safetensors.index.json"
    if index.exists():
        mapping = json.loads(settings.safe_file(root,"model.safetensors.index.json").read_text())["weight_map"]
        if not mapping:
            raise ValueError("Empty weight inventory")
        for name in set(mapping.values()):
            settings.safe_file(root,name)
    else:
        settings.safe_file(root,"model.safetensors")
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if ".cache" in path.relative_to(root).parts or relative in {RECEIPT,"part7_download_source.json"}:
            continue
        if path.is_symlink():
            raise ValueError("Snapshot symlink refused")
        if path.is_file():
            files[relative] = file_digest(settings.safe_file(root,relative))
    return files


def verify_saved(root, config):
    """Verify a previously staged snapshot's identity and all file hashes.

    Inputs: snapshot directory and model settings. Returns receipt or rejects
    stale/tampered/incomplete state; does not contact Hub or load the model.
    """
    receipt = json.loads(settings.safe_file(root,RECEIPT).read_text())
    if receipt["repo_id"] != config.model_repo or receipt["revision"] != config.model_revision:
        raise ValueError("Existing model snapshot identity differs")
    if inventory(root) != receipt["files"]:
        raise ValueError("Staged model file inventory changed")
    return receipt


def prepare(config, destination):
    """Download/check pinned HF export inside a Kubernetes pod and publish atomically.

    Inputs: model settings, fresh or already verified PVC destination. Reads mounted
    Secret through the 7C credential helper; locks concurrent head/worker init pods.
    Retains resumable partial downloads with matching identity after failure.
    Returns receipt; no GPU inference, tokenizer construction or model loading.
    """
    config.validate()
    destination = destination.resolve()
    if not destination.is_relative_to(Path("/workspace")):
        raise ValueError("Stage models only on the persistent workspace")
    api = import_module("infrastructure.7c_kuberay.checkpoint_transfer").credentials()
    destination.parent.mkdir(parents=True,exist_ok=True)
    with (destination.parent / (destination.name + ".lock")).open("a") as lock:
        deadline = time.monotonic() + 600
        while True:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Model staging lock timed out")
                time.sleep(2)
        if destination.exists():
            return verify_saved(destination,config)
        info = api.model_info(config.model_repo,revision=config.model_revision)
        if info.sha != config.model_revision:
            raise ValueError("Hub did not resolve the pinned export commit")
        partial = destination.parent / (destination.name + ".download")
        identity = dict(repo_id=config.model_repo,revision=config.model_revision)
        if partial.exists():
            if json.loads((partial/"part7_download_source.json").read_text()) != identity:
                raise ValueError("Partial download belongs to another model")
        else:
            partial.mkdir()
            (partial/"part7_download_source.json").write_text(json.dumps(identity))
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=config.model_repo,revision=config.model_revision,token=api.token,
                          local_dir=str(partial),allow_patterns=["*.json","*.safetensors","*.model",
                          "*.txt","*.jinja","*.tiktoken","README.md","LICENSE*"])
        receipt = dict(**identity,files=inventory(partial),source="7A weights-only serving export")
        (partial/RECEIPT).write_text(json.dumps(receipt,indent=2) + "\n")
        (partial/"part7_download_source.json").unlink()
        partial.rename(destination)
        return receipt


def main():
    """Parse explicit cloud init arguments; download only inside prepared K8s pods.

    No local mode loads/downloads model checkpoints. A pod identity and mounted
    Secret are required by credentials(); printed receipt contains no credentials.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud", action="store_true", required=True)
    parser.add_argument("--repo",default=settings.DEFAULT_REPO)
    parser.add_argument("--revision",default=settings.DEFAULT_REVISION)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    receipt = prepare(settings.ServingConfig(model_repo=args.repo,model_revision=args.revision),args.output)
    print(json.dumps(receipt,indent=2))


if __name__ == "__main__":
    main()
