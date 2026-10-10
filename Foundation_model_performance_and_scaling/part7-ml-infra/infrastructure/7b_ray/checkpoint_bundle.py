"""Full-checkpoint integrity without tensor deserialization; Hub transfers are cloud-only."""
from __future__ import annotations

import argparse
import hashlib
from importlib import import_module
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import time
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

reference = import_module("infrastructure.7a_verl.run_7a")
MANIFEST = "checkpoint_manifest.json"


def digest(path):
    """Return a file's SHA-256 using bounded-memory reads.

    Args:
        path (Path): Existing file to read in 8 MiB blocks.
    Returns:
        str: Hexadecimal SHA-256 digest; file contents are not interpreted.
    Raises:
        OSError: The file cannot be opened or read.
    """
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def payload_files(root):
    """Collect checkpoint payload paths without reading tensor contents.

    Args:
        root (Path): Checkpoint directory. Only data.pt and files under actor/
            are payloads; the manifest and Hub .cache metadata are excluded.
    Returns:
        dict[str, Path]: Relative POSIX file names mapped to local paths.
    Raises:
        ValueError: A symlink or unexpected payload location is encountered.
    """
    if root.is_symlink():
        raise ValueError("Checkpoint symlink rejected")
    paths = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"Checkpoint symlink rejected: {relative}")
        if path.is_file() and relative != MANIFEST and ".cache" not in path.relative_to(root).parts:
            if relative != "data.pt" and not relative.startswith("actor/"):
                raise ValueError(f"Unexpected checkpoint payload: {relative}")
            paths[relative] = path
    return paths


def layout(root, world_size):
    """Validate the checkpoint step, FSDP2 topology and required files.

    Read fsdp_config.json, require every rank's model/optimizer/extra shard,
    and check data/tokenizer files. This checks layout, not tensor validity.

    Args:
        root (Path): Directory named global_step_N.
        world_size (int): Expected number of training ranks.
    Returns:
        tuple[int, dict[str, Path]]: Saved step and validated payload inventory.
    Raises:
        ValueError: Topology, naming, file completeness or shard layout is invalid.
        OSError: Required metadata cannot be read.
    """
    match = re.fullmatch(r"global_step_([1-9][0-9]*)", root.name)
    if not match:
        raise ValueError("Checkpoint directory must be named global_step_N")
    config = json.loads((root / "actor/fsdp_config.json").read_text())
    if config != {"FSDP_version": 2, "world_size": world_size}:
        raise ValueError("FSDP2/world-size mismatch")
    required = {"data.pt", "actor/fsdp_config.json", "actor/huggingface/config.json",
                "actor/huggingface/tokenizer_config.json", "actor/huggingface/tokenizer.json"}
    for rank in range(world_size):
        for kind in ("model", "optim", "extra_state"):
            required.add(f"actor/{kind}_world_size_{world_size}_rank_{rank}.pt")
    files = payload_files(root)
    if not required.issubset(files):
        raise ValueError(f"Missing checkpoint files: {sorted(required - files.keys())}")
    for name, path in files.items():
        if path.stat().st_size == 0:
            raise ValueError(f"Empty checkpoint file: {name}")
        if name.endswith(".pt") and name.startswith("actor/") and name not in required:
            raise ValueError(f"Unexpected rank/state shard: {name}")
    return int(match[1]), files


def create_manifest(root, identity, world_size=2):
    """Record a completed checkpoint's identity and file hashes atomically.

    Validate layout and the producer's latest-step marker, hash each payload,
    then rename a temporary JSON file to checkpoint_manifest.json.

    Args:
        root (Path | str): Producer checkpoint directory named global_step_N.
        identity (dict): Producer identities containing the pinned verl commit,
            model_files and data_manifest.
        world_size (int): Expected rank count; defaults to two.
    Returns:
        dict: Manifest with schema, step, topology, identity and file hashes.
    Raises:
        ValueError: Layout, completion marker or identity is incompatible.
        FileExistsError: A manifest or temporary file already exists.
    Side effects:
        Writes the manifest; failed writes can leave the temporary file.
    """
    root = Path(root)
    step, files = layout(root, world_size)
    marker = root.parent / "latest_checkpointed_iteration.txt"
    if not marker.is_file() or int(marker.read_text().strip()) < step:
        raise ValueError("Checkpoint completion marker missing or behind saved step")
    if identity.get("verl_commit") != reference.VERL_COMMIT:
        raise ValueError("Unexpected verl commit")
    if not identity.get("model_files") or not identity.get("data_manifest"):
        raise ValueError("Model/data identities required")
    manifest = dict(schema=1, step=step, world_size=world_size, fsdp_version=2,
                    identity=identity, created_at=time.time(),
                    files={name: dict(bytes=path.stat().st_size, sha256=digest(path))
                           for name, path in sorted(files.items())})
    target = root / MANIFEST
    if target.exists():
        raise FileExistsError("Manifest exists; verify it instead of overwriting")
    temporary = root / (MANIFEST + ".tmp")
    with temporary.open("x") as stream:
        json.dump(manifest, stream, indent=2)
    temporary.replace(target)
    return manifest


def verify_manifest(root, world_size=2):
    """Verify a checkpoint against its saved manifest without loading tensors.

    Validate identity/topology and relative paths, compare the complete file
    inventory, then recompute every payload's size and SHA-256.

    Args:
        root (Path | str): Producer or restored global_step_N directory.
        world_size (int): Expected rank count; defaults to two.
    Returns:
        dict: Verified manifest. No producer completion marker is required
            on a restored copy because it is outside the uploaded directory.
    Raises:
        ValueError: Identity, topology, inventory or integrity checks fail.
        OSError: Manifest or payload cannot be read.
    """
    root = Path(root)
    manifest = json.loads((root / MANIFEST).read_text())
    if manifest.get("schema") != 1 or manifest.get("world_size") != world_size or manifest.get("fsdp_version") != 2:
        raise ValueError("Manifest schema/topology mismatch")
    identity = manifest.get("identity", {})
    if identity.get("verl_commit") != reference.VERL_COMMIT or not identity.get("model_files") or not identity.get("data_manifest"):
        raise ValueError("Checkpoint identity missing or incompatible")
    for name in manifest["files"]:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or str(path) != name:
            raise ValueError("Unsafe manifest path")
    step, files = layout(root, world_size)
    if step != manifest.get("step") or set(files) != set(manifest["files"]):
        raise ValueError("Manifest step/file inventory mismatch")
    for name, path in files.items():
        expected = manifest["files"][name]
        if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError(f"Checkpoint integrity failure: {name}")
    return manifest


def resume_settings(manifest, checkpoint, total_steps):
    """Build verl settings that continue beyond the checkpoint step.

    Args:
        manifest (dict): Previously verified manifest with the saved step.
        checkpoint (Path | str): Directory from which verl will load state.
        total_steps (int): Final global-step limit, not additional step count.
    Returns:
        dict: Native resume_path settings with an absolute path, step limit
            and checkpoint deletion disabled. No training is launched.
    Raises:
        ValueError: The final step limit does not exceed the saved step.
    """
    if total_steps <= manifest["step"]:
        raise ValueError("Total training steps must exceed checkpoint step")
    return dict(resume_mode="resume_path", resume_from_path=str(Path(checkpoint).resolve()),
                total_training_steps=total_steps, del_local_ckpt_after_load=False)


def cloud_guard():
    """Check GPU availability before the explicitly cloud-only transfer path.

    Inputs:
        No arguments; reads physical GPU names through nvidia-smi.
    Returns:
        None when exactly two GPUs are listed. GPU count alone does not prove
        that the machine is cloud-hosted; CLI callers also assert --cloud.
    Raises:
        RuntimeError: The GPU count differs from the initial 7B topology.
        OSError or subprocess.CalledProcessError: nvidia-smi fails.
    """
    inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True)
    if len(inventory.strip().splitlines()) != 2:
        raise RuntimeError("Initial 7B transfer workflow requires the two-GPU cloud host")


def upload(root, repo_id):
    """Upload a verified full checkpoint directly to a private Hub dataset repo.

    Check GPU availability and local integrity, enforce a private destination,
    upload payloads/manifest, and check their presence at the resulting commit.

    Args:
        root (Path): Completed checkpoint directory with a manifest.
        repo_id (str): Explicit owner/repository destination; authentication
            comes from the saved Hub login or HF_TOKEN.
    Returns:
        dict: Repository ID/type and commit revision for the upload receipt.
    Raises:
        ValueError: Local validation or destination privacy checks fail.
        RuntimeError: Uploaded files are missing from the commit listing.
        Hub API errors: Authentication or transfer fails.
    Side effects:
        May create a private repository and upload large training-state files.
        Remote file listings do not replace post-download integrity checks.
    """
    cloud_guard()
    verify_manifest(root)
    from huggingface_hub import HfApi
    api = HfApi()
    import_module("infrastructure.7a_verl.hf_artifacts").private_repo(api, repo_id, "dataset")
    api.upload_large_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(root),
                            ignore_patterns=[".cache/**", "**/.cache/**"])
    revision = api.repo_info(repo_id=repo_id, repo_type="dataset").sha
    expected = set(verify_manifest(root)["files"]) | {MANIFEST}
    if not expected.issubset(api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)):
        raise RuntimeError("Uploaded checkpoint inventory incomplete")
    return dict(repo_id=repo_id, repo_type="dataset", revision=revision)


def restore(repo_id, revision, destination):
    """Download an immutable Hub checkpoint into a fresh directory and verify it.

    Args:
        repo_id (str): Private dataset repository containing checkpoint payloads.
        revision (str): Full 40-character hexadecimal Hub commit hash.
        destination (Path | str): New directory named global_step_N.
    Returns:
        dict: Manifest verified against the downloaded file contents.
    Raises:
        ValueError: Revision, destination naming or downloaded integrity is invalid.
        FileExistsError: The destination already exists.
        Hub API errors: Authentication or download fails.
    Side effects:
        Downloads full state on the cloud host. Failure may leave a partial
        directory; it is not automatically deleted or reused.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Restore requires a full immutable Hub commit hash")
    destination = Path(destination)
    if not re.fullmatch(r"global_step_([1-9][0-9]*)", destination.name):
        raise ValueError("Restore directory must be named global_step_N")
    if destination.exists():
        raise FileExistsError("Restore requires a fresh directory")
    cloud_guard()
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision,
                      local_dir=str(destination), allow_patterns=[MANIFEST, "actor/**", "data.pt"])
    return verify_manifest(destination)


def main():
    """Parse checkpoint-helper CLI arguments and execute one selected operation.

    Inputs:
        sys.argv supplies create/verify/upload/restore, checkpoint path and
        operation-specific identity/repository/revision options.
    Returns:
        None; prints manifest metadata or an upload receipt as JSON to stdout.
    Side effects:
        Dispatches file creation or network transfer as selected; transfers
        require --cloud. Invalid arguments exit through argparse.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["create", "verify", "upload", "restore"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--repo")
    parser.add_argument("--revision")
    parser.add_argument("--cloud", action="store_true")
    args = parser.parse_args()
    if args.action in ("upload", "restore") and (not args.cloud or not args.repo):
        parser.error("Hub transfers require --cloud and --repo")
    if args.action == "create":
        if not args.identity:
            parser.error("create requires --identity")
        result = create_manifest(args.checkpoint, json.loads(args.identity.read_text()))
    elif args.action == "verify":
        result = verify_manifest(args.checkpoint)
    elif args.action == "upload":
        result = upload(args.checkpoint, args.repo)
    else:
        if not args.revision:
            parser.error("restore requires --revision")
        result = restore(args.repo, args.revision, args.checkpoint)
    if args.action in ("create", "verify", "restore"):
        result = {key: result[key] for key in ("schema", "step", "world_size")}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
