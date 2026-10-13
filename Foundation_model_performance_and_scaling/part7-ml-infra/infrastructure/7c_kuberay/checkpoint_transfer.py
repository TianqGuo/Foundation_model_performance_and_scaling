"""Private checkpoint transfers inside KubeRay using 7B byte-integrity validation."""
from importlib import import_module
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
bundle = import_module("infrastructure.7b_ray.checkpoint_bundle")


def validate_options(upload_repo=None, restore_repo=None, revision=None):
    """Validate repository names and immutable restore revision without contacting Hub.

    Inputs: optional upload/restore repository IDs and revision. Returns None;
    reject malformed IDs, incomplete restore pairs or mutable revisions. `auto`
    is allowed only for upload, with account/run-based naming resolved later.
    """
    pattern = r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*"
    for repo, allow_auto in ((upload_repo, True), (restore_repo, False)):
        if repo and not (allow_auto and repo == "auto") and not re.fullmatch(pattern, repo):
            raise ValueError("Use owner/repository for checkpoint retention")
    if bool(restore_repo) != bool(revision):
        raise ValueError("Restore requires both repository and immutable revision")
    if revision and not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Restore requires a full 40-hex Hub commit")


def credentials():
    """Read the mounted Secret inside a Kubernetes pod; return an authenticated API.

    No environment export, interactive login or credential logging occurs. Reject
    standalone execution and missing/empty Secret. This is a transfer guard, not
    an attestation of cloud hosting; callers explicitly select cloud execution.
    """
    if not os.environ.get("POD_UID") or not os.environ.get("NODE_NAME"):
        raise RuntimeError("Checkpoint transfer requires the prepared Kubernetes pod")
    from huggingface_hub import HfApi
    token = Path("/run/secrets/huggingface/token").read_text().strip()
    if not token:
        raise ValueError("Mounted Hugging Face Secret is empty")
    return HfApi(token=token)


def upload(root, repo_id, run_name):
    """Upload complete verified state directly from PVC to a private Hub dataset.

    Inputs: completed checkpoint, repository or `auto`, unique run name. Returns
    repository/commit/step receipt after file-listing verification. Creates a private
    repository if needed; tensor deserialization and GPU computation are absent.
    Listing checks are transfer evidence, not verified training-state restoration.
    """
    validate_options(upload_repo=repo_id)
    if not Path(root).resolve().is_relative_to(Path("/workspace")):
        raise ValueError("Checkpoint upload source must be on the persistent workspace")
    manifest = bundle.verify_manifest(root)
    api = credentials()
    if repo_id == "auto":
        import_module("infrastructure.7c_kuberay.render").dns_label(run_name)
        repo_id = api.whoami()["name"] + "/part7-7c-" + run_name
    import_module("infrastructure.7a_verl.hf_artifacts").private_repo(api, repo_id, "dataset")
    api.upload_large_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(root),
        ignore_patterns=[".cache/**", "**/.cache/**"])
    revision = api.repo_info(repo_id=repo_id, repo_type="dataset").sha
    validate_options(restore_repo=repo_id, revision=revision)
    expected = set(manifest["files"]) | {bundle.MANIFEST}
    if not expected.issubset(api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)):
        raise RuntimeError("Uploaded checkpoint inventory incomplete")
    return dict(repo_id=repo_id, repo_type="dataset", revision=revision, saved_step=manifest["step"],
                full_checkpoint=True, verification="Hub listing; post-download integrity and state audits still required")


def restore(repo_id, revision, destination):
    """Download pinned full state into a fresh PVC directory and verify file hashes.

    Inputs: private repository, immutable commit, new global_step_N directory.
    Returns verified manifest. Uses the mounted Secret; rejects reused paths or
    public repositories. Partial downloads remain on failure for diagnosis.
    """
    validate_options(restore_repo=repo_id, revision=revision)
    destination = Path(destination)
    if not destination.resolve().is_relative_to(Path("/workspace")):
        raise ValueError("Restore destination must be on the persistent workspace")
    if not re.fullmatch(r"global_step_[1-9][0-9]*", destination.name):
        raise ValueError("Restore destination must be named global_step_N")
    if destination.exists():
        raise FileExistsError("Restore requires a fresh directory")
    api = credentials()
    if not api.repo_info(repo_id=repo_id, repo_type="dataset", revision=revision).private:
        raise ValueError("Use a private checkpoint repository")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision, token=api.token,
        local_dir=str(destination), allow_patterns=[bundle.MANIFEST, "actor/**", "data.pt"])
    return bundle.verify_manifest(destination)
