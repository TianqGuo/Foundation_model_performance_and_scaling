"""Cloud-only export/upload. Use --help; authentication uses HF_TOKEN or hf login."""
from __future__ import annotations

import argparse
import fnmatch
import getpass
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def checkpoint_actor(run):
    candidates = list((run / "checkpoints").glob("global_step_*/actor"))
    if not candidates:
        raise FileNotFoundError("No actor checkpoint to export")
    return max(candidates, key=lambda path: int(path.parent.name.removeprefix("global_step_")))


def artifact_ignores(include_checkpoint):
    patterns = [".cache/**", "**/.cache/**", "hf_model/**", "hf_upload.json"]
    if not include_checkpoint:
        patterns.append("checkpoints/**")
    return patterns


def private_repo(api, repo_id, repo_type):
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=True, exist_ok=True)
    if not api.repo_info(repo_id=repo_id, repo_type=repo_type).private:
        raise ValueError("Use a private Hugging Face repository for run artifacts/models")


def artifact_files(run, include_checkpoint):
    patterns = artifact_ignores(include_checkpoint)
    return [str(path.relative_to(run)) for path in run.rglob("*") if path.is_file()
            and not any(fnmatch.fnmatch(str(path.relative_to(run)), pattern) for pattern in patterns)]


def upload_destinations(api, run, model_repo, artifact_repo):
    account = api.whoami()["name"]
    name = "part7-7a-" + run.name.replace("_", "-")
    return dict(account=account,
                model_repo=f"{account}/{name}" if model_repo == "auto" else model_repo,
                artifact_repo=f"{account}/{name}-artifacts" if artifact_repo == "auto" else artifact_repo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud", action="store_true", required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model-repo", help="private model repo; use a distinct repo per exported run")
    parser.add_argument("--artifact-repo", help="private dataset repo dedicated to this run")
    parser.add_argument("--include-checkpoint", action="store_true", help="also upload full training state")
    parser.add_argument("--prepare-upload", action="store_true", help="authenticate and resolve destinations before training; repo value auto uses token owner")
    parser.add_argument("--destinations-file", type=Path)
    args = parser.parse_args()
    if not args.model_repo and not args.artifact_repo:
        parser.error("Select at least one upload destination")
    if args.include_checkpoint and not args.artifact_repo:
        parser.error("--include-checkpoint requires --artifact-repo")
    run = args.run.resolve()
    if args.prepare_upload:
        if not args.destinations_file:
            parser.error("--prepare-upload requires --destinations-file")
        from huggingface_hub import HfApi, get_token, login
        if not get_token():
            if not sys.stdin.isatty():
                raise RuntimeError("No Hugging Face login: supply HF_TOKEN for noninteractive execution")
            login(token=getpass.getpass("Hugging Face write token: "), add_to_git_credential=False)
        api = HfApi()
        destinations = upload_destinations(api, run, args.model_repo, args.artifact_repo)
        for repo, kind in ((destinations["model_repo"], "model"), (destinations["artifact_repo"], "dataset")):
            if repo:
                private_repo(api, repo, kind)
        args.destinations_file.write_text(json.dumps(destinations, indent=2))
        print(f"Hugging Face account verified: {destinations['account']}")
        return
    status = json.loads((run / "job_status.json").read_text())
    if not status.get("success"):
        raise ValueError("Upload helper expects a successfully completed run")
    from huggingface_hub import HfApi
    api = HfApi()
    # Resolve authentication/privacy before spending time merging model shards.
    for repo, kind in ((args.model_repo, "model"), (args.artifact_repo, "dataset")):
        if repo:
            private_repo(api, repo, kind)
    receipt = {}
    if args.model_repo:
        identity = json.loads((run / "identities.json").read_text())
        source_commit = subprocess.check_output(
            ["git", "-C", str(ROOT / ".venv-7a/src/verl"), "rev-parse", "HEAD"], text=True).strip()
        if source_commit != identity["verl_commit"]:
            raise ValueError("Export must use the same verl commit as the training run")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Model export requires the cloud GPU environment")
        actor = checkpoint_actor(run)
        target = run / "hf_model"
        subprocess.run([sys.executable, "-m", "verl.model_merger", "merge",
                        "--backend", "fsdp", "--local_dir", str(actor),
                        "--target_dir", str(target)], check=True)
        commit = api.upload_folder(repo_id=args.model_repo, repo_type="model",
                                   folder_path=str(target), ignore_patterns=[".cache/**"])
        files = api.list_repo_files(repo_id=args.model_repo, revision=commit.oid)
        if "config.json" not in files or not any(name.endswith(".safetensors") for name in files):
            raise RuntimeError("Uploaded model is missing config or weights")
        receipt["model"] = dict(repo_id=args.model_repo, revision=commit.oid,
                                checkpoint=str(actor.relative_to(run)), dtype="bfloat16")
        (run / "hf_upload.json").write_text(json.dumps(receipt, indent=2))
    if args.artifact_repo:
        expected = artifact_files(run, args.include_checkpoint)
        if args.include_checkpoint:
            checkpoint_actor(run)
        api.upload_large_folder(repo_id=args.artifact_repo, repo_type="dataset",
                                folder_path=str(run),
                                ignore_patterns=artifact_ignores(args.include_checkpoint))
        revision = api.repo_info(repo_id=args.artifact_repo, repo_type="dataset").sha
        files = api.list_repo_files(repo_id=args.artifact_repo, repo_type="dataset", revision=revision)
        if "resolved_config.yaml" not in expected or not set(expected).issubset(files):
            raise RuntimeError("Uploaded run evidence is incomplete")
        receipt["artifacts"] = dict(repo_id=args.artifact_repo, repo_type="dataset",
                                    revision=revision, full_checkpoint=args.include_checkpoint)
    (run / "hf_upload.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    print("Hub file listings verified; model reload and training recovery remain separate checks.")


if __name__ == "__main__":
    main()
