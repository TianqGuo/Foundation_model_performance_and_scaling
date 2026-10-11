"""Bounded cloud recovery experiment; invoked by part7B.sh, never provisions resources."""
import argparse
from datetime import datetime, timezone
import getpass
from importlib import import_module
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
bundle = import_module("infrastructure.7b_ray.checkpoint_bundle")


def events(run, folder="recovery"):
    """Read process JSONL audit records from one run; return a list of dictionaries.

    Inputs: run directory and optional evidence subfolder. Missing events return an empty list so evidence checks
    fail explicitly; malformed records propagate rather than being discarded.
    """
    return [json.loads(line) for path in sorted((run / folder).glob("*.jsonl"))
            for line in path.read_text().splitlines() if line.strip()]


def review_recovery(baseline, interrupted, resumed, saved_step):
    """Check retained state/placement evidence and next-batch continuation.

    Inputs: three run directories and saved step. Returns a concise report;
    raises ValueError for missing/mismatched audits, unsuccessful phases, missing
    placement or a different next source batch. Does not load checkpoint tensors.
    Successful review proves only states audited by hooks, not vLLM RNG recovery.
    """
    for run in (baseline, interrupted, resumed):
        status = json.loads((run / "job_status.json").read_text())
        if not status.get("success"):
            raise ValueError(f"Unsuccessful phase: {run.name}")
    if not json.loads((interrupted / "job_status.json").read_text()).get("planned_interruption"):
        raise ValueError("Missing controlled interruption evidence")
    records = events(resumed)
    ranks = [record for record in records if record["event"] == "rank_restore"]
    if len(ranks) != 2 or {record["rank"] for record in ranks} != {0, 1}:
        raise ValueError("Require exactly two rank restoration audits")
    for record in ranks:
        if record["world_size"] != 2 or set(record["components"]) != {"model", "optimizer", "scheduler", "rng"}:
            raise ValueError("Incomplete rank state audit")
        if any(value["expected"] != value["actual"] for value in record["components"].values()):
            raise ValueError("Rank restoration mismatch")
    trainers = [record for record in records if record["event"] == "trainer_restore"]
    if len(trainers) != 1 or trainers[0]["step"] != saved_step:
        raise ValueError("Missing saved-step restoration")
    if set(trainers[0]["components"]) != {"data", "step"} or any(
        value["expected"] != value["actual"] for value in trainers[0]["components"].values()
    ):
        raise ValueError("Data/global-step restoration mismatch")
    placements = [record for record in records if record["event"] == "ray_placement"]
    if not placements or not placements[0]["placement_groups"] or placements[0]["resources"].get("GPU", 0) != 2:
        raise ValueError("Missing two-GPU Ray placement evidence")
    diagnostics = events(resumed, "diagnostics")
    updates = [r for r in diagnostics if r["event"] == "first_optimizer_update"]
    if (len(updates) != 2 or {r["rank"] for r in updates} != {0, 1}
            or any(r["world_size"] != 2 or r["sharded_tensors"] <= 0 for r in updates)
            or len({(r["host"], r["visible_devices"], r["cuda_device"]) for r in updates}) != 2):
        raise ValueError("Missing actual two-rank sharded optimizer updates on distinct devices")
    completions = [r for r in records if r["event"] == "rank_load_complete"]
    if len(completions) != 2 or {r["rank"] for r in completions} != {0, 1}:
        raise ValueError("Missing per-rank load timing")
    update_delays = {str(r["rank"]): r["time"] - next(c["time"] for c in completions if c["rank"] == r["rank"])
                     for r in updates}
    if any(delay < 0 for delay in update_delays.values()):
        raise ValueError("Optimizer update occurred before restoration audit completed")
    next_step = saved_step + 1
    reference = [r["source_indices"] for r in events(baseline)
                 if r["event"] == "training_batch" and r["step"] == next_step]
    recovered = [r["source_indices"] for r in records
                 if r["event"] == "training_batch" and r["step"] == next_step]
    if len(reference) != 1 or reference != recovered:
        raise ValueError("Next training source batch differs from baseline")
    if not any(r["event"] == "checkpoint_complete" and r["step"] >= next_step for r in records):
        raise ValueError("No completed checkpoint after resumed training")
    return dict(state_restore_verified=True, world_size=2, saved_step=saved_step,
                first_resumed_step=next_step, next_source_indices=recovered[0],
                post_audit_to_first_optimizer_update_seconds=update_delays,
                scope="same-host fresh Ray runtime; pinned Hub download",
                limitations=["vLLM server RNG/cache/in-flight requests are not restored",
                             "driver-process RNG is not part of verl's worker checkpoint",
                             "no replacement-node recovery or bitwise rollout equivalence claim"])


def run_phase(name, args, extra=()):
    """Launch one native-verl phase in a fresh child process with a 45-minute cap.

    Inputs: phase name, experiment args and additional CLI arguments. Returns
    elapsed seconds; nonzero exit/timeout propagates. Logs stream to the shell's
    console log; each phase retains its own native Ray logs/status/audits.
    """
    command = [sys.executable, "-u", str(ROOT / "infrastructure/7b_ray/run_7b.py"),
               "--cloud", "--phase", name, "--data", str(args.data),
               "--output", str(args.output / name), "--model-id", args.model_id,
               "--total-steps", str(args.total_steps), "--interrupt-step", str(args.interrupt_step)]
    if args.model_path:
        command.extend(["--model-path", str(args.model_path)])
    if args.model_revision:
        command.extend(["--model-revision", args.model_revision])
    command.extend(extra)
    started = time.monotonic()
    subprocess.run(command, check=True, timeout=45 * 60)
    return time.monotonic() - started


def evidence_files(root):
    """List lightweight experiment evidence, excluding checkpoint/download/cache trees.

    Input: experiment directory. Returns relative POSIX paths; does not read
    tensors. All run logs/configs/audits are retained; no model/optimizer payloads
    from checkpoints or restored copies are eligible for the evidence upload.
    """
    return [path.relative_to(root).as_posix() for path in sorted(root.rglob("*"))
            if path.is_file() and not {"checkpoints", "restored", ".cache"}.intersection(path.relative_to(root).parts)]


def main():
    """Authenticate once and run baseline/interruption/Hub restore/resume/review.

    Input: CLI paths/model settings; --cloud is mandatory. --authenticate-only
    prepares saved login and returns without creating a repo or launching phases. Returns None and writes
    timings, immutable upload receipt and review report. Uses existing HF_TOKEN
    or login; prompts securely only if neither exists. Requires a fresh output,
    sufficient free disk and an already provisioned two-GPU cloud host. Partial
    failures preserve all evidence/state and never automatically delete payloads.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud", action="store_true", required=True)
    parser.add_argument("--authenticate-only", action="store_true",
                        help="prepare saved Hub login in the terminal before timeout")
    parser.add_argument("--data", type=Path, default=ROOT / "results/7a/step2_math_final")
    parser.add_argument("--output", type=Path, default=ROOT / "results/7b" /
                        datetime.now(timezone.utc).strftime("recovery_%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--repo", help="optional private dataset repo; defaults to authenticated owner/run name")
    parser.add_argument("--model-id", default=bundle.reference.DEFAULT_MODEL_ID)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-revision")
    parser.add_argument("--total-steps", type=int, default=3)
    parser.add_argument("--interrupt-step", type=int, default=2)
    args = parser.parse_args()
    if not 0 < args.interrupt_step < args.total_steps:
        parser.error("total steps must exceed a positive interruption step")
    args.output = args.output.resolve()
    args.data = args.data.resolve()
    if args.output.exists():
        raise FileExistsError("Use a fresh experiment output directory")
    bundle.cloud_guard()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output.parent).free < 100 * 1024**3:
        raise RuntimeError("Allow at least 100 GiB free disk for full-state recovery copies")
    from huggingface_hub import HfApi, get_token, login
    if not get_token():
        if not sys.stdin.isatty():
            raise RuntimeError("Supply HF_TOKEN or saved Hub login for noninteractive execution")
        login(token=getpass.getpass("Hugging Face write token: "), add_to_git_credential=False)
    api = HfApi()
    account = api.whoami()["name"]
    if args.authenticate_only:
        print(f"Hugging Face account verified: {account}", flush=True)
        return
    repo = args.repo or f"{account}/part7-7b-{args.output.name.replace('_', '-')}"
    import_module("infrastructure.7a_verl.hf_artifacts").private_repo(api, repo, "dataset")
    args.output.mkdir()
    (args.output / "destination.json").write_text(json.dumps(dict(repo_id=repo, repo_type="dataset")))
    timings = {}
    environment = ROOT / "results/7a/environment"
    if environment.is_dir():
        shutil.copytree(environment, args.output / "environment")
    try:
        timings["baseline_seconds"] = run_phase("baseline", args)
        timings["interrupted_phase_seconds"] = run_phase("interrupt", args)
        checkpoint = args.output / "interrupt/checkpoints" / f"global_step_{args.interrupt_step}"
        identity = json.loads((args.output / "interrupt/identities.json").read_text())
        started = time.monotonic()
        bundle.create_manifest(checkpoint, identity)
        timings["manifest_seconds"] = time.monotonic() - started
        started = time.monotonic()
        receipt = bundle.upload(checkpoint, repo)
        timings["upload_seconds"] = time.monotonic() - started
        (args.output / "hf_checkpoint.json").write_text(json.dumps(receipt, indent=2))
        restored = args.output / "restored" / checkpoint.name
        started = time.monotonic()
        bundle.restore(repo, receipt["revision"], restored)
        timings["download_and_integrity_seconds"] = time.monotonic() - started
        timings["resumed_phase_seconds"] = run_phase("resume", args, ["--checkpoint", str(restored)])
        report = review_recovery(args.output / "baseline", args.output / "interrupt",
                                 args.output / "resume", args.interrupt_step)
        (args.output / "recovery_review.json").write_text(json.dumps(report, indent=2))
        (args.output / "timings.json").write_text(json.dumps(timings, indent=2))
        # Record the console before retention; its final upload messages stay local.
        import os
        console = os.environ.get("PART7_EXECUTION_LOGS")
        if console and (Path(console) / "console.log").is_file():
            (args.output / "execution").mkdir()
            shutil.copyfile(Path(console) / "console.log", args.output / "execution/console.log")
        expected = evidence_files(args.output)
        started = time.monotonic()
        evidence_commit = api.upload_folder(repo_id=repo, repo_type="dataset", path_in_repo="evidence",
                                           folder_path=str(args.output),
                                           ignore_patterns=["**/checkpoints/**", "checkpoints/**", "restored/**",
                                                            "**/.cache/**", ".cache/**"])
        remote = set(api.list_repo_files(repo_id=repo, repo_type="dataset", revision=evidence_commit.oid))
        if not {"evidence/" + name for name in expected}.issubset(remote):
            raise RuntimeError("Uploaded experiment evidence is incomplete")
        timings["evidence_upload_seconds"] = time.monotonic() - started
        receipt["evidence_revision"] = evidence_commit.oid
        (args.output / "hf_checkpoint.json").write_text(json.dumps(receipt, indent=2))
        print(json.dumps(report, indent=2))
        print(f"Retain lightweight evidence from {args.output}; checkpoint is at {repo}@{receipt['revision']}")
    finally:
        (args.output / "timings.json").write_text(json.dumps(timings, indent=2))


if __name__ == "__main__":
    main()
