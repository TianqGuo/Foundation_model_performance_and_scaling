"""Coordinate bounded KubeRay phases with portable data and pinned Hub recovery."""
import argparse
from datetime import datetime, timezone
from importlib import import_module
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
control = import_module("infrastructure.7c_kuberay.job_control")
staging = import_module("infrastructure.7c_kuberay.data_staging")
previous = import_module("infrastructure.7b_ray.experiment")


def unpack_evidence(archive_path, destination):
    """Extract only bounded regular-file evidence into a fresh local directory.

    Inputs: controller archive and output. Returns directory; reject traversal,
    links, tensors, excessive file counts or total expanded size above 256 MiB.
    CPU-only: never deserializes tensors or imports the GPU stack.
    """
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        if len(members) > 100000 or sum(m.size for m in members) > 256 * 1024**2:
            raise ValueError("Evidence archive exceeds bounded limits")
        for member in members:
            path = PurePosixPath(member.name)
            if (not member.isfile() or path.is_absolute() or ".." in path.parts or "\\" in member.name
                    or path.suffix in {".pt", ".bin", ".safetensors"}):
                raise ValueError("Evidence archive contains unsafe or tensor entries")
        destination.mkdir(parents=True, exist_ok=False)
        archive.extractall(destination, filter="data")
    return destination


def review(root):
    """Reuse 7B state audits and require three distinct worker pods/Ray runtimes.

    Input: extracted baseline/interruption/resume evidence directories. Returns
    bounded recovery report; reject missing cleanup, repeated runtimes/pods or
    changed physical placement. No replacement-node/bitwise claims are introduced.
    """
    runs = [root / phase / "evidence" for phase in ("baseline", "interrupt", "resume")]
    report = previous.review_recovery(*runs, saved_step=2)
    identities = [json.loads((run / "identities.json").read_text()) for run in runs]
    workers = [identity["platform"]["worker"] for identity in identities]
    if len({w["pod_uid"] for w in workers}) != 3 or len({w["node_id"] for w in workers}) != 3:
        raise ValueError("Require fresh worker pods and Ray runtimes for all training phases")
    if len({w["physical_node"] for w in workers}) != 1 or not workers[0]["physical_node"]:
        raise ValueError("Initial recovery experiment requires the same Kubernetes GPU node")
    for phase in ("preflight", "baseline", "interrupt", "resume"):
        if not json.loads((root / phase / "cleanup.json").read_text()).get("verified"):
            raise ValueError("Missing verified RayJob cleanup")
    receipt = json.loads((runs[1] / "hf_checkpoint.json").read_text())
    restore = json.loads((runs[2] / "hf_restore.json").read_text())
    if (not receipt.get("full_checkpoint") or restore.get("repo_id") != receipt["repo_id"]
            or restore.get("revision") != receipt["revision"] or restore.get("saved_step") != 2
            or not restore.get("file_hashes_verified")):
        raise ValueError("Pinned checkpoint transfer receipts differ")
    import yaml
    resume_config = yaml.safe_load((runs[2] / "resolved_config.yaml").read_text())
    resume_path = resume_config["trainer"].get("resume_from_path", "")
    if (resume_config["trainer"].get("resume_mode") != "resume_path"
            or not resume_path.startswith("/workspace/restored/") or resume_path != restore["destination"]):
        raise ValueError("Resume did not use a fresh downloaded checkpoint path")
    if len({identity["platform"]["image"] for identity in identities}) != 1:
        raise ValueError("Training phases used different image digests")
    optimizer_counts = {}
    for phase, run, expected in zip(("baseline", "interrupt", "resume"), runs, (6, 4, 2)):
        records = previous.events(run, "diagnostics")
        first = [record for record in records if record["event"] == "first_optimizer_update"]
        if len(first) != 2 or {record["rank"] for record in first} != {0, 1} or any(
                record["world_size"] != 2 or record["sharded_tensors"] <= 0 for record in first):
            raise ValueError("Missing actual sharded training ranks: " + phase)
        counts = {str(rank): sum(record["event"] == "optimizer_update" and record.get("rank") == rank
                                for record in records) for rank in (0, 1)}
        if set(counts.values()) != {expected}:
            raise ValueError("Unexpected bounded optimizer update count: " + phase)
        optimizer_counts[phase] = counts
    report.update(optimizer_calls_per_rank=optimizer_counts, scope="same Kubernetes GPU node; distinct RayJob pods/runtimes; pinned Hub download",
                  worker_pod_uids=[w["pod_uid"] for w in workers], checkpoint=receipt)
    return report


def execute(args):
    """Stage bundle, then run four sequential RayJobs and review retained evidence.

    Input: validated CLI args. Returns None; creates jobs only on existing cluster,
    uses cleanup-on-success between phases, retains lightweight local archives and
    full interrupted state on the Hub. Stops on failure; no automatic crash restart.
    """
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    staging.stage(args.data, args.image, args.context, args.namespace, args.name + "-stage",
                  args.pvc, args.node_name, output / "staging", args.pull_secret)
    receipt = None
    timings = {}
    for phase in ("preflight", "baseline", "interrupt", "resume"):
        name = args.name + "-" + phase
        extra = dict(upload_checkpoint_repo="auto") if phase == "interrupt" else {}
        if phase == "resume":
            extra = dict(checkpoint="/workspace/restored/" + args.name + "/global_step_2",
                         restore_repo=receipt["repo_id"], restore_revision=receipt["revision"])
        manifest = control.renderer.render_job(args.image, args.namespace, name, args.pvc,
            args.secret, args.node_name, args.pull_secret, phase=phase, **extra)
        started = time.monotonic()
        phase_output = output / phase
        control.run(manifest, args.context, phase_output, cleanup_on_success=True)
        evidence = unpack_evidence(phase_output / "workspace_evidence.tar.gz", phase_output / "evidence")
        timings[phase + "_seconds"] = time.monotonic() - started
        (output / "timings.json").write_text(json.dumps(timings, indent=2))
        if phase == "interrupt":
            receipt = json.loads((evidence / "hf_checkpoint.json").read_text())
            import_module("infrastructure.7c_kuberay.checkpoint_transfer").validate_options(
                restore_repo=receipt["repo_id"], revision=receipt["revision"])
    report = review(output)
    (output / "recovery_review.json").write_text(json.dumps(report, indent=2))
    print("7C experiment completed; review retained evidence before releasing cloud nodes.")


def main():
    """Preview locally or coordinate explicitly selected cloud Kubernetes execution.

    Input: --check/--cloud, immutable image/node/resources, prepared local bundle.
    Returns None. CPU check verifies data and manifests without kubectl/model use.
    Cloud requires explicit context; neither mode builds images/provisions nodes.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--cloud", action="store_true")
    parser.add_argument("--image", required=True)
    parser.add_argument("--node-name", required=True)
    parser.add_argument("--context")
    parser.add_argument("--namespace", default="part7")
    parser.add_argument("--pvc", default="part7-workspace")
    parser.add_argument("--secret", default="part7-huggingface")
    parser.add_argument("--pull-secret")
    parser.add_argument("--name", default=datetime.now(timezone.utc).strftime("part7-7c-%Y%m%d-%H%M%S"))
    parser.add_argument("--data", type=Path, default=ROOT / "results/7a/step2_math_final")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.output = args.output or ROOT / "results/7c" / args.name
    for phase in ("stage", "preflight", "baseline", "interrupt", "resume"):
        control.renderer.dns_label(args.name + "-" + phase)
    staging.reference.verify_bundle(args.data)
    control.validate_manifest(control.renderer.render_job(args.image, args.namespace,
        args.name + "-preflight", args.pvc, args.secret, args.node_name, args.pull_secret))
    if args.check:
        print("Validated portable bundle and initial RayJob; no Kubernetes contacts.")
        return
    if not args.context:
        parser.error("Cloud execution requires --context")
    execute(args)


if __name__ == "__main__":
    main()
