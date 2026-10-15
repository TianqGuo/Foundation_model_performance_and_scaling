"""One cloud entry point: setup/readiness, prepare data, train and audit recovery."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
setup = import_module("infrastructure.7c_kuberay.setup_cluster")
cluster = import_module("infrastructure.7c_kuberay.cluster_check")
experiment = import_module("infrastructure.7c_kuberay.experiment")
IMAGE = "sclion/part7-training@sha256:b1d8240b348d2a036d311f19fe75784825613607129c1d93526cfa7f530fb8e8"
INPUT_HASHES = {
    "train": ("data/math/train.jsonl", "22ac3f59ec3b41afd056eb2a92b5447239d217624e798ee963361ea90401e91b"),
    "validation": ("data/math/validation.jsonl", "876aad1d989ec7c856dac86efd0460752a30de8898e67d8ff2e2c189739eb414"),
    "grader": ("cs336_alignment/drgrpo_grader.py", "f88b29b0e76b3c28f969cb8b2f945e01b70aed2753306445d30822a192c54ec7"),
    "prompt": ("cs336_alignment/prompts/r1_zero.prompt", "56213732da87e12ace60a2b88a16df0af18280fb0563a957cbecc938a6a2ac0e"),
    "selection_helper": ("cs336_alignment/section7_grpo/run_utils.py", "05b4247ee7fd61a5a197d7f220cbd66308eb24eafbc69afe8ebbf762776a4a0a"),
}


def verify_references(part5):
    """Hash-check the Git-cloned Part 5 data/grader/prompt against accepted 7A inputs.

    Input: read-only Part 5 root. Returns None or raises on missing/changed inputs.
    Never downloads, edits Part 5, imports its training stack or loads a model.
    """
    for name, (relative, expected) in INPUT_HASHES.items():
        if hashlib.sha256((part5 / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Part 5 {name} differs from reviewed 7A input")


def prepare_bundle(image, part5, destination):
    """Prepare/reuse the portable bundle on cloud using the published environment.

    Inputs: image digest, verified read-only Part 5 root, destination directory.
    Reuses only a hash-verified 1023/32 bundle with accepted source identities.
    Otherwise Docker executes the existing image adapter without network/GPUs,
    mounting only Part 5 read-only and the output parent writable. No laptop copy.
    """
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        docker = ["docker"]
        if subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
            docker = ["sudo", "-n", "docker"]
        subprocess.run([*docker, "run", "--rm", "--network", "none", "--env", "NVIDIA_VISIBLE_DEVICES=void",
            "--mount", f"type=bind,source={part5.resolve()},target=/part5,readonly",
            "--mount", f"type=bind,source={destination.parent.resolve()},target=/prepared",
            "--entrypoint", "/opt/part7/.venv-7a/bin/python", image,
            "/opt/part7/workloads/math_workload.py", "--part5-root", "/part5",
            "--output", "/prepared/" + destination.name, "--drop-empty-solutions"], check=True)
    manifest = experiment.staging.reference.verify_bundle(destination)
    if manifest["train_count"] != 1023 or manifest["validation_count"] != 32:
        raise ValueError("Expected the accepted 1023/32 bundle")
    if manifest.get("evaluation", {}).get("seed") != 12345 or manifest.get("training_prefix_count") != 1024:
        raise ValueError("Prepared bundle must use the accepted selection settings")
    for name, (_, expected) in INPUT_HASHES.items():
        if manifest["inputs"][name]["sha256"] != expected:
            raise ValueError(f"Prepared bundle {name} identity differs")
    return manifest


def cloud_cluster_present():
    """Require a full systemd cloud VM and report whether K3s already exists.

    No inputs; returns bool. Performs a process check only, with no mutation.
    Actual inventory/GPU/kernel checks remain in setup and the experiment.
    """
    if Path("/.dockerenv").exists() or subprocess.check_output(["ps", "-p", "1", "-o", "comm="], text=True).strip() != "systemd":
        raise ValueError("Run --cloud on the full GPU VM")
    return bool(shutil.which("k3s") or Path("/etc/rancher/k3s").exists())


def execute(args):
    """Run all cloud stages and retain an overall receipt, stopping on first failure.

    Input: parsed CLI args. Fresh hosts install via native setup; existing clusters
    are checked and never reset. --fresh-setup requires actual installation, so a
    reuse run cannot be mistaken for clean-install evidence. Native experiment
    performs data staging, GPU preflight, training/interruption/Hub recovery audits.
    Successful phases clean their Ray resources; failures retain them for diagnosis.
    """
    installed = cloud_cluster_present()
    if args.fresh_setup and installed:
        raise ValueError("--fresh-setup requires a clean VM/cluster state; no automatic uninstall")
    verify_references(args.part5_root)
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = dict(success=False, image=args.image, setup_mode="reuse" if installed else "fresh",
                   fresh_setup_verified=False, stages=[])
    try:
        if not installed:
            print("7C E2E: installing Kubernetes from the normal fresh path", flush=True)
            setup.install(args.output / "setup", with_hf_secret=True)
            receipt["fresh_setup_verified"] = True
        os.environ["KUBECONFIG"] = str(Path.home() / ".kube/part7-config")
        print("7C E2E: checking Kubernetes, storage and operator", flush=True)
        report = cluster.inspect("default", "part7", setup.NODE, "part7-workspace", "default", "kuberay-operator")
        (args.output / "cluster_check.json").write_text(json.dumps(report, indent=2) + "\n")
        # Query only the Secret's name. Token value remains inside Kubernetes.
        cluster.control.kubectl("default", "part7", ["get", "secret", "part7-huggingface", "-o", "name"])
        receipt["stages"].append("cluster_ready")
        print("7C E2E: preparing accepted data bundle on the VM", flush=True)
        prepare_bundle(args.image, args.part5_root, args.data)
        receipt["stages"].append("data_ready")
        print("7C E2E: staging, GPU preflight, baseline, interruption and Hub recovery", flush=True)
        run = argparse.Namespace(data=args.data, image=args.image, context="default", namespace="part7",
            name=args.name, pvc="part7-workspace", node_name=setup.NODE, secret="part7-huggingface",
            pull_secret=None, output=args.output / "experiment")
        experiment.execute(run)
        receipt["stages"].append("recovery_review_passed")
        receipt["success"] = True
    finally:
        (args.output / "e2e_review.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("7C E2E completed; retain and review " + str(args.output), flush=True)


def main():
    """Parse cloud/preview mode and optional fresh-install requirement.

    --check reads/hashes references and validates identity only; no subprocess,
    Kubernetes contact, model import or local GPU work. Existing image is reused.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--cloud", action="store_true")
    parser.add_argument("--fresh-setup", action="store_true")
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--part5-root", type=Path, default=ROOT.parent / "part5-alignment")
    parser.add_argument("--data", type=Path, default=ROOT / "results/7a/step2_math_final")
    parser.add_argument("--name", default=datetime.now(timezone.utc).strftime("part7-e2e-%Y%m%d-%H%M%S"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.output = args.output or ROOT / "results/7c" / args.name
    experiment.control.renderer.pinned_image(args.image)
    for phase in ("stage", "preflight", "baseline", "interrupt", "resume"):
        experiment.control.renderer.dns_label(args.name + "-" + phase)
    verify_references(args.part5_root)
    if args.check:
        print("Validated source hashes and image identity; no cloud contacts. Workflow: setup/readiness -> data -> GPU preflight -> training/recovery.")
    else:
        execute(args)


if __name__ == "__main__":
    main()
