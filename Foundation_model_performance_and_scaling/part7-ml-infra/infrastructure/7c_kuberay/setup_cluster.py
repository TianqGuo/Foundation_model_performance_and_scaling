"""Fresh cloud VM bootstrap. Usage: part7C.sh setup --check | --cloud.

Installs single-node K3s and pinned charts; never rents a machine or trains a model.
"""
import argparse
from datetime import datetime, timezone
import json
import hashlib
import getpass
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
K3S_VERSION = "v1.32.13+k3s1"
PLUGIN_VERSION = "0.17.1"
OPERATOR_VERSION = "1.4.2"
NODE = "part7-gpu"
STORAGE = "/var/lib/part7-workspace"


def documents():
    """Return credential-free Kubernetes objects for this fixed single-node setup.

    No inputs or side effects. Charts use K3s's Helm controller; storage is a
    pre-bound Retain hostPath PV/PVC shared by same-node pods, not external backup.
    """
    charts = []
    for name, repo, chart, version, namespace in (
        ("part7-nvidia", "https://nvidia.github.io/k8s-device-plugin", "nvidia-device-plugin", PLUGIN_VERSION, "kube-system"),
        ("kuberay-operator", "https://ray-project.github.io/kuberay-helm/", "kuberay-operator", OPERATOR_VERSION, "default"),
    ):
        charts.append(dict(apiVersion="helm.cattle.io/v1", kind="HelmChart",
                           metadata=dict(name=name, namespace="kube-system"),
                           spec=dict(repo=repo, chart=chart, version=version, targetNamespace=namespace)))
    return [dict(apiVersion="v1", kind="Namespace", metadata=dict(name="part7")), *charts,
            dict(apiVersion="v1", kind="PersistentVolume", metadata=dict(name="part7-workspace"),
                 spec=dict(capacity=dict(storage="100Gi"), accessModes=["ReadWriteOnce"],
                           persistentVolumeReclaimPolicy="Retain", storageClassName="",
                           claimRef=dict(namespace="part7", name="part7-workspace"),
                           hostPath=dict(path=STORAGE, type="Directory"),
                           nodeAffinity=dict(required=dict(nodeSelectorTerms=[dict(matchExpressions=[
                               dict(key="kubernetes.io/hostname", operator="In", values=[NODE])])])))),
            dict(apiVersion="v1", kind="PersistentVolumeClaim",
                 metadata=dict(name="part7-workspace", namespace="part7"),
                 spec=dict(accessModes=["ReadWriteOnce"], storageClassName="", volumeName="part7-workspace",
                           resources=dict(requests=dict(storage="100Gi"))))]


def run(command, *, input_text=None, timeout=120):
    """Execute an argument list with bounded wait; return stdout or raise.

    Optional stdin carries Kubernetes JSON. Child output is captured rather than
    printed; errors report the executable only, avoiding credential disclosure.
    """
    result = subprocess.run(command, input=input_text, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{command[0]} failed (exit {result.returncode}); inspect the VM service logs")
    return result.stdout.strip()


def host_check(allow_installed_k3s=False):
    """Inspect the host before mutations; return noncredential hardware evidence.

    Input: allow only a separately verified partial K3s installation.
    Requires x86_64 Ubuntu VM/systemd, sudo, existing NVIDIA runtime/two GPUs,
    32 CPUs, 120 GiB RAM and 200 GiB free host disk. Refuses containers and existing
    Kubernetes/storage paths; never changes drivers or host CUDA installations.
    """
    if Path("/.dockerenv").exists() or run(["systemd-detect-virt"]) in {"docker", "lxc", "podman"}:
        raise ValueError("Use a full VM, not a Docker rental")
    if run(["ps", "-p", "1", "-o", "comm="]) != "systemd" or os.uname().machine != "x86_64":
        raise ValueError("Require an x86_64 VM running systemd")
    if 'ID=ubuntu' not in Path("/etc/os-release").read_text():
        raise ValueError("This initial bootstrap supports Ubuntu VMs")
    for executable in ("sudo", "curl", "nvidia-smi", "nvidia-container-runtime"):
        if not shutil.which(executable):
            raise ValueError(f"Required existing host tool: {executable}")
    run(["sudo", "-n", "true"])
    if Path("/etc/kubernetes").exists() or Path(STORAGE).exists() or (
            not allow_installed_k3s and (shutil.which("k3s") or Path("/etc/rancher/k3s").exists())):
        raise ValueError("Fresh VM required; existing cluster/storage will not be overwritten")
    gpu = run(["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"])
    memory = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                      if line.startswith("MemTotal:"))) * 1024
    if len(gpu.splitlines()) != 2 or len(os.sched_getaffinity(0)) < 32 or memory < 120 * 1024**3:
        raise ValueError("Require exactly two GPUs, 32 CPUs and 120 GiB host RAM")
    if shutil.disk_usage("/var/lib").free < 200 * 1024**3:
        raise ValueError("Require 200 GiB free host storage")
    return dict(gpus=gpu.splitlines(), cpus=len(os.sched_getaffinity(0)), memory_bytes=memory)


def label_gpu_node(kubectl):
    """Mark the hardware-verified node eligible for the pinned NVIDIA chart.

    Input: configured kubectl argument prefix. Sets the GPU-present label on this
    single node; returns None. The chart requires it when NFD is not installed.
    This label enables plugin placement, not a claim that GPU allocation works.
    """
    run([*kubectl, "label", "node", NODE, "nvidia.com/gpu.present=true", "--overwrite"])


def write_private_config(path, content):
    """Exclusively create a UTF-8 mode-0600 kubeconfig from supplied content.

    Uses builtin open (Path.open has no opener argument). Refuses existing files
    or symlinks and does not print credential content; returns None.
    """
    with open(path, "x", encoding="utf-8", opener=lambda name, flags: os.open(name, flags, 0o600)) as stream:
        stream.write(content + "\n")


def ensure_private_config(path, content, allow_reuse=False):
    """Create config or reuse an exact private file during verified continuation.

    Inputs: destination, current server config and reuse permission. Reject links,
    different credentials, unsafe permissions or another owner; never overwrite.
    """
    if path.is_symlink():
        raise ValueError("Refusing symlink kubeconfig")
    if path.exists():
        info = path.stat()
        if not allow_reuse or info.st_uid != os.getuid() or info.st_mode & 0o777 != 0o600:
            raise ValueError("Refusing existing or unsafe kubeconfig")
        if path.read_text().strip() != content.strip():
            raise ValueError("Existing kubeconfig differs from this server")
    else:
        write_private_config(path, content)


def wait_for_node(kubectl):
    """Poll node registration/API readiness, then wait for the Ready condition.

    Input: kubectl prefix. Retries transient startup errors for up to 180 seconds;
    returns None or raises a bounded timeout. No resources or credentials changed.
    """
    print("Waiting for Kubernetes node registration...", flush=True)
    deadline = time.monotonic() + 180
    while True:
        try:
            nodes = json.loads(run([*kubectl, "get", "nodes", "-o", "json"]))
            if any(node["metadata"]["name"] == NODE for node in nodes["items"]):
                break
        except (RuntimeError, subprocess.TimeoutExpired):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError("Node registration failed; inspect sudo journalctl -u k3s")
        time.sleep(3)
    print("Waiting for node Ready...", flush=True)
    run([*kubectl, "wait", "node/" + NODE, "--for=condition=Ready", "--timeout=180s"], timeout=200)


def verify_partial_setup(previous):
    """Validate evidence for continuation before storage/chart creation.

    Input: previous failed setup directory. Checks receipt pins, no later objects,
    and the actual installed K3s version/node identity. Returns saved receipt;
    refuses completed/advanced/unknown installations without changing them.
    """
    receipt = json.loads((previous / "setup_review.json").read_text())
    if receipt.get("success") or receipt.get("k3s") != K3S_VERSION or receipt.get("node") != NODE:
        raise ValueError("Continuation requires this helper's failed pinned setup receipt")
    if (previous / "setup_objects.json").exists():
        raise ValueError("Setup advanced to chart creation; inspect rather than retry")
    if K3S_VERSION not in run(["/usr/local/bin/k3s", "--version"]):
        raise ValueError("Existing K3s version differs from setup pin")
    nodes = json.loads(run(["sudo", "-n", "k3s", "kubectl", "--request-timeout=30s", "get", "nodes", "-o", "json"]))
    if len(nodes["items"]) != 1 or nodes["items"][0]["metadata"]["name"] != NODE:
        raise ValueError("Continuation requires the original single part7-gpu node")
    return receipt


def install(output, with_hf_secret=False, continue_after_k3s=None):
    """Install on a checked fresh VM and save noncredential setup evidence.

    Inputs: fresh output directory and optional interactive Hub Secret flag. Writes system services, charts, shared storage
    and private user kubeconfig; waits for GPU/operator/API/PVC readiness. Failures
    retain partial installation for diagnosis. Optional continue_after_k3s accepts
    only verified failure before storage/chart creation, skipping reinstallation.
    No model loading, training or GPU kernel test is performed. Optional Hub token
    is read silently (or from HF_TOKEN) and sent through stdin, never written to evidence.
    """
    previous = verify_partial_setup(continue_after_k3s) if continue_after_k3s else None
    host = host_check(allow_installed_k3s=bool(previous))
    if previous and previous["host"]["gpus"] != host["gpus"]:
        raise ValueError("Continuation must use the original VM GPUs")
    kubeconfig = Path.home() / ".kube" / "part7-config"
    if kubeconfig.exists() and not previous:
        raise ValueError("Private part7 kubeconfig already exists; refusing overwrite")
    token = None
    if with_hf_secret:
        token = os.environ.get("HF_TOKEN") or getpass.getpass("Hugging Face write token (hidden): ")
        if not token.strip():
            raise ValueError("Empty Hub token")
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(success=False, host=host, k3s=K3S_VERSION,
                   nvidia_plugin=PLUGIN_VERSION, kuberay=OPERATOR_VERSION,
                   node=NODE, storage=STORAGE, context="default",
                   limitations=["GPU pod/image test pending", "host storage is lost on VM deletion"])
    try:
        if previous:
            receipt["continued_from"] = str(continue_after_k3s.resolve())
            receipt["installer_sha256"] = previous.get("installer_sha256")
        else:
            installer = output / "install-k3s.sh"
            run(["curl", "--fail", "--location", "--max-time", "120", "https://get.k3s.io", "-o", str(installer)])
            receipt["installer_sha256"] = hashlib.sha256(installer.read_bytes()).hexdigest()
            run(["sudo", "-n", "env", f"INSTALL_K3S_VERSION={K3S_VERSION}", "sh", str(installer),
                 "server", "--node-name", NODE, "--default-runtime", "nvidia", "--disable", "traefik",
                 "--disable", "servicelb", "--write-kubeconfig-mode", "600"], timeout=600)
        kubeconfig.parent.mkdir(parents=True, exist_ok=True)
        # Credentials stay outside results, never in retained evidence or stdout.
        config = run(["sudo", "-n", "cat", "/etc/rancher/k3s/k3s.yaml"])
        ensure_private_config(kubeconfig, config, allow_reuse=bool(previous))
        kubectl = ["/usr/local/bin/kubectl", "--kubeconfig", str(kubeconfig), "--context", "default", "--request-timeout=30s"]
        wait_for_node(kubectl)
        label_gpu_node(kubectl)
        run(["sudo", "-n", "mkdir", STORAGE])
        run(["sudo", "-n", "chown", "1000:1000", STORAGE])
        objects = documents()
        (output / "setup_objects.json").write_text(json.dumps(objects, indent=2) + "\n")
        for obj in objects:
            run([*kubectl, "create", "-f", "-"], input_text=json.dumps(obj))
        print("Waiting for NVIDIA GPU resources and KubeRay operator...", flush=True)
        # Chart creation precedes asynchronous Helm installation; poll boundedly.
        deadline = time.monotonic() + 600
        while True:
            node = json.loads(run([*kubectl, "get", "node", NODE, "-o", "json"]))
            deployments = json.loads(run([*kubectl, "get", "deployments", "-n", "default", "-o", "json"]))
            ready = any(item["metadata"]["name"] == "kuberay-operator" and
                        item.get("status", {}).get("availableReplicas", 0) >= 1 for item in deployments["items"])
            if int(node["status"].get("allocatable", {}).get("nvidia.com/gpu", 0)) == 2 and ready:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("GPU plugin/operator not ready; inspect kube-system Helm jobs")
            time.sleep(5)
        run([*kubectl, "wait", "crd/rayjobs.ray.io", "--for=condition=Established", "--timeout=120s"], timeout=140)
        run([*kubectl, "wait", "pvc/part7-workspace", "-n", "part7",
             "--for=jsonpath={.status.phase}=Bound", "--timeout=120s"], timeout=140)
        receipt["installed_k3s"] = run(["/usr/local/bin/k3s", "--version"])
        if K3S_VERSION not in receipt["installed_k3s"]:
            raise ValueError("Installed K3s does not match the requested pin")
        # Reuse the same readiness rules the experiment expects.
        from importlib import import_module
        import sys
        sys.path.insert(0, str(ROOT))
        check = import_module("infrastructure.7c_kuberay.cluster_check")
        def get_object(kind, name, namespace=None):
            """Read one Kubernetes JSON object; optional namespace scopes the query."""
            args = [*kubectl, "get", kind, name, "-o", "json"]
            if namespace:
                args += ["-n", namespace]
            return json.loads(run(args))
        receipt["cluster"] = check.validate(get_object("node", NODE),
            get_object("pvc", "part7-workspace", "part7"),
            get_object("deployment", "kuberay-operator", "default"),
            get_object("crd", "rayjobs.ray.io"))
        if token:
            secret = dict(apiVersion="v1", kind="Secret", type="Opaque",
                          metadata=dict(name="part7-huggingface", namespace="part7"),
                          stringData=dict(token=token))
            run([*kubectl, "create", "-f", "-"], input_text=json.dumps(secret))
        receipt["hub_secret_created"] = bool(token)
        receipt["success"] = True
    finally:
        (output / "setup_review.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Setup complete. Export KUBECONFIG={kubeconfig}; GPU pod verification remains pending.")


def main():
    """Parse explicit check/cloud mode; preview locally or bootstrap a fresh VM.

    --check prints pins/objects without host inspection, commands or downloads.
    --cloud installs only after all host checks pass. Output defaults under results.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--cloud", action="store_true")
    parser.add_argument("--continue-after-k3s", type=Path, help="Failed setup directory; continue only before kubeconfig/storage creation")
    parser.add_argument("--with-hf-secret", action="store_true", help="Create Hub Secret from HF_TOKEN or hidden prompt")
    parser.add_argument("--output", type=Path, default=ROOT / "results/7c" /
                        ("setup_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
    args = parser.parse_args()
    if args.check:
        print(json.dumps(dict(k3s=K3S_VERSION, nvidia_plugin=PLUGIN_VERSION,
                              kuberay=OPERATOR_VERSION, objects=documents()), indent=2))
    else:
        install(args.output, args.with_hf_secret, args.continue_after_k3s)


if __name__ == "__main__":
    main()
