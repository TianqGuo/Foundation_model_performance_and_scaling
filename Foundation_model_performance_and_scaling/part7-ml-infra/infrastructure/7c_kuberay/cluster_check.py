"""Read-only Kubernetes prerequisites check; no jobs, model loads or provisioning."""
import argparse
from decimal import Decimal
from importlib import import_module
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
control = import_module("infrastructure.7c_kuberay.job_control")


def quantity(value):
    """Convert common Kubernetes resource quantities to Decimal base units.

    Input: CPU/memory/storage quantity string. Returns cores for CPU or bytes for
    memory/storage, depending on the input. Reject unsupported formats rather than
    silently reporting sufficient capacity. No cluster or package dependencies.
    """
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(m|Ki|Mi|Gi|Ti|K|M|G|T)?", str(value))
    if not match:
        raise ValueError("Unsupported Kubernetes resource quantity")
    scales = {None:1, "m":Decimal("0.001"), "Ki":1024, "Mi":1024**2, "Gi":1024**3,
              "Ti":1024**4, "K":1000, "M":1000**2, "G":1000**3, "T":1000**4}
    return Decimal(match[1]) * scales[match[2]]


def validate(node, pvc, operator, crd):
    """Validate initial-node capacity, shared PVC, operator readiness and served API.

    Inputs: Kubernetes JSON objects. Returns concise noncredential report; reject
    unready/unschedulable nodes, insufficient allocatable capacity, unbound/RWOP
    storage, unavailable/wrong operator pin or absent served RayJob v1 API.
    Capacity is not currently free capacity, GPU memory or a performance measurement.
    """
    if node.get("spec", {}).get("unschedulable") or not any(
            condition["type"] == "Ready" and condition["status"] == "True"
            for condition in node["status"].get("conditions", [])):
        raise ValueError("Selected Kubernetes node is not ready/schedulable")
    if node["metadata"].get("labels", {}).get("kubernetes.io/hostname") != node["metadata"]["name"]:
        raise ValueError("Selected node name must match the current hostname selector")
    if any(taint.get("effect") in {"NoSchedule", "NoExecute"} for taint in node.get("spec", {}).get("taints", [])):
        raise ValueError("Current pod templates have no matching tolerations for node taints")
    resources = node["status"]["allocatable"]
    if (quantity(resources.get("cpu", "0")) < 26 or quantity(resources.get("memory", "0")) < 104 * 1024**3
            or quantity(resources.get("nvidia.com/gpu", "0")) < 2):
        raise ValueError("Node lacks initial 26 CPU / 104 GiB / 2 GPU allocatable capacity")
    modes = pvc["spec"]["accessModes"]
    if pvc.get("status", {}).get("phase") != "Bound":
        raise ValueError("PVC is not Bound")
    if "ReadWriteOncePod" in modes or not {"ReadWriteOnce", "ReadWriteMany"}.intersection(modes):
        raise ValueError("PVC must permit head/worker read-write sharing on one node")
    capacity = pvc["status"].get("capacity", {}).get("storage", "0")
    if quantity(capacity) < 100 * 1024**3:
        raise ValueError("Allow at least 100 GiB persistent capacity for full-state recovery")
    status = operator.get("status", {})
    if (status.get("observedGeneration", 0) < operator["metadata"].get("generation", 1)
            or status.get("availableReplicas", 0) < 1):
        raise ValueError("KubeRay operator deployment is not available at its current generation")
    images = [container["image"] for container in operator["spec"]["template"]["spec"]["containers"]]
    version = operator["metadata"].get("labels", {}).get("app.kubernetes.io/version")
    if version not in {"1.4.2", "v1.4.2"} and not any(image.endswith(":v1.4.2") for image in images):
        raise ValueError("Confirm the candidate KubeRay v1.4.2 operator identity")
    if not any(v["name"] == "v1" and v.get("served") for v in crd["spec"]["versions"]):
        raise ValueError("RayJob v1 API is not served")
    if not any(c["type"] == "Established" and c["status"] == "True" for c in crd.get("status", {}).get("conditions", [])):
        raise ValueError("RayJob CRD is not established")
    return dict(node=node["metadata"]["name"], allocatable=resources, pvc=pvc["metadata"]["name"],
        access_modes=modes, storage_capacity=capacity, operator_images=images,
        rayjob_api="ray.io/v1", scope="read-only capacity/API check; not free resources or GPU/image execution")


def inspect(context, namespace, node_name, pvc_name, operator_namespace, operator_name):
    """Fetch prerequisite metadata in an explicit context and validate it.

    Inputs: cluster/object identities. Returns report; runs only kubectl get, never
    queries Secret contents or creates/modifies resources. Credentials remain in
    kubectl configuration. API/permission errors stop the check and propagate.
    """
    for value in (namespace, pvc_name, operator_namespace, operator_name):
        control.renderer.dns_label(value)
    api = control.kubectl
    node = json.loads(api(context, namespace, ["get", "node", node_name, "-o", "json"]))
    pvc = json.loads(api(context, namespace, ["get", "pvc", pvc_name, "-o", "json"]))
    operator = json.loads(api(context, operator_namespace, ["get", "deployment", operator_name, "-o", "json"]))
    crd = json.loads(api(context, namespace, ["get", "crd", "rayjobs.ray.io", "-o", "json"]))
    return validate(node, pvc, operator, crd)


def main():
    """Run explicit-context read-only inspection and optionally save a fresh report.

    Input: CLI names/context. Returns None and prints safe summarized metadata;
    no model, GPU operation, installation, Secret reading or provisioning occurs.
    Existing report paths are rejected to preserve evidence.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True)
    parser.add_argument("--node-name", required=True)
    parser.add_argument("--namespace", default="part7")
    parser.add_argument("--pvc", default="part7-workspace")
    parser.add_argument("--operator-namespace", default="default")
    parser.add_argument("--operator-name", default="kuberay-operator")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect(args.context, args.namespace, args.node_name, args.pvc, args.operator_namespace, args.operator_name)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
