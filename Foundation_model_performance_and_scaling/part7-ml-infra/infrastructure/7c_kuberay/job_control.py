"""Submit one generated RayJob, retain lightweight evidence, optionally clean up."""
import argparse
from copy import deepcopy
from importlib import import_module
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
renderer = import_module("infrastructure.7c_kuberay.render")


def validate_manifest(document):
    """Require an exact renderer-produced RayJob, returning namespace/name/phase.

    Input: parsed JSON object. Output: validated identity tuple. Reject edited
    commands/resources/credentials or unrelated objects before cluster contact.
    This controller deliberately supports only the bounded Part 7 topology.
    """
    metadata, spec = document["metadata"], document["spec"]
    scope = metadata["annotations"]["part7/scope"]
    phase = "preflight" if scope == "gpu-preflight-only" else scope.removeprefix("training-")
    head = spec["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]
    volumes = {v["name"]: v for v in head["volumes"]}
    command = shlex.split(spec["entrypoint"])
    checkpoint = command[command.index("--checkpoint") + 1] if "--checkpoint" in command else None
    transfers = {flag: command[command.index(flag) + 1] if flag in command else None
                 for flag in ("--upload-checkpoint-repo", "--restore-repo", "--restore-revision")}
    expected = renderer.render_job(head["containers"][0]["image"], metadata["namespace"], metadata["name"],
        volumes["workspace"]["persistentVolumeClaim"]["claimName"], volumes["hub-token"]["secret"]["secretName"],
        head["nodeSelector"]["kubernetes.io/hostname"],
        head.get("imagePullSecrets", [{}])[0].get("name"), phase, checkpoint,
        transfers["--upload-checkpoint-repo"], transfers["--restore-repo"], transfers["--restore-revision"])
    if document != expected:
        raise ValueError("Use an unmodified manifest from part7C.sh render")
    return metadata["namespace"], metadata["name"], phase


def terminal_state(status):
    """Map RayJob status to success/failure/pending, without trusting pod exit alone.

    Input: RayJob status dictionary. Returns string; deployment failure takes
    precedence over job success. Unknown/transitional states remain pending.
    """
    if str(status.get("jobDeploymentStatus", "")).lower() in {"failed", "validationfailed"}:
        return "failure"
    job = str(status.get("jobStatus", "")).upper()
    if job in {"FAILED", "STOPPED"}:
        return "failure"
    return "success" if job == "SUCCEEDED" else "pending"


def kubectl(context, namespace, arguments, *, binary=False):
    """Run kubectl in an explicit context/namespace with bounded API/command waits.

    Inputs: context, namespace and argument list. Returns stdout bytes or text;
    raises a sanitized RuntimeError on failure. No shell expansion or credentials
    are printed. Cluster mutations occur only when requested by the caller.
    """
    result = subprocess.run(["kubectl", "--context", context, "--namespace", namespace,
        "--request-timeout=30s", *arguments], capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError("kubectl operation failed: " + arguments[0])
    return result.stdout if binary else result.stdout.decode()


def collect(context, namespace, cluster, output):
    """Capture pod placement/status and available logs for the named Ray cluster.

    Inputs: explicit cluster identity and local evidence path. Returns pod list;
    never reads Secrets or dumps environment values. Log errors are recorded and
    retried on later snapshots. Full Kubernetes pod specs are deliberately omitted.
    """
    renderer.dns_label(cluster)
    pods = json.loads(kubectl(context, namespace, ["get", "pods", "-l", "ray.io/cluster=" + cluster, "-o", "json"]))["items"]
    summaries, errors = [], []
    for pod in pods:
        name = pod["metadata"]["name"]
        summaries.append(dict(name=name, uid=pod["metadata"]["uid"], labels=pod["metadata"].get("labels", {}),
            node=pod["spec"].get("nodeName"), status=pod.get("status", {}),
            containers=[{key: c.get(key) for key in ("name", "image", "resources")} for c in pod["spec"]["containers"]]))
        for container in pod["spec"]["containers"]:
            try:
                logs = kubectl(context, namespace, ["logs", name, "-c", container["name"], "--timestamps"])
                (output / (name + "_" + container["name"] + ".log")).write_text(logs)
            except RuntimeError:
                errors.append(dict(pod=name, container=container["name"]))
    (output / "pods.json").write_text(json.dumps(summaries, indent=2))
    (output / "log_capture_errors.json").write_text(json.dumps(errors, indent=2))
    return pods


def retain_workspace(context, namespace, head, name, output):
    """Stream small PVC run evidence from the head and require successful phase status.

    Inputs: head pod/job identity and local output. Returns None. Writes compressed
    evidence without downloading tensors/model caches or extracting paths. Retains
    checkpoint manifests; rejects unsuccessful/missing job_status before cleanup.
    """
    script = """import sys,tarfile
from pathlib import Path
root=Path(sys.argv[1])
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for path in sorted(root.rglob('*')):
  parts=path.relative_to(root).parts
  if path.is_file() and not path.is_symlink() and (path.name=='checkpoint_manifest.json' or
   (not any(p in {'checkpoints','restored','.cache','hf_model'} for p in parts) and
    path.suffix not in {'.pt','.bin','.safetensors'})):
   archive.add(path,arcname=path.relative_to(root).as_posix(),recursive=False)
"""
    data = kubectl(context, namespace, ["exec", head, "-c", "ray-head", "--", "python", "-c", script,
                                       "/workspace/runs/" + name], binary=True)
    (output / "workspace_evidence.tar.gz").write_bytes(data)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        status = json.load(archive.extractfile("job_status.json"))
    if status.get("success") is not True:
        raise RuntimeError("PVC phase status is not successful; keep resources for inspection")
    (output / "phase_status.json").write_text(json.dumps(status, indent=2))


def retain_driver_logs(context, namespace, head, job_id, output):
    """Fetch the Ray job driver's complete logs through the head's local dashboard.

    Inputs: head pod, Ray submission ID and evidence path. Returns None; writes
    driver.log. Missing IDs or API/exec failure prevent automatic cleanup. Uses
    Ray's public SDK, not a dashboard service exposed outside the cluster.
    """
    if not isinstance(job_id, str) or not job_id:
        raise RuntimeError("RayJob status has no Ray submission ID for driver logs")
    script = "import sys; from ray.job_submission import JobSubmissionClient; sys.stdout.write(JobSubmissionClient('http://127.0.0.1:8265').get_job_logs(sys.argv[1]))"
    logs = kubectl(context, namespace, ["exec", head, "-c", "ray-head", "--", "python", "-c", script, job_id])
    (output / "driver.log").write_text(logs)


def cleanup(context, namespace, name, uid, cluster, output):
    """Delete only this newly created RayJob after evidence, verifying UID and absence.

    Inputs: recorded job UID/cluster and evidence path. Returns None; writes cleanup
    receipt. Never deletes PVCs, Secrets, namespaces or unrelated jobs. Refuse name
    reuse; fail if the cluster or its Ray pods survive the bounded cleanup check.
    """
    current = json.loads(kubectl(context, namespace, ["get", "rayjob", name, "-o", "json"]))
    if current["metadata"]["uid"] != uid:
        raise RuntimeError("RayJob UID changed; refuse cleanup")
    kubectl(context, namespace, ["delete", "rayjob", name, "--wait=true", "--timeout=90s"])
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        remaining = kubectl(context, namespace, ["get", "raycluster", cluster, "--ignore-not-found", "-o", "json"])
        pods = json.loads(kubectl(context, namespace, ["get", "pods", "-l", "ray.io/cluster=" + cluster, "-o", "json"]))["items"]
        if not remaining.strip() and not pods:
            (output / "cleanup.json").write_text(json.dumps(dict(verified=True, rayjob_uid=uid,
                raycluster=cluster, scope="RayJob, RayCluster and Ray pods; PVC/Secret preserved"), indent=2))
            return
        time.sleep(3)
    raise RuntimeError("Cleanup did not complete; inspect retained cluster identity")


def run(document, context, output, cleanup_on_success=False):
    """Create one bounded RayJob, monitor status and retain evidence before cleanup.

    Inputs: generated manifest, explicit context, fresh local output, cleanup flag.
    Returns None. Keeps Ray cluster alive for collection; failures/timeouts retain
    resources and receipt for diagnosis. This does not provision Kubernetes nodes.
    """
    namespace, name, phase = validate_manifest(document)
    output.mkdir(parents=True, exist_ok=False)
    submitted = deepcopy(document)
    # Automatic shutdown can remove pods/logs before the client collects evidence.
    submitted["spec"]["shutdownAfterJobFinishes"] = False
    path = output / "submitted_rayjob.json"
    path.write_text(json.dumps(submitted, indent=2))
    if kubectl(context, namespace, ["get", "rayjob", name, "--ignore-not-found", "-o", "json"]).strip():
        raise FileExistsError("RayJob already exists; use a unique run name")
    kubectl(context, namespace, ["create", "--dry-run=server", "-f", str(path), "-o", "name"])
    created = json.loads(kubectl(context, namespace, ["create", "-f", str(path), "-o", "json"]))
    uid = created["metadata"]["uid"]
    receipt = dict(success=False, context=context, namespace=namespace, name=name, uid=uid, phase=phase,
                   cleanup_requested=cleanup_on_success)
    (output / "submission.json").write_text(json.dumps(receipt, indent=2))
    deadline = time.monotonic() + submitted["spec"]["activeDeadlineSeconds"] + 300
    try:
        while time.monotonic() < deadline:
            job = json.loads(kubectl(context, namespace, ["get", "rayjob", name, "-o", "json"]))
            if job["metadata"]["uid"] != uid:
                raise RuntimeError("RayJob UID changed")
            status = job.get("status", {})
            (output / "rayjob_status.json").write_text(json.dumps(status, indent=2))
            cluster = status.get("rayClusterName")
            pods = collect(context, namespace, cluster, output) if cluster else []
            state = terminal_state(status)
            if state == "failure":
                raise RuntimeError("RayJob failed; resources retained for inspection")
            if state == "success":
                heads = [p for p in pods if p["metadata"].get("labels", {}).get("ray.io/node-type") == "head"]
                if len(heads) != 1:
                    raise RuntimeError("Cannot identify one head pod for evidence collection")
                if json.loads((output / "log_capture_errors.json").read_text()):
                    raise RuntimeError("Pod log retention incomplete; resources retained")
                retain_driver_logs(context, namespace, heads[0]["metadata"]["name"], status.get("jobId"), output)
                retain_workspace(context, namespace, heads[0]["metadata"]["name"], name, output)
                if cleanup_on_success:
                    cleanup(context, namespace, name, uid, cluster, output)
                receipt["success"] = True
                return
            time.sleep(10)
        raise TimeoutError("Bounded monitoring expired; resources retained")
    finally:
        (output / "submission.json").write_text(json.dumps(receipt, indent=2))


def main():
    """Parse local --check or explicit --cloud submission with context/output.

    Input: generated JSON manifest and arguments. Returns None. CPU check validates
    without invoking kubectl; cloud mode may create/delete GPU jobs but never creates
    a cluster/operator/storage/Secret. Cleanup requires the explicit success flag.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--cloud", action="store_true")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--context")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cleanup-on-success", action="store_true")
    args = parser.parse_args()
    document = json.loads(args.manifest.read_text())
    identity = validate_manifest(document)
    if args.check:
        print("Validated generated RayJob:", "/".join(identity))
        return
    if not args.context or not args.output:
        parser.error("Cloud submission requires --context and fresh --output")
    run(document, args.context, args.output.resolve(), args.cleanup_on_success)


if __name__ == "__main__":
    main()
