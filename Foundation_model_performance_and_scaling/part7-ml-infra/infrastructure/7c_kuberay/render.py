"""Render a digest-pinned CPU-head/two-GPU-worker RayJob preflight, without API calls.

JSON output is also valid Kubernetes YAML. Select preflight or one native verl
training/recovery phase; manifest preparation does not complete milestone 7C.
"""
import argparse
import json
from pathlib import Path
import re
import shlex

RAY_VERSION = "2.49.2"
KUBERAY_VERSION = "1.4.2"


def dns_label(value):
    """Validate a namespace/object DNS label; return it or raise ValueError.

    Input: string, at most 63 lowercase alphanumeric/hyphen characters. Used for
    metadata and output paths so shell entrypoints cannot contain injected text.
    """
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", value):
        raise ValueError("Use a lowercase Kubernetes DNS label of at most 63 characters")
    return value


def pinned_image(value):
    """Require an immutable registry image digest; return it or raise ValueError.

    Input: registry/repository@sha256:64-lowercase-hex reference. Tags, whitespace
    and shell syntax are rejected. A valid format does not prove the image exists.
    """
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9./:_-]*@sha256:[a-f0-9]{64}", value):
        raise ValueError("Provide a registry image pinned by @sha256:64-hex, not a mutable tag")
    return value


def render_job(image, namespace, name, pvc, secret, node_name, pull_secret=None, phase="preflight", checkpoint=None):
    """Return a RayJob document for cloud preflight or native training/recovery.

    Inputs: immutable image, namespace/job/PVC/Secret names and chosen GPU node;
    optional image pull Secret name, phase and shared checkpoint path. Output: Kubernetes object dictionary. No
    Secrets/PVCs are created. Both pods share the selected node and existing PVC,
    enabling initial single-node RWO storage. Requests reserve two whole GPUs;
    shared-memory limits count against pod RAM. All labels/digest are validated.
    """
    image = pinned_image(image)
    namespace, name, pvc, secret = map(dns_label, (namespace, name, pvc, secret))
    if not isinstance(node_name, str) or len(node_name) > 253 or not re.fullmatch(r"[a-z0-9][a-z0-9.-]*[a-z0-9]|[a-z0-9]", node_name):
        raise ValueError("Provide the actual Kubernetes GPU node hostname")
    if pull_secret:
        dns_label(pull_secret)

    def pod(role, cpu, memory, shm, gpus):
        """Build one pod spec with resources, persistent workspace and file Secret."""
        resources = {"cpu": str(cpu), "memory": memory}
        if gpus:
            resources["nvidia.com/gpu"] = gpus
        container = dict(name="ray-" + role, image=image, imagePullPolicy="IfNotPresent",
                         workingDir="/opt/part7", resources=dict(requests=dict(resources), limits=dict(resources)),
                         securityContext=dict(allowPrivilegeEscalation=False, capabilities=dict(drop=["ALL"])),
                         env=[dict(name="PART7_IMAGE", value=image),dict(name="POD_NAME", valueFrom=dict(fieldRef=dict(fieldPath="metadata.name"))),
                              dict(name="POD_UID", valueFrom=dict(fieldRef=dict(fieldPath="metadata.uid"))),
                              dict(name="NODE_NAME", valueFrom=dict(fieldRef=dict(fieldPath="spec.nodeName")))],
                         volumeMounts=[dict(name="workspace", mountPath="/workspace"),
                                       dict(name="dshm", mountPath="/dev/shm"),
                                       dict(name="hub-token", mountPath="/run/secrets/huggingface", readOnly=True)])
        if role == "head":
            container["ports"] = [dict(name=key, containerPort=number)
                                  for key, number in (("gcs", 6379), ("dashboard", 8265), ("client", 10001), ("metrics", 8080))]
        spec = dict(automountServiceAccountToken=False,
                    securityContext=dict(runAsNonRoot=True, runAsUser=1000, runAsGroup=1000,
                                         fsGroup=1000, fsGroupChangePolicy="OnRootMismatch"),
                    nodeSelector={"kubernetes.io/hostname": node_name},
                    containers=[container],
                    volumes=[dict(name="workspace", persistentVolumeClaim=dict(claimName=pvc)),
                             dict(name="dshm", emptyDir=dict(medium="Memory", sizeLimit=shm)),
                             dict(name="hub-token", secret=dict(secretName=secret, defaultMode=0o440,
                                                                items=[dict(key="token", path="token")]))])
        if pull_secret:
            spec["imagePullSecrets"] = [dict(name=pull_secret)]
        return dict(metadata=dict(labels={"part7-milestone": "7c", "part7-run": name}), spec=spec)

    if phase not in {"preflight", "baseline", "interrupt", "resume"}:
        raise ValueError("Unknown workload phase")
    if phase == "resume" and not checkpoint:
        raise ValueError("Resume manifest requires a checkpoint path")
    if checkpoint and (not isinstance(checkpoint, str) or not checkpoint.startswith("/workspace/")
                       or ".." in Path(checkpoint).parts or "\\" in checkpoint):
        raise ValueError("Checkpoint must be an absolute path under /workspace")
    command = ["python", "-u", "/opt/part7/infrastructure/7c_kuberay/" +
               ("preflight.py" if phase == "preflight" else "run_7c.py"),
               "--cloud", "--output", f"/workspace/runs/{name}"]
    if phase != "preflight":
        command += ["--phase", phase]
    if phase == "resume":
        command += ["--checkpoint", checkpoint]
    return dict(apiVersion="ray.io/v1", kind="RayJob",
                metadata=dict(name=name, namespace=namespace, labels={"part7-milestone": "7c"},
                              annotations={"part7/kuberay-version": KUBERAY_VERSION,
                                           "part7/scope": "gpu-preflight-only" if phase == "preflight" else "training-" + phase}),
                spec=dict(entrypoint=shlex.join(command),
                          entrypointNumCpus=0, entrypointNumGpus=0,
                          submissionMode="K8sJobMode", backoffLimit=0,
                          shutdownAfterJobFinishes=True, ttlSecondsAfterFinished=600,
                          activeDeadlineSeconds=900 if phase == "preflight" else 2700,
                          rayClusterSpec=dict(rayVersion=RAY_VERSION, enableInTreeAutoscaling=False,
                              headGroupSpec=dict(rayStartParams={"dashboard-host": "0.0.0.0", "num-cpus": "2", "num-gpus": "0",
                                                                 "object-store-memory": "536870912"},
                                                 template=pod("head", 2, "8Gi", "2Gi", 0)),
                              workerGroupSpecs=[dict(groupName="training", replicas=1, minReplicas=1, maxReplicas=1,
                                  rayStartParams={"num-cpus": "24", "num-gpus": "2", "object-store-memory": "4294967296"},
                                  template=pod("worker", 24, "96Gi", "16Gi", 2))])))


def main():
    """Parse immutable image/resource names and write one fresh JSON manifest.

    Input: CLI arguments. Output: a local file (or stdout); no cluster contacts.
    Existing output files are rejected. Use part7C.sh render for the same command.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--namespace", default="part7")
    parser.add_argument("--name", default="part7-7c-preflight")
    parser.add_argument("--pvc", default="part7-workspace")
    parser.add_argument("--secret", default="part7-huggingface")
    parser.add_argument("--node-name", required=True)
    parser.add_argument("--pull-secret")
    parser.add_argument("--phase", choices=["preflight", "baseline", "interrupt", "resume"], default="preflight")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = render_job(args.image, args.namespace, args.name, args.pvc, args.secret, args.node_name, args.pull_secret, args.phase, args.checkpoint)
    content = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(content)
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
