# 7C runbook — preparation increment

This runner currently prepares an image or a GPU-preflight manifest. It does not
install Kubernetes/KubeRay, create storage or Secrets, deploy training, or perform
recovery. [7C_PLAN.md](7C_PLAN.md) records the remaining implementation and evidence.
Do not rent another ordinary GPU container solely to run this stage: first confirm
access to Kubernetes with NVIDIA GPU resources and a compatible storage class.

## Cluster and image requirements

The initial layout requires one selected node with two whole GPUs, at least
26 allocatable CPUs and 104 GiB memory for the Ray pods, plus capacity for system,
operator and submission pods. GPU model/memory and budget must be agreed before
provisioning. The NVIDIA device plugin must advertise `nvidia.com/gpu` resources.
The manifest requests one worker with two GPUs, not two independently placed pods.
The head requests no GPU. This retains the same-node two-rank topology used in 7B.

Use an existing namespace, a durable PVC accessible to UID/GID 1000, and a Secret
with a `token` key containing a suitable Hub token. Both Ray pods mount the PVC
at `/workspace` and the token read-only at `/run/secrets/huggingface/token`.
Both pods select the same node so an initial ReadWriteOnce volume can be shared
there. Confirm storage capacity, attachment semantics and filesystem permissions;
allow substantial capacity for caches and complete optimizer checkpoints when
training is added. Shared-memory mounts are 2 GiB on the head and 16 GiB on the
worker. They consume pod memory; do not subtract them from resource planning.

KubeRay v1.4.2 is the candidate operator pin. The manifest's actual CRD schema has
been checked, but live operator compatibility and storage/GPU setup are pending.
Use a registry image accessible to the cluster; `--pull-secret` optionally names
an existing registry Secret. Never put credential values into source or manifests.

## Prepared commands

From the Part 7 root, render locally with an actual published registry digest:

```bash
bash infrastructure/7c_kuberay/part7C.sh render \
  --image REGISTRY/PART7_IMAGE@sha256:ACTUAL_64_HEX_DIGEST \
  --node-name ACTUAL_KUBERNETES_GPU_NODE \
  --namespace part7 --pvc part7-workspace --secret part7-huggingface \
  --name part7-7c-preflight --output results/7c/preflight.json
```

The uppercase values are placeholders, not runnable defaults. Output is JSON,
which Kubernetes accepts as a manifest. Existing output files are rejected to
avoid replacing a reviewed manifest accidentally. Rendering creates no resources.

For the eventual cloud image build, obtain the immutable digest of the compatible
custom base image. It must provide uv, git and CUDA 13 development tools:

```bash
bash infrastructure/7c_kuberay/part7C.sh build --cloud \
  REGISTRY/BASE_IMAGE@sha256:ACTUAL_64_HEX_DIGEST part7-training:7c
```

This builds the isolated locked environment and source once. It does not download
a model or dataset, push an image or deploy a job. The printed local configuration
ID is not the published registry manifest digest required by the renderer.
The Dockerfile has not yet been build-tested; retain build output and resolve any
base-image compatibility issues before deployment. Parts 1–6 images/environments
are preserved. Git clone/pull remains the preferred delivery method for approved
source commits; rsync can deliver uncommitted source to the cloud build machine.

## What the preflight will establish

Once cluster access and image publication are ready, the rendered RayJob connects
to the existing Ray cluster and runs two concurrent GPU probes. Success requires
two distinct GPU assignments on the same worker pod, one visible CUDA device per
actor, a successful CUDA kernel, compatible imports and Hub Secret authentication.
It saves `preflight.json` and `job_status.json` under
`/workspace/runs/<job-name>`. No model loading or training occurs in this probe.

The job is bounded to 15 minutes, with no automatic retry and cluster shutdown
requested after completion. A 10-minute finished-job TTL is configured. These
are intended settings, not verified cleanup behavior; retain pod/operator logs
and Kubernetes status before automatic cleanup, and verify actual lifecycle.
The external PVC and Secret must remain available for later jobs. The full
apply/watch/evidence-retention/training/recovery shell workflow is still pending.
