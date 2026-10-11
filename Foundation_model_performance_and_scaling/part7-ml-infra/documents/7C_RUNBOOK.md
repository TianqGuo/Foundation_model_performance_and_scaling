# 7C runbook — preparation increment

This runner builds an image, renders preflight/training/recovery manifests, or
executes a native training phase inside a prepared pod. It does not install
Kubernetes/KubeRay, create storage or Secrets, or submit jobs. [7C_PLAN.md](7C_PLAN.md) records the remaining implementation and evidence.
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
apply/watch/evidence-retention/cleanup shell workflow is still pending. Native
training/recovery phases are now prepared as described below; cloud verification
remains pending.

## Existing-cluster training/recovery increment

`infrastructure/7c_kuberay/run_7c.py` now composes the unchanged 7A workload,
connects with `ray.init(address="auto")`, prepares the model on a two-GPU worker
actor and runs native verl's TaskRunner on that worker without reserving a GPU
for the coordinator. The preparation actor is released before training starts.
The launcher explicitly merges verl runtime defaults when attaching to Ray.
Existing 7B trainer/worker audits implement checkpoint-boundary interruption,
manifest creation and native full-state resume; model/data/source/environment
identities must match. Separate phase RayJobs are required for fresh-runtime
recovery. This code is CPU-checked, not cloud-verified. Fifteen 7C CPU tests pass,
including all four phase manifests against the pinned CRD. Cloud paths must stay
on the PVC under `/workspace`; recovery also requires the same recorded image digest.

The renderer now accepts `--phase preflight|baseline|interrupt|resume` and
`--checkpoint /workspace/.../global_step_2` for resume. Use a unique `--name`
for each job/output. Training phases have a 45-minute active deadline. The
prepared data bundle must exist at `/workspace/data/step2_math_final`; automatic
model downloads use the mounted Hub Secret and shared `/workspace/models`.
No token is passed in Ray runtime environment or generated manifests.

Inside a prepared pod, the milestone runner can invoke a phase directly:

```bash
bash infrastructure/7c_kuberay/part7C.sh phase --cloud --phase interrupt \
  --output /workspace/runs/7c-interrupt
bash infrastructure/7c_kuberay/part7C.sh phase --cloud --phase resume \
  --checkpoint /workspace/runs/7c-interrupt/checkpoints/global_step_2 \
  --output /workspace/runs/7c-resume
```

Normally Kubernetes invokes these phases through rendered RayJobs. End the
interrupted RayJob/cluster before starting the resume RayJob; invoking both in
the same surviving runtime does not establish fresh-runtime recovery. No
apply/watch/cleanup controller or direct Hub checkpoint transfer is integrated
into 7C yet. PVC manifests can be resumed, but durable external backup and
Kubernetes lifecycle/evidence review remain required before 7C completion.
