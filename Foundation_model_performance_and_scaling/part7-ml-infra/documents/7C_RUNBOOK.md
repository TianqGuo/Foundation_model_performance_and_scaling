# 7C runbook — preparation increment

This runner builds an image, renders preflight/training/recovery manifests, or
executes a native training phase inside a prepared pod. It does not install
Kubernetes/KubeRay or create storage or Secrets. The `job` action now submits
and monitors one generated RayJob on an existing cluster. [7C_PLAN.md](7C_PLAN.md) records the remaining implementation and evidence.
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
The external PVC and Secret must remain available for later jobs. The submission
controller described below retains evidence before optional cleanup. Native
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
the same surviving runtime does not establish fresh-runtime recovery. The
submission controller and optional direct Hub checkpoint transfer are implemented. PVC manifests can be resumed, but durable external backup and
Kubernetes lifecycle/evidence review remain required before 7C completion.

For a concrete generated deployment document and comparison with 7A/7B, see
[the manifest guide](MANIFEST_GUIDE.md) and [RayJob example](examples/7C_RAYJOB_EXAMPLE.yaml).

## Submit, monitor and retain one job

`job_control.py`, exposed by `part7C.sh job`, now handles one generated RayJob.
Its CPU check makes no Kubernetes API calls:

```bash
bash infrastructure/7c_kuberay/part7C.sh job --check \
  --manifest results/7c/preflight.json
```

Once cluster access, image, PVC/Secret, workload data and the hardware/budget are
ready, submit from a machine with configured `kubectl` access. All GPU execution
remains on the cloud cluster; the controller itself only runs Kubernetes commands:

```bash
bash infrastructure/7c_kuberay/part7C.sh job --cloud \
  --manifest results/7c/preflight.json \
  --context ACTUAL_KUBECTL_CONTEXT \
  --output results/7c/preflight_deployment \
  --cleanup-on-success
```

The context placeholder must identify the intended cluster. Use unique manifest
job names and fresh evidence directories. The controller accepts the exact JSON
output from the renderer, not edited manifests or the reading-example YAML.
It performs server-side dry-run validation, refuses an existing RayJob of that
name, creates the job and monitors its RayJob status. Polling has the configured
job deadline plus five minutes; Kubernetes commands have individual timeouts.

The submitted copy changes `shutdownAfterJobFinishes` to `false`, so pods stay
available during evidence collection. It records this actual submitted manifest,
job UID, Kubernetes status, pod placement/images/resources and available pod logs.
At success it retrieves the full Ray driver log using the head's local dashboard
and saves `workspace_evidence.tar.gz`. The archive includes run configuration,
identities, metrics/diagnostics, rollouts, audits and checkpoint manifests; tensor
files, checkpoint payloads, model exports and caches are excluded. No archive is
extracted automatically, and no Secret contents or pod environment values are
queried for evidence.

`--cleanup-on-success` deletes only the newly created RayJob after required logs,
workspace evidence and successful phase status are retained. It verifies that
its UID still matches and waits for the associated RayCluster and Ray pods to
vanish. The receipt preserves the PVC/Secret scope. This is lifecycle checking,
not a claim of checkpoint backup or a check that rented nodes have been released.

On failure, timeout or retention error, resources remain for inspection. Without
`--cleanup-on-success`, even a successful job leaves its cluster available; inspect
`submission.json` and use the recorded context/namespace/name for later cleanup.
Retained GPU pods continue occupying resources until removed. Cluster/node rental
cleanup is separate from RayJob deletion.

Run preflight, baseline, interruption and resume as separate jobs. For recovery,
clean up the interrupted job/cluster first, then render a new resume job pointing
to the checkpoint still on the same PVC. A new pod/runtime is intended, not a
verified replacement physical node. Direct Hub backup/restore is prepared as described below. Automatic bundle
staging, registry publication and an end-to-end multi-phase experiment wrapper
remain pending. Do not treat these prepared commands as cloud-verified.

Validation: 25 CPU tests pass, including controller success/failure, evidence-before-
cleanup, UID mismatch, lightweight archive exclusions and pinned schema checks.
No live Kubernetes API was contacted during development.

## Direct private Hub checkpoint retention and restore

`checkpoint_transfer.py` reuses 7B's checkpoint manifest/layout/hash verification,
with KubeRay-specific Secret authentication and persistent paths. It reads the
existing `token` Secret file, requires pod identity and private dataset repositories,
and never exports the token into runtime settings. Downloads are byte transfers,
not tensor deserialization. Training-state restoration is still audited by verl's
native resume and the 7B hooks afterward.

To retain the complete step-2 checkpoint, add `--upload-checkpoint-repo auto` when
rendering the interruption manifest:

```bash
bash infrastructure/7c_kuberay/part7C.sh render \
  --image REGISTRY/PART7_IMAGE@sha256:ACTUAL_64_HEX_DIGEST \
  --node-name ACTUAL_KUBERNETES_GPU_NODE --name part7-7c-interrupt-UNIQUE \
  --phase interrupt --upload-checkpoint-repo auto \
  --output results/7c/interrupt.json
```

The lowercase Kubernetes name must replace the `UNIQUE` placeholder. `auto` selects
`ACCOUNT/part7-7c-JOB_NAME`, a private dataset repository. An explicit repository ID
is also supported. Upload verification records the pinned commit in
`/workspace/runs/JOB_NAME/hf_checkpoint.json`. The controller includes that receipt
in its lightweight evidence archive. With upload selected, phase success is recorded
only after upload and remote inventory checks succeed; an upload failure blocks
success cleanup. Full state transfers cloud-to-Hub without passing through a laptop.

Render a separate resume RayJob using the recorded repository and full 40-hex commit:

```bash
bash infrastructure/7c_kuberay/part7C.sh render \
  --image REGISTRY/PART7_IMAGE@sha256:ACTUAL_64_HEX_DIGEST \
  --node-name ACTUAL_KUBERNETES_GPU_NODE --name part7-7c-resume-UNIQUE \
  --phase resume --checkpoint /workspace/restored/UNIQUE/global_step_2 \
  --restore-repo ACCOUNT/RECORDED_REPOSITORY --restore-revision FULL_40_HEX_COMMIT \
  --output results/7c/resume.json
```

Use a fresh destination under `/workspace`; existing directories are rejected.
The launcher downloads the selected commit, verifies every checkpoint file hash,
then composes native resume settings. A partial download remains for inspection;
choose another fresh destination after a failed attempt. No latest-branch fallback
or automatic overwrite occurs. Omit restore options to resume an already verified
PVC checkpoint. Upload can also be enabled for baseline/resume final checkpoints.

These options are available directly through `part7C.sh phase` as well. Transfer
helpers use the installed pinned Hub client and large-folder upload path already
used in reviewed 7B; no dependency upgrade is introduced. Public uploads remain
unsupported. Hub inventories alone do not establish training-state recovery.
Thirty-two 7C CPU tests pass, including mocked authentication/private upload,
pinned download, invalid revisions, fresh paths and renderer/controller integration.
Actual 7C Hub transfers, image and GPU Kubernetes execution remain unverified.

Reference: [Hub downloads](https://huggingface.co/docs/huggingface_hub/guides/download)
and [Hub uploads](https://huggingface.co/docs/huggingface_hub/guides/upload).
