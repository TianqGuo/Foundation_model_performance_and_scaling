# 7C runbook — preparation increment

The milestone runner can bootstrap a fresh Ubuntu GPU VM, build an image, and
stage/submit/review the bounded RayJob experiment. Existing clusters can skip the
bootstrap. [7C_PLAN.md](7C_PLAN.md) records pending live verification.

## Fresh GPU VM setup

The reviewed Lambda diagnostics showed a KVM Ubuntu 22.04 VM with systemd,
passwordless sudo, two H100 80 GB GPUs, 52 CPUs, 442 GiB RAM and NVIDIA container
tooling. This is preliminary host suitability evidence, not a Kubernetes test;
the instance was destroyed before installation. Ordinary Vast Docker rentals
failed this prerequisite because they lacked host permissions.

Deliver source using Git clone/pull after user-approved commits/pushes. For
uncommitted source, rsync the Part 7 directory (excluding `.git`, `.venv*`, caches,
credentials and `results/`) to the VM. The prepared data bundle is transferred
separately before the experiment, as described below. No Git actions are automatic.

From the Part 7 root on a **fresh, dedicated Ubuntu x86_64 GPU VM**:

```bash
# Preview objects locally or on cloud; performs no installation or host inspection.
bash infrastructure/7c_kuberay/part7C.sh setup --check

# Cloud VM only: installs Kubernetes and reads the Hub write token without echoing it.
bash infrastructure/7c_kuberay/part7C.sh setup --cloud --with-hf-secret
export KUBECONFIG="$HOME/.kube/part7-config"

bash infrastructure/7c_kuberay/part7C.sh cluster \
  --context default --node-name part7-gpu
```

`setup_cluster.py` refuses containers, existing clusters/storage and insufficient
hardware before installing. It requires exactly two GPUs, at least 32 CPUs,
120 GiB host RAM and 200 GiB free disk, plus existing NVIDIA driver/runtime,
passwordless sudo and curl. It leaves the existing host CUDA toolkit and Docker
configuration untouched. Docker remains useful for the separate image build.

It installs candidate pins K3s `v1.32.13+k3s1`, NVIDIA device plugin `0.17.1` and
KubeRay `1.4.2`; the bundled K3s Helm controller installs the charts. The K3s
installer is downloaded from its official URL, retained and hashed in the setup
receipt; its script itself is not immutable. K3s uses its own containerd with the
NVIDIA default runtime. Setup waits for a Ready node, two advertised GPU resources,
the operator, established RayJob API and Bound PVC, then applies our cluster checks.
Actual GPU assignment/computation is verified separately by the image preflight.

The created resources match runner defaults: node `part7-gpu`, namespace `part7`,
PVC `part7-workspace`, and optional Secret `part7-huggingface` with key `token`.
The operator runs in `default`. `--with-hf-secret` uses `HF_TOKEN` if already set,
otherwise a hidden prompt; no CLI login is required. Without that flag, provide
an existing Secret before the GPU preflight. Tokens and the private mode-0600
kubeconfig never enter the retained result files.

The static 100 GiB RWO PV is pre-bound to this PVC, uses
`/var/lib/part7-workspace`, and permits same-node head/worker sharing. The capacity
is an advertised scheduling value, not a host filesystem quota. Its Retain policy
preserves data on claim deletion; **destroying the VM destroys this storage**.
Upload complete checkpoints to the Hub and retain lightweight evidence before
releasing the VM. This is not a multi-node storage solution.

Receipts and credential-free setup objects are under `results/7c/setup_*`.
A partial installation is left for diagnosis; this helper refuses a general rerun over an existing cluster. A verified early
kubeconfig-write failure can use the continuation command below. Inspect `sudo journalctl -u k3s` and
`kubectl get pods -A` on failure; do not uninstall or delete storage before retention.
Image building/publication, the GPU preflight and recovery experiment remain
separate steps below. No live bootstrap has been verified yet.

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

For the eventual cloud image build, select a compatible custom base image reference. The build helper pulls it and
resolves its registry digest automatically. It must provide uv, git and CUDA 13 development tools:

```bash
bash infrastructure/7c_kuberay/part7C.sh build --cloud \
  REGISTRY/BASE_IMAGE@sha256:ACTUAL_64_HEX_DIGEST part7-training:7c
```

The helper uses Docker directly or passwordless sudo on a Lambda VM, and retains
base-pull/build logs plus image/base identities under `results/7c/image_build_*`.
A tag can be supplied instead of the base digest; the Dockerfile always receives
the resolved digest. Inherited base-image entrypoints are cleared so KubeRay can
start its own commands.

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
verified replacement physical node. Direct Hub backup/restore is prepared as described below. Portable bundle staging and an end-to-end multi-phase experiment wrapper are
now prepared as described below; registry publication remains external. Do not treat these prepared commands as cloud-verified.

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

## End-to-end bounded experiment

`part7C.sh experiment` now coordinates portable data staging, preflight, baseline,
controlled interruption with direct Hub upload, a fresh pinned Hub download/resume,
evidence review and successful-job cleanup. The image must already be built and
published; Kubernetes, KubeRay, GPU nodes, namespace, PVC and Hub Secret must
already be available. This workflow is CPU-tested, not cloud-verified.

From the Part 7 root, check the existing prepared data bundle and initial manifest
without Kubernetes contacts (replace the placeholders with actual values):

```bash
bash infrastructure/7c_kuberay/part7C.sh experiment --check \
  --image REGISTRY/PART7_IMAGE@sha256:ACTUAL_64_HEX_DIGEST \
  --node-name ACTUAL_KUBERNETES_GPU_NODE
```

After hardware/budget and cluster prerequisites are confirmed, one controller
command runs the sequence:

```bash
bash infrastructure/7c_kuberay/part7C.sh experiment --cloud \
  --image REGISTRY/PART7_IMAGE@sha256:ACTUAL_64_HEX_DIGEST \
  --node-name ACTUAL_KUBERNETES_GPU_NODE --context ACTUAL_KUBECTL_CONTEXT
```

Optional arguments are `--data PATH`, `--namespace NAME`, `--pvc NAME`,
`--secret NAME`, `--pull-secret NAME`, `--name LOWERCASE_UNIQUE_RUN_NAME` and
`--output FRESH_LOCAL_DIRECTORY`. Defaults discover the existing local
`results/7a/step2_math_final` bundle and create a UTC-named directory under
`results/7c`. No manual data copy or phase-by-phase command sequence is required.
If the prepared bundle is absent, prepare it with the existing 7A adapter first;
the experiment does not recreate historical dataset selections or change grading.

The sequence is:

1. Package only manifest-recorded data/prompt/grader files after local hash checks.
   A temporary CPU pod on the selected node mounts the existing PVC and verifies
   the copied bundle at `/workspace/data/step2_math_final`. Matching existing data
   can be reused; mismatched data is never overwritten. The staging pod has no
   GPU or Hub Secret and is removed after its UID is checked.
2. Run GPU preflight as a RayJob, retain evidence and verify cleanup.
3. Run a three-step baseline RayJob, retain evidence and verify cleanup.
4. Run an interruption RayJob through completed step 2. Upload full state directly
   to a run-specific private Hub dataset repository, retain the commit receipt,
   then retain evidence and verify cleanup.
5. Start a separate resume RayJob. Download that exact commit into
   `/workspace/restored/RUN_NAME/global_step_2`, verify hashes, restore/audit state
   and finish step 3. Retain evidence and verify cleanup.
6. Reuse the 7B restoration review and require distinct worker pod UIDs/Ray node
   IDs for all training phases, the same physical GPU node/image digest, matching
   Hub transfer receipts, native fresh-path resume, verified cleanup, actual
   two-rank sharding and 6/4/2 optimizer calls per rank. Write `recovery_review.json`
   and per-phase controller timings.

The local evidence extraction accepts only regular files, rejects tensors/path
escapes/links, and limits expanded data to 256 MiB. It loads no checkpoint tensors.
Full state stays on the PVC and Hub, rather than being downloaded to the controller.
The experiment stops on any failed phase/retention/review; failed RayJobs remain
for inspection. It does not retry automatically or restart crashed containers.
A failed experiment needs diagnosis and a new unique run name/fresh output path.
Partial incoming/downloaded data is preserved; PVC/Secret deletion and rented-node
release are separate decisions after evidence review.

Validation: 40 CPU tests pass, including synthetic data packaging, mocked staging,
four-phase ordering, pinned fresh restore, safe evidence extraction and review
requirements. No real cluster, Hub transfer, Docker build, model or GPU was used.
The staging image requires `tar` for `kubectl cp`; the Dockerfile now checks it.
[Official copy command reference](https://kubernetes.io/docs/reference/kubectl/generated/kubectl_cp/).

## Read-only cluster prerequisite inspection

Before the experiment, an existing cluster can be checked without creating jobs:

```bash
bash infrastructure/7c_kuberay/part7C.sh cluster \
  --context ACTUAL_KUBECTL_CONTEXT --node-name ACTUAL_KUBERNETES_GPU_NODE \
  --operator-namespace ACTUAL_OPERATOR_NAMESPACE \
  --output results/7c/cluster_prerequisites.json
```

`--operator-name` defaults to `kuberay-operator`; namespace defaults to `default`.
Override these for the actual installation. Workload namespace/PVC defaults match
other 7C commands. This invokes only metadata `kubectl get` operations and does
not query Secrets, create resources, load models or execute GPUs.

The check requires a ready/schedulable node whose hostname label matches our
current selector, at least 26 allocatable CPUs/104 GiB RAM/two advertised GPUs,
a Bound PVC with at least 100 GiB capacity and RWO/RWX access, an available
candidate v1.4.2 operator and an established served RayJob v1 API. RWOP is rejected
because it cannot support our simultaneous head/worker mounts. Current pod templates
have no explicit node-taint tolerations; restrictive node taints require a reviewed
placement/toleration change before execution. Capacity is not free capacity, actual
GPU memory, PVC free disk, registry access or a storage/network performance test.
The live staging/GPU preflight and experiment remain necessary. This helper does
not install missing prerequisites or prove image/operator compatibility by itself.

Validation: 46 7C CPU tests pass, including fabricated capacity/storage/operator/API
cases and proof that inspection uses only read operations. No real cluster was
contacted; cluster access, hardware budget and image build/publication are pending.

## Early kubeconfig-write failure (2026-10-13)

The first user-run installation reached the private kubeconfig write, then failed
because `Path.open()` does not support `opener`. The corrected helper uses builtin
`open()` and CPU-tests actual mode-0600/exclusive creation. K3s installation may
already have completed; do not rerun fresh setup or uninstall it.

After delivering the corrected `setup_cluster.py` to the same VM, identify the
original failed directory with `ls -d results/7c/setup_*`, then run:

```bash
bash infrastructure/7c_kuberay/part7C.sh setup --cloud --with-hf-secret \
  --continue-after-k3s results/7c/setup_ORIGINAL_TIMESTAMP
```

Replace the placeholder with the original directory. Continuation verifies its
failed receipt, K3s pin, original GPU identities and single `part7-gpu` node.
It requires that kubeconfig, storage and later setup objects do not yet exist,
skips K3s reinstallation, and saves a new receipt. Other partial failures need
inspection; this is not a general retry mechanism. Do not print kubeconfig or
Secret contents when sharing diagnostics. Cloud continuation is pending review.

## NVIDIA plugin DaemonSet desired=0

The pinned chart requires an NVIDIA feature label or `nvidia.com/gpu.present=true`.
This setup checks physical GPUs and now adds the latter label before installing
charts; it does not install Node Feature Discovery. For an already running setup
with Ready node/operator but no GPU-plugin pod, repair the verified GPU node:

```bash
sudo k3s kubectl label node part7-gpu nvidia.com/gpu.present=true --overwrite
sudo k3s kubectl rollout status daemonset/part7-nvidia-nvidia-device-plugin \
  -n kube-system --timeout=180s
sudo k3s kubectl get node part7-gpu \
  -o jsonpath='{.status.allocatable.nvidia\.com/gpu}{"\n"}'
```

Expected allocatable GPU count is `2`. MPS daemon desired=0 is expected; this
workload does not use MPS. If the setup process is still waiting, let it finish.
If it has timed out, retain its receipt and inspect before another setup invocation;
the early-kubeconfig continuation does not support this later failure boundary.
Reference: [pinned chart affinity](https://github.com/NVIDIA/k8s-device-plugin/blob/v0.17.1/deployments/helm/nvidia-device-plugin/values.yaml).


Reviewed Lambda setup receipt: `setup_20261014T040415Z` reports success after the
manual GPU label repair, with two advertised H100 GPUs, Bound PVC, pinned operator/
K3s and created Hub Secret. This establishes bootstrap/API/storage readiness, not
GPU kernel execution or token validity. New VMs must receive both source fixes.
The training image still needs a compatible base reference and registry destination;
registry publication and actual image preflight have not been performed.

## Docker Hub build and publication

User-selected public destination: `sclion/part7-training`. Deliver the latest source
first; a Git clone does not include agent edits until user-approved commits/pushes.
For uncommitted source, rsync `infrastructure/` and `workloads/` to their matching
Part 7 paths on the VM, excluding caches, `.venv*`, `.env*`, `*.pem` and `*.key`.

On the cloud VM, from Part 7:

```bash
bash infrastructure/7c_kuberay/part7C.sh image --check --publish
bash infrastructure/7c_kuberay/part7C.sh image --cloud --publish
```

The second command requests a Docker Hub personal access token with write
permission at a hidden prompt (or reads `DOCKERHUB_TOKEN` already in the environment).
Create it in Docker account settings → Personal access tokens. Do not paste tokens
into chat or source. Docker authentication uses stdin and a temporary configuration
outside results; it is removed afterwards. No manual Docker login is required.

Default base: `nvidia/cuda:13.0.3-devel-ubuntu22.04`; `uv` is copied from its official
`0.12.19` binary image. Both are pulled/resolved to immutable digests for the build;
git, certificates and tar are installed inside the training image. Existing host
CUDA 12.8 and Parts 1–6 environments are untouched. These candidates remain
cloud-build/import unverified. `--base` can override the base when compatibility
is reviewed. `--repository` overrides the destination.

The command reuses the build helper, runs only a no-network/no-GPU Python-version
startup check, then publishes a timestamped tag and verifies a pull of the
reported registry digest. Public upload exposes source/dependencies; the allowed
context excludes data/results/checkpoints/common credential files. Use the reviewed
official base; arbitrary custom bases require their own content review.
No model loads, GPU preflight or training occur in this action.

Retain `results/7c/image_*/image_build.json`, `image_publish.json` and build/push logs.
The publication receipt's `image` field is the reference to pass to `render` or
`experiment`; local Docker IDs are not accepted as registry identities. Failures
leave build logs/local image for diagnosis; never claim publication or GPU
compatibility solely from a successful build. Registry upload is opt-in through
`--publish`; no Git staging, commit or push is performed.

References: [uv Docker integration](https://docs.astral.sh/uv/guides/integration/docker/),
[Docker login](https://docs.docker.com/reference/cli/docker/login),
[Docker push](https://docs.docker.com/reference/cli/docker/image/push).

## Publication succeeded but temporary-auth cleanup failed

A sudo Docker build can create root-owned `buildx` directories under its temporary
configuration. The old Python TemporaryDirectory cleanup could not remove those
files, raising PermissionError after successful publication. The controller now
uses privileged cleanup of only its newly allocated temporary directory. This
controller fix does not require rebuilding the already published training image.

For the reviewed failed cleanup, remove only the exact path printed in that error:

```bash
sudo rm -rf -- /tmp/part7-docker-auth-ei1zerx0
```

Do not use a wildcard or remove your persistent Docker configuration. The image
receipt was saved before cleanup failed; retain `image_build.json`,
`image_publish.json` and logs. The two InvalidDefaultArgInFrom warnings are static
checks of empty Dockerfile ARG defaults; this build supplied valid digest arguments
and completed, so those warnings did not cause the cleanup error. GPU preflight
is still required before training.
