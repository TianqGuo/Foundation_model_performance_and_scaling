# 7C — Kubernetes training

Updated: 2026-10-12. Implementation authorized on 2026-10-11.

## Current status

The source workflow is prepared: isolated image definition, RayJob rendering,
GPU preflight, existing-cluster verl phases, private Hub checkpoint transfer,
portable data staging and a single multi-phase experiment command. **Fifty-one CPU
tests pass** (including the pinned RayJob schema and mocked cloud operations).
No Docker image has been built/published, no Kubernetes cluster contacted, no
real Hub transfer performed, and no model/GPU execution run for 7C. **7C is not
complete.** Reviewed 7A and bounded same-host 7B remain complete.

Accepted completion requires an immutable image and real GPU RayJob, explicit
resources/shared memory, durable storage and Secret injection, actual two-rank
FSDP2 training, interruption/recovery, retained placement/status/logs/metrics and
checkpoint evidence, and verified cleanup. CPU configuration checks are preparation.
See [the runbook](7C_RUNBOOK.md), [manifest guide](MANIFEST_GUIDE.md) and
[generated reading example](examples/7C_RAYJOB_EXAMPLE.yaml).

## Prepared implementation

All milestone files live under `infrastructure/7c_kuberay/`.

| File | Responsibility |
| --- | --- |
| `Dockerfile` / `Dockerfile.dockerignore` | Isolated 7A lock/pinned verl source; immutable compatible CUDA 13 base; source allowlist excluding datasets/results/credentials. |
| `setup_cluster.py` | Bootstrap a checked fresh Ubuntu GPU VM with pinned K3s/device-plugin/KubeRay, same-node storage, private kubeconfig and optional Hub Secret. |
| `build_image.sh` | Cloud image build, with no model execution or registry push. |
| `render.py` | Generate digest-pinned RayJob JSON for preflight/baseline/interruption/resume; Kubernetes pod and Ray settings, optional Hub transfer arguments. |
| `preflight.py` | Check concurrent distinct GPU assignments, CUDA computation, imports and mounted Secret authentication; no model loaded. |
| `run_7c.py` | Compose the existing 7A workload, attach to KubeRay's Ray runtime, prepare models on a GPU worker, run native verl and reuse 7B audits. |
| `checkpoint_transfer.py` | Complete private Hub checkpoint upload and pinned fresh-path download, using mounted Secret and 7B byte/hash verification. |
| `job_control.py` | Submit/monitor one exact generated manifest, retain pod/driver/PVC evidence, optionally verify owned-job cleanup. |
| `cluster_check.py` | Read-only node capacity/placement, PVC access/capacity and operator/API prerequisites; no job creation or Secret reads. |
| `data_staging.py` | Package only verified manifest-listed files and stage them onto PVC through a bounded CPU pod; no GPU/Secret needed. |
| `experiment.py` / `part7C.sh` | Coordinate staging → preflight → baseline → interruption/upload → fresh Hub restore/resume → evidence review and per-job cleanup. |

The initial topology is one CPU head plus one two-GPU worker on an explicitly
selected node, sharing an existing durable PVC. Pods use UID/GID 1000; the Hub
Secret is mounted as a file, never placed in manifests/runtime environment as a
credential value. Candidate KubeRay pin is v1.4.2; Ray remains 2.49.2. The CRD is
CPU-validated, but actual operator/Ray/image compatibility remains unverified.

The TaskRunner is placed on the GPU worker without reserving a GPU; preparation
reserves/releases both GPUs before training. Ray Train does not wrap verl. Native
7B audits verify model/Adam/scheduler/saved-worker RNG and trainer step/data before
continuation. The experiment additionally requires distinct worker pods/Ray IDs,
same physical node/image, matching pinned transfer receipts, fresh native resume
path, verified cleanup and 6/4/2 sharded optimizer calls per rank.

## Remaining decisions and verification

1. Confirm Kubernetes access, GPU plugin, storage class/PVC permissions/capacity,
   registry access, base image digest, hardware and budget before provisioning.
   An ordinary rented GPU container is not automatically a Kubernetes GPU node.
2. Build/publish the isolated image and record its actual registry digest. Preserve
   Parts 1–6 images/environments. Registry publication is not automated here.
3. Run and retain real staging/preflight evidence, then the bounded experiment.
   Review training ranks/sharding, pinned Hub download, state audits, fresh pod/Ray
   identities, evaluation/checkpoint events and timings. Record any image/API issues.
4. Confirm durable full checkpoint retention and lightweight external evidence,
   verify RayJob/RayCluster/pod cleanup, then separately decide rented-node release.

Successful controller cleanup deletes only its newly created RayJob after evidence
retention, verifies RayCluster/Ray-pod absence and preserves PVC/Secrets/nodes.
Failures retain Ray resources for diagnosis. Temporary staging pod cleanup verifies
its UID. Full state stays cloud/PVC/Hub; local extraction rejects tensors/links/path
escapes and bounds evidence size. Failed partial data/downloads are not overwritten.

No automatic crash/reboot restart, replacement-node recovery, vLLM server RNG/cache/
in-flight restoration, bitwise rollout equivalence, autoscaling or TP/PP claim is
introduced. Kueue remains an optional later admission-control follow-up. No commits
or pushes without approval. Source preparation does not authorize paid provisioning.

## Primary references

- [Pinned KubeRay release](https://github.com/ray-project/kuberay/releases/tag/v1.4.2)
- [Pinned RayJob schema](https://github.com/ray-project/kuberay/blob/v1.4.2/ray-operator/config/crd/bases/ray.io_rayjobs.yaml)
- [Kubernetes GPU scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
- [KubeRay storage](https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/storage.html)
- [Hub downloads](https://huggingface.co/docs/huggingface_hub/guides/download)

CPU preparation follow-up: added `part7C.sh cluster` to inspect prerequisites in an
explicit Kubernetes context. It rejects single-pod-only PVC access and incompatible
initial placement/capacity/operator/API conditions before expensive work. Forty-six
CPU tests pass; all live verification and hardware/budget decisions remain pending.

## Fresh-VM bootstrap increment

User-authorized fresh-VM setup is now implemented through `part7C.sh setup`.
The destroyed Lambda VM's pasted diagnostics establish preliminary host suitability
only. Fifty-one CPU tests pass, including no-mutation preview/host-rejection,
credential-error redaction, same-node Retain storage and CPU-head GPU hiding.
The helper does not rent resources or overwrite existing clusters. Live K3s,
containerd/NVIDIA, charts, storage and Secret behavior remain unverified. The host
CUDA 12.8 toolkit is not replaced; the training image still supplies CUDA 13.x.
Next: review image prerequisites/destination, then cloud bootstrap/build/preflight
and the recovery experiment within agreed hardware/budget. 7C remains incomplete.


2026-10-13 cloud bootstrap correction: the user-run Lambda setup reached the
kubeconfig write and exposed invalid `Path.open(opener=...)`. Corrected to builtin
exclusive mode-0600 open; added a narrowly validated early-failure continuation
that skips K3s reinstallation. Fifty-three CPU tests pass, including actual private
file creation and continuation guards. Cloud completion remains pending; no
installation success, GPU pod or 7C completion claim is made.


NVIDIA placement correction: user diagnostics show Ready K3s and Running KubeRay,
but plugin DaemonSet desired=0 and no allocatable GPUs. The pinned NVIDIA chart
requires a GPU-present/NFD label, while this bootstrap does not install NFD.
Setup now labels the hardware-verified node `nvidia.com/gpu.present=true` before
chart creation. Fifty-four CPU tests pass. Live GPU plugin readiness remains
pending; current instance can be repaired by that node-label command without
reinstallation. MPS desired=0 is expected with no MPS-capable label.


Image-build preparation follow-up: reviewed pasted successful Lambda bootstrap
receipt `setup_20261014T040415Z` (two H100 GPUs advertised, Bound PVC, operator/API,
Hub Secret created). This is not GPU kernel or token-authentication evidence.
The build helper now resolves a supplied base tag to an immutable digest, supports
sudo Docker, retains logs/identity hashes and clears inherited image boot hooks.
No image was built/published; registry/base selection and live preflight are pending.
No commits/pushes or changes to Parts 1–6.


Docker Hub publication preparation: user selected public `sclion/part7-training`.
Added `part7C.sh image --check|--cloud [--publish]` using candidate official NVIDIA
CUDA 13.0.3 Ubuntu 22.04 base and uv 0.12.19, resolved to recorded digests. Build
installs its own utilities, preserves host environments, tests Python startup with
GPU visibility disabled, and optionally publishes via hidden-token temporary
Docker authentication. Registry digest receipts are verified by pull; no Docker,
Hub publication or GPU execution performed locally. Fifty-seven CPU tests pass;
cloud build and GPU preflight remain pending. No Git commits/pushes.


Reviewed user image-build/publication output: Python 3.12.14 startup succeeded and
`sclion/part7-training@sha256:211e567f0bfd55617db36102b6e00891e6a136420850788c2874045dffcb3294`
was pushed and pulled by digest. Post-publication cleanup failed because sudo Docker
created root-owned buildx files in the temporary auth directory. Fixed cleanup to
use the same privilege level, scoped only to its newly allocated temporary path.
Fifty-eight CPU tests pass, including cleanup after failures and sudo scoping;
one optional pinned-schema test is skipped because its temporary jsonschema/CRD
fixtures are unavailable in the current local environment (previously verified).
Actual corrected cleanup and GPU/import preflight are still pending cloud evidence;
publication success does not imply GPU compatibility. No rebuild is needed solely
for this controller-side cleanup correction. No Git commits/pushes.


Layer optimization after reviewed cloud history: final recursive ownership change
added 16.7 GB above an 8.49 GB installation layer; source COPY layers were tiny.
Moved ownership changes into the dependency installation RUN and source COPY
ownership flags; uv cache now uses a BuildKit mount with copy link mode. This
prevents the separate ownership-copy layer and excludes package cache from images.
No package/base/model/algorithm changes. CPU suite remains 58 passing/one optional
schema skip; Docker/GPU execution was not run locally. Optimized cloud size/time
and GPU compatibility remain pending; one dependency rebuild is required before
future source-only cache reuse. Existing published digest preserved.


Node-readiness correction: user logs show successful K3s startup and a Ready
`part7-gpu` after the initial named-node wait failed; exact original stderr was
not retained, so a registration race is suspected, not proved. Setup now polls
registration/transient API availability before waiting for Ready, with progress
messages. `--continue-after-k3s ORIGINAL_FAILED_DIRECTORY` also permits reusing an
exact, user-owned mode-0600 kubeconfig before storage/chart creation, without
reinstallation. Existing symlinks/different configs are rejected. Sixty CPU tests
pass; one optional schema test skipped. Cloud continuation still needs review.


Clean-retest requirement: user correctly identified missing post-fix fresh-install
E2E evidence. Added full install control-flow CPU scenarios (external host/API
commands simulated; actual private config/receipt writes and validators execute).
Sixty-two CPU tests pass with one optional schema skip. These are not cloud E2E.
Runbook now describes guarded uninstall of the dedicated pre-storage failed K3s
cluster and the normal fresh setup command, preserving repository/build evidence
and Docker image. User cloud clean installation plus GPU preflight must be
reviewed before declaring the path verified. No local K3s uninstall/install,
cloud mutation, provisioning, commits or pushes performed.


First preflight submission was rejected before cluster creation: the controller
held the cluster with shutdownAfterJobFinishes=false but retained the renderer's
finished-job TTL, an invalid KubeRay combination. Controller now removes TTL when
holding pods for evidence; it also prints status changes and stops promptly on
UID-scoped InvalidRayJobSpec events. Sixty-three CPU tests pass; one optional
schema check skipped. Cloud submission/preflight still pending; no pods/training
were started by the rejected job. This controller-only fix needs no image rebuild.


GPU preflight reached pods after ~81-second image pulls, but KubeRay health probes
failed with `wget: command not found`, causing head/worker restarts. Added wget
in a small Dockerfile layer after dependency installation (preserves the large
cached environment). Image publication now requires a nonroot no-network/no-GPU
check of bash/wget/tar/git/ray/python before push. Sixty-four CPU tests pass and
one optional schema test skips; fixed image build/pod readiness remain pending.
Old failed RayJob resources must be retained/released before fresh two-GPU retry.
Kubernetes reinstall is unnecessary. No local Docker/model/GPU execution or Git
commits/pushes performed.


Preflight with wget-fixed image reached Ray RUNNING and GPUProbe workers but failed
inside vLLM/PyTorch import: UID 1000 had no passwd entry, so getpass.getuser raised
KeyError in compiler-cache initialization. Added a real UID/GID 1000 account via
small late image layer. The no-GPU image check now resolves passwd/getpass, checks
writable HOME and imports torch._inductor.codecache before publication. Sixty-four
CPU tests pass, one optional schema check skips; image check itself must execute
on cloud (local tests inspect invocation only). No inference/model training or
GPU success claim. Cloud rebuilt-image/preflight evidence remains pending.


One-command cloud E2E runner added after user feedback. `part7C.sh e2e --cloud`
composes native fresh setup or existing-cluster validation, automatic cloud-side
data preparation, staging and the complete preflight/training/Hub recovery audit.
`--fresh-setup` rejects existing K3s to distinguish genuine fresh-install evidence.
Confirmed Part 5 train/validation are Git-tracked; local rsync is not required.
Data is prepared with the existing published image, original five input hashes,
1023/32 selection, readonly Part 5 mount and no container GPU/network. Controller
needs no image rebuild. Sixty-nine CPU tests pass; one optional schema skips.
User pasted clean setup and later GPU preflight/controller success were reviewed;
full wrapper cloud execution/training-recovery remains pending. Local orchestration
tests simulate external systems and are not cloud E2E. No provisioning, Git commit/
push, host dependency changes or edits to Parts 1–6.
