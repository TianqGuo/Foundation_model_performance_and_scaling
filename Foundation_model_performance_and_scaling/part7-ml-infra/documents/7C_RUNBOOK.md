# 7C runbook — verified cloud workflow

7C passed its bounded fresh-install/training/recovery acceptance run. See
[7C_PLAN.md](7C_PLAN.md) for measured results, pinned identities and limits.
Historical implementation/failure notes remain in the root [PLAN.md](../PLAN.md).

## End-to-end workflow and code map

This is the main workflow reference for 7C. `e2e.py` runs on the cloud VM and
connects existing helpers; it does not replace their implementation. The published
training image is a prerequisite and is reused by this workflow.

```text
On the cloud VM (outside the training pods)
part7C.sh e2e
  -> e2e.py: verify sources; install/check Kubernetes, KubeRay and PVC
  -> prepare the accepted Part 5 data bundle on the VM
  -> experiment.py: coordinate the experiment
       -> data_staging.py: copy bundle to PVC through a temporary pod
       -> render.py: generate each RayJob manifest
       -> job_control.py: submit to Kubernetes and monitor/retain evidence
            -> KubeRay creates Ray head/worker pods from the training image
            -> preflight RayJob: preflight.py checks two GPU allocations
            -> baseline RayJob: run_7c.py launches native verl training
            -> interrupt RayJob: save step-2 checkpoint and upload to Hub
            -> resume RayJob: download pinned checkpoint and resume at step 3
       -> compare restored training state and record recovery review
  -> retain overall review and phase cleanup receipts
```

Each successful RayJob is cleaned up after its evidence is retained, before the
next phase needs the GPUs. A failure stops progress and retains resources for
inspection. The interruption is controlled at a complete checkpoint boundary;
this does not simulate a VM crash.

| Responsibility | Main code |
| --- | --- |
| Shell entry point and overall composition | [part7C.sh](../infrastructure/7c_kuberay/part7C.sh), [e2e.py](../infrastructure/7c_kuberay/e2e.py) |
| Kubernetes installation and read-only cluster checks | [setup_cluster.py](../infrastructure/7c_kuberay/setup_cluster.py), [cluster_check.py](../infrastructure/7c_kuberay/cluster_check.py) |
| Experiment sequence, state review and PVC staging | [experiment.py](../infrastructure/7c_kuberay/experiment.py), [data_staging.py](../infrastructure/7c_kuberay/data_staging.py) |
| RayJob manifest, submission, logs and cleanup | [render.py](../infrastructure/7c_kuberay/render.py), [job_control.py](../infrastructure/7c_kuberay/job_control.py) |
| Code executed inside RayJob pods | [preflight.py](../infrastructure/7c_kuberay/preflight.py), [run_7c.py](../infrastructure/7c_kuberay/run_7c.py), [checkpoint_transfer.py](../infrastructure/7c_kuberay/checkpoint_transfer.py) |

Kubernetes schedules the pods; KubeRay manages the Ray cluster/job lifecycle;
Ray schedules application actors/tasks, and verl runs the training workflow.
The controller submits manifests to Kubernetes rather than directly to KubeRay.
7C's bounded fresh-install/same-node recovery workflow has passed cloud review;
see [results and limits](7C_PLAN.md). For the serving counterpart, see the
[7D workflow](7D_RUNBOOK.md#end-to-end-workflow-and-code-map).

## One-command cloud execution

Use a full Ubuntu x86_64 GPU VM with systemd, passwordless sudo, existing NVIDIA
container runtime/driver and Docker. Initial setup requires exactly two GPUs,
32 CPUs, 120 GiB host RAM and 200 GiB free host disk. The verified VM had two H100
80 GB GPUs, 52 CPUs and about 442 GiB RAM. An ordinary restricted Docker rental
is not a Kubernetes host. Agree hardware/spending cap before paid provisioning.

Clone/pull the full project after user-approved source publication, then enter
Part 7. Git operations are never performed automatically. Rsync is an alternative
for uncommitted source; exclude credentials, `.git`, `.venv*`, caches and results.
The Git clone includes the Part 5 JSONL source data and reference code.

On a clean VM:

```bash
bash infrastructure/7c_kuberay/part7C.sh e2e --cloud --fresh-setup
```

On an already prepared healthy cluster:

```bash
bash infrastructure/7c_kuberay/part7C.sh e2e --cloud
```

Fresh setup reads `HF_TOKEN` if present, otherwise prompts invisibly for a Hub
write token. Existing setup reuses the Kubernetes Secret. Do not share tokens.
`--fresh-setup` refuses existing K3s; it does not uninstall or reset anything.
Normal mode installs only when K3s is absent; partial/unhealthy clusters stop for
inspection. The existing published image is reused: no Docker Hub token or image
rebuild is required for normal E2E runs.

Default image:

```text
sclion/part7-training@sha256:b1d8240b348d2a036d311f19fe75784825613607129c1d93526cfa7f530fb8e8
```

`--image`, `--part5-root`, `--data`, `--name` and `--output` allow overrides.
`e2e --check` validates references/image identity only; it is not a cloud test.

## Stages and expected behavior

1. Install/check K3s, NVIDIA plugin, KubeRay, Ready node, GPU capacity, API and PVC.
2. Generate/reuse the portable 1,023/32 data bundle from Part 5 in a no-network,
   no-GPU Docker container. Input/output hashes and selection settings are checked;
   the Part 5 mount is read-only. No laptop copy or host pyarrow install is needed.
3. Copy the verified bundle onto the shared PVC through a temporary CPU staging
   pod. Docker and K3s use separate image stores; initial pulls may happen twice.
4. Run preflight with two GPU actors: CUDA computation, distinct allocation,
   required imports and mounted-Secret Hub authentication.
5. Run three-step baseline, separate training interrupted after a complete step-2
   checkpoint, private Hub upload, exact-revision download and fresh-RayJob resume
   at step 3. Native state audits compare against the uninterrupted reference.
6. Retain logs/placement/configuration/status and review recovery. Clean successful
   phase RayJobs/clusters/pods only after evidence collection. Failures retain
   resources for inspection and stop the workflow.

The final message is `7C E2E completed; retain and review ...`. Its output contains
`e2e_review.json`, `cluster_check.json`, optional `setup/`, and `experiment/`.
Check `success`, `fresh_setup_verified`, `experiment/recovery_review.json`, and
per-phase `cleanup.json` before releasing cloud resources. `fresh_setup_verified`
is false for a cluster-reuse run, even if the training experiment succeeds.

The accepted local evidence is `results/7c/part7-e2e-20261018-031219` (UTC run ID).
The dependency-light reviewer was rerun locally and all four cleanup receipts
verified. Checkpoint tensors are not required on the laptop for this review.

## Storage, authentication and retention

Defaults: node `part7-gpu`, namespace `part7`, operator in `default`, PVC
`part7-workspace`, Secret `part7-huggingface` with key `token`. Pods use UID/GID
1000 and mount the Secret as a read-only file. The private user kubeconfig is
`$HOME/.kube/part7-config` (mode 0600), outside results. Never copy it into evidence.

The pre-bound 100 GiB RWO PV uses `/var/lib/part7-workspace`. Same-node head/worker
pods can share it; it is not a multi-node filesystem or enforced disk quota.
Retain policy protects against claim deletion, not VM destruction. Keep full
state on HF and copy lightweight controller evidence before deleting the VM.
Automatic external backup of the complete controller evidence is not implemented.

Accepted full checkpoint: `sclion/part7-7c-part7-e2e-20261018-031219-interrupt` at
`122e784fddc0a315a65e5ad9b2ed932dcc41b117` (private HF dataset). The checkpoint manifest is
uploaded with the state. `hf_checkpoint.json` records repository/revision on PVC
and in controller evidence; the recovery manifest pins those values. The receipt
itself is not included in that checkpoint upload. If the controller is lost,
select the repository/revision using retained evidence or HF history.

From local Part 7, an optional evidence copy is:

```bash
rsync -az --partial -e "ssh -i ~/.ssh/id_ed25519" \
  ubuntu@CLOUD_IP:~/Foundation_model_performance_and_scaling/Foundation_model_performance_and_scaling/part7-ml-infra/results/7c/RUN_NAME \
  ./results/7c/
```

Replace host/run placeholders; create local `results/7c/` first. Copy only the
controller run directory, not `/workspace` checkpoint/model directories.

## Image build/publication — only when the image changes

```bash
bash infrastructure/7c_kuberay/part7C.sh image --check --publish
bash infrastructure/7c_kuberay/part7C.sh image --cloud --publish
```

Default public destination: `sclion/part7-training`. Publication asks invisibly for
a Docker Hub write token (or reads `DOCKERHUB_TOKEN`); it uses stdin and isolated
temporary authentication. No manual login required. Sudo-created auth files are
removed using the matching privilege level. The allowed build context excludes
datasets, results, checkpoints, Git metadata and common credential files.

The default base is NVIDIA CUDA 13.0.3 development/Ubuntu 22.04; uv 0.12.19 comes
from its official binary image. Both tags resolve to recorded immutable digests.
Host CUDA 12.8 need not be replaced. The image supplies its own Python 3.12.14,
locked 7A stack, wget and a real UID/GID 1000 user. Before push, a no-network,
no-GPU check verifies tools, identity, writable HOME and PyTorch compiler-cache
import. Real GPU/import compatibility is tested in Kubernetes preflight.

BuildKit caches uv outside image layers, sets dependency ownership during
installation and uses COPY ownership flags. Source-only changes can reuse the
large dependency layer if base/environment inputs and local cache remain unchanged.
No cache survives a VM deletion unless separately exported. Do not prune it during
builds. The early 16.7 GB ownership-copy layer was removed; observed export-stage
time fell from 515.8 to 240.3 seconds across those two builds, not a controlled
benchmark. Later small-layer fixes exported in about 0.2 seconds with cache reuse.

Keep `image_build.json`, `image_publish.json` and logs. The publication receipt's
`image` field is the registry reference; a local Docker configuration ID is not
an interchangeable identity. Existing images stay usable by their old digests.

## Individual actions — diagnosis or targeted reruns

```bash
export KUBECONFIG="$HOME/.kube/part7-config"
bash infrastructure/7c_kuberay/part7C.sh cluster --context default --node-name part7-gpu
kubectl get rayjobs,pods -n part7
kubectl get events -n part7 --sort-by=.metadata.creationTimestamp
```

`setup --cloud --with-hf-secret` bootstraps a fresh VM without training. `render`
generates the workload RayJob; `job --check` validates it locally and `job --cloud`
submits it. `experiment --cloud` assumes Kubernetes and a prepared bundle already
exist; prefer `e2e` for normal use. Use each action's `--help` for exact options.
For isolated preflight, render phase `preflight`, submit with an explicit context,
fresh output and `--cleanup-on-success`; choose a unique job name for every retry.

## Troubleshooting

| Symptom | Cause/action |
| --- | --- |
| `kubectl` not found | Check whether setup ran on this VM. Image building does not install Kubernetes. |
| Node wait fails during startup | Current setup polls registration/API availability before waiting for Ready. Inspect `sudo journalctl -u k3s`; do not reinstall blindly. |
| Plugin DaemonSet desired=0 | Pinned chart needs `nvidia.com/gpu.present=true`; current setup labels the hardware-verified node before chart creation. |
| RayJob exists without pods, `InvalidRayJobSpec` | Current controller removes finished-job TTL when disabling automatic shutdown to retain evidence; it detects invalid-spec events promptly. |
| Probes fail with missing wget | Use the accepted image or rebuild current Dockerfile; do not disable probes. |
| `getpwuid(): uid not found: 1000` | Use the accepted image with a real passwd entry. Numeric USER alone is insufficient. |
| PermissionError cleaning temporary Docker auth | Current publisher uses privileged cleanup for its exact allocated directory when Docker used sudo. Do not print auth contents. |
| Docker ARG default warnings | Build supplies digest arguments; these warnings did not block accepted builds. |
| Long first image pull/export | Inspect pod events/disk activity before restarting. Preserve caches; Docker and K3s stores differ. |

`setup --continue-after-k3s ORIGINAL_FAILED_DIRECTORY` is limited to a verified
same-VM failure before storage/chart creation and can reuse only an exact private
kubeconfig. It is not a general recovery/retry tool or a substitute for clean E2E.
Uninstalling K3s deletes its local cluster/pod/storage state; review retention before
using the native uninstall script. No automatic uninstall is part of the runner.

## Verification limits and next milestone

Accepted 7C is same-node, controlled checkpoint-boundary recovery. It does not
restore vLLM server/driver RNG, cache or in-flight requests and makes no claim of
bitwise equivalence, unexpected crash recovery, replacement-node recovery or
multi-node scalability. CPU mocks are supplemental; the retained cloud run is
the acceptance evidence. Current local suite: 69 passing, one optional schema skip.

7D will separately choose its serving stack/model revision and implement
RayService deployment, request benchmarks and restart/readiness recovery. Do not
use the FSDP training-state dataset directly as a serving model export.
