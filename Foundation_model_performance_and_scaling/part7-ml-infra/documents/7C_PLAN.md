# 7C — Kubernetes training

Updated: 2026-10-12. Implementation authorized on 2026-10-11.

## Current status

The source workflow is prepared: isolated image definition, RayJob rendering,
GPU preflight, existing-cluster verl phases, private Hub checkpoint transfer,
portable data staging and a single multi-phase experiment command. **Forty CPU
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
| `build_image.sh` | Cloud image build, with no model execution or registry push. |
| `render.py` | Generate digest-pinned RayJob JSON for preflight/baseline/interruption/resume; Kubernetes pod and Ray settings, optional Hub transfer arguments. |
| `preflight.py` | Check concurrent distinct GPU assignments, CUDA computation, imports and mounted Secret authentication; no model loaded. |
| `run_7c.py` | Compose the existing 7A workload, attach to KubeRay's Ray runtime, prepare models on a GPU worker, run native verl and reuse 7B audits. |
| `checkpoint_transfer.py` | Complete private Hub checkpoint upload and pinned fresh-path download, using mounted Secret and 7B byte/hash verification. |
| `job_control.py` | Submit/monitor one exact generated manifest, retain pod/driver/PVC evidence, optionally verify owned-job cleanup. |
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
