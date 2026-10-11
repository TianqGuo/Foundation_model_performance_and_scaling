# 7C — Kubernetes training

## Status and acceptance

Implementation authorized on 2026-10-11. The first increment prepares the image,
RayJob renderer and cloud GPU preflight. Ten CPU tests pass, including validation
against the pinned KubeRay v1.4.2 RayJob CRD. No image has been built or published,
no Kubernetes cluster contacted, and no GPU preflight or training performed. The second increment adds native
existing-cluster training/interruption/resume configuration and launcher code.
7C remains incomplete; 7A and bounded 7B remain complete.

Accepted completion requires an immutable image and real GPU RayJob, explicit
resources/shared memory, durable storage and Secret injection, actual two-rank
FSDP2 training, interruption/recovery, retained placement/status/logs/metrics and
checkpoint evidence, and verified cleanup. Manifest validity alone is insufficient.
See [the runbook](7C_RUNBOOK.md) for prepared commands and access requirements.

## Prepared code

- `infrastructure/7c_kuberay/Dockerfile` builds an isolated Part 7 environment
  from the verified 7A lock and pinned verl source. The base must already contain
  uv, git and CUDA 13 development tools; it is supplied by immutable digest.
  A Dockerfile-specific allowlist excludes datasets, results, caches and keys.
- `render.py` produces a local RayJob JSON manifest without contacting Kubernetes.
  It requires a published image digest, an explicit GPU node, an existing PVC
  and a Secret name. A CPU head and one two-GPU worker share that node and PVC.
- `preflight.py` connects to KubeRay's existing Ray cluster. Two concurrent actors
  verify distinct Ray GPU assignments, CUDA visibility and a tiny CUDA computation
  on the same worker pod. It checks pinned imports and Secret authentication,
  and retains status, identities and placement on the PVC. It loads no model.
- `part7C.sh` exposes render/build actions. `build_image.sh` builds on cloud;
  publishing and deployment are not automated in this increment. The `phase`
  action invokes the existing-cluster launcher inside a prepared pod.

Candidate operator pin: KubeRay v1.4.2; Ray remains the verified 2.49.2.
The CRD was checked on CPU, but operator/Ray/image compatibility must still be
verified on the selected Kubernetes cluster. Images use UID/GID 1000 and mounted
Secret files, rather than credential values in generated manifests.

## Next implementation and verification

1. Confirm Kubernetes access, GPU device plugin, storage class, registry access,
   compatible base digest and resource budget before paid provisioning. A rented
   GPU Docker container is not automatically a usable Kubernetes GPU node.
2. Build/publish the isolated image, record its registry manifest digest and run
   the bounded GPU preflight. Confirm placement and durable evidence after exit.
3. Cloud-verify the prepared thin existing-cluster verl launcher: CPU-head driver, worker-side GPU/model
   checks, shared data/model paths and native training. Do not reuse 7B's local
   Ray startup or driver-side two-GPU guard unchanged; do not wrap verl in Ray Train.
4. Cloud-verify reused 7B integrity/restoration audits for checkpoint-boundary interruption and
   a fresh RayJob resume from pinned Hub state. Record changed pod/runtime IDs,
   same topology, restored state and subsequent updates. Do not claim vLLM server
   RNG recovery or bitwise rollout equivalence.
5. Retain lightweight evidence outside the cluster and full state on the Hub;
   inspect cleanup without deleting the PVC or Secret needed for recovery.

No new training backend, TP/PP, autoscaling or shared-cluster queue implementation
is introduced. Kueue remains an optional later admission-control follow-up.

## Primary references

- [Pinned KubeRay release](https://github.com/ray-project/kuberay/releases/tag/v1.4.2)
- [Pinned RayJob schema](https://github.com/ray-project/kuberay/blob/v1.4.2/ray-operator/config/crd/bases/ray.io_rayjobs.yaml)
- [Kubernetes GPU scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
- [KubeRay storage](https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/storage.html)

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

For a concrete generated deployment document and comparison with 7A/7B, see
[the manifest guide](MANIFEST_GUIDE.md) and [RayJob example](examples/7C_RAYJOB_EXAMPLE.yaml).
