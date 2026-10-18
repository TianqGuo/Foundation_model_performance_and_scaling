# Reading the Part 7 configuration and manifests

Start with [the generated 7C RayJob example](examples/7C_RAYJOB_EXAMPLE.yaml).
It is a full renderer output for the baseline training phase, formatted as YAML
for reading. The image digest and node are placeholders: do not deploy this example.
The renderer's normal JSON output represents the same Kubernetes object.

## Concrete files across 7A–7C

| Stage | File to open | Purpose and consumer |
| --- | --- | --- |
| 7A | [Smoke configuration source](../infrastructure/7a_verl/config/grpo_smoke.yaml) | Training/rollout/algorithm settings composed by our launcher for verl. |
| 7A | [Actual resolved cloud configuration](../results/7a/smoke_execution_20261005T045405Z_dwaxLH/resolved_config.yaml) | Complete configuration recorded during the reviewed cloud run; paths describe that historical instance. |
| 7A | [Actual data bundle manifest](../results/7a/step2_math_final/manifest.json) | Provenance and hashes for prepared data/prompt/reward files, checked by our adapters/launchers. |
| 7B | [Actual resolved resume configuration](../results/7b/recovery_execution_20261011T022932Z_Ponp8Z/resume/resolved_config.yaml) | Native resume settings and training configuration recorded in the reviewed recovery run. |
| 7B | [Actual Hub checkpoint receipt](../results/7b/recovery_execution_20261011T022932Z_Ponp8Z/hf_checkpoint.json) | Repository, pinned revision and transfer information; a receipt, not the checkpoint manifest itself. |
| 7C | [Generated RayJob example](examples/7C_RAYJOB_EXAMPLE.yaml) | Kubernetes deployment configuration interpreted by the Kubernetes API and KubeRay operator. |

The results links require locally retained ignored evidence; they are not published
source examples. The full 7B checkpoint manifest (`checkpoint_manifest.json`) lives
with the checkpoint payload retained on the Hub. Full checkpoint directories were
excluded from the lightweight local copy; no checkpoint download is needed merely
to understand the distinction. See [7B_PLAN.md](7B_PLAN.md).

## What to inspect in the RayJob example

- `apiVersion: ray.io/v1` and `kind: RayJob`: identify the custom Kubernetes
  resource registered by KubeRay. Submit it to the Kubernetes API.
- `metadata`: job name, namespace and labels.
- `spec.entrypoint`: runs `run_7c.py --cloud --phase baseline` inside the cluster.
- `spec.rayClusterSpec.headGroupSpec.template.spec`: Kubernetes settings for
  the head pod, including the image, CPU/memory, node selection and mounts.
- `spec.rayClusterSpec.workerGroupSpecs[0].template.spec`: the worker pod's
  Kubernetes settings. Its container requests/limits include two NVIDIA GPUs,
  24 CPUs and 96 GiB RAM.
- `volumeMounts` and `volumes`: persistent workspace, memory-backed `/dev/shm`,
  and read-only Secret-file injection. PVC/Secret names reference existing objects;
  their contents and creation specifications are not included in this RayJob.
- `rayStartParams`: Ray process settings, such as logical GPU count and object
  store memory. These supplement Kubernetes resource requests; they do not
  allocate physical Kubernetes GPUs by themselves.
- `activeDeadlineSeconds`, `shutdownAfterJobFinishes` and `ttlSecondsAfterFinished`:
  requested job lifecycle behavior. The controller holds pods for evidence by
  disabling automatic shutdown and removing TTL; explicit cleanup was verified
  in the accepted run.

## Which Kubernetes configuration already exists?

**Workload-level Kubernetes configuration is implemented in the RayJob's pod
templates.** KubeRay creates pods from these templates, and Kubernetes schedules
and enforces their resource settings. There is no missing separate pod YAML that
must duplicate these settings.

**Cluster prerequisites are not provisioned by this renderer.** Kubernetes nodes,
NVIDIA device plugin, storage classes, KubeRay operator installation, namespace,
PVC and Secret creation are supplied by `setup_cluster.py` on a fresh suitable
VM, not by this renderer. The accepted fresh-install E2E run verified these
prerequisites and the workload lifecycle; see [7C results](7C_PLAN.md).

A Helm chart packages templates and values that generate manifests. Our Python
renderer generates a RayJob directly. Installing KubeRay through its Helm chart
and submitting our RayJob are separate operations.
