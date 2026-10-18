# Part 7 — Questions and answers

High-level notes from our discussions. Implementation details are in
[PLAN.md](../PLAN.md), [GRPO mapping](7A_GRPO_MAPPING.md) and the
[cloud runbook](7A_RUNBOOK.md). Current status: the bounded 7A cloud smoke,
artifact uploads and pinned model-weight reload are verified. 7B same-host training-state recovery is also cloud-verified and reviewed.
Driver/vLLM RNG and replacement-node recovery remain outside its scope. See
[the 7B plan](7B_PLAN.md).

## What did 7A and 7B verify?

Two A100 40 GB GPUs ran the verl workload with two training ranks, 338 sharded
parameter tensors per rank, sampled sharded gradients/Adam moments, six optimizer
calls per rank, evaluation and checkpoint save. The upload run had nonzero
gradients on steps 2/3. Its BF16 model export and private Hub uploads completed,
and a new smoke loaded that exported model at the recorded commit successfully.
Logs and receipts are retained locally; the instance has been released.

This is bounded infrastructure verification, not a reliable model-quality gain.
The reload smoke had zero training rewards; 32-prompt evaluation scores varied
between runs. 7B subsequently verified same-host interruption/recovery, Ray
placement and exact restoration of audited state on two A100 80 GB GPUs.
Multi-node execution and broader numerical/tokenizer equivalence remain
unverified. The original 7A full checkpoints were discarded; the reviewed 7B
full checkpoint and small evidence are retained privately on the Hub.

## Which model and dataset are we starting with? Can we switch later?

Start with **Qwen2.5-Math-1.5B and MATH**, carried forward from Part 5.
Qwen is the model; GRPO/GSPO are training algorithms. A larger model is a later
practice goal. Model paths, prompt handling, sequence limits and resource settings
are configurable, but each switch still needs architecture/tokenizer support and
GPU-memory validation. A path change alone does not establish compatibility.

## Where do the 1,023 training and 32 evaluation prompts come from?

They are subsets of Part 5's existing MATH files, not newly generated examples.
Preparation selects the first 1,024 training records and explicitly excludes one
empty solution, leaving 1,023. The 32 evaluation records are selected from Part 5's
validation file using its fixed-subset helper with seed 12345. Saved original
indices and file hashes identify the exact selections.

## Is 32 evaluation prompts enough? Is that the Part 5 evaluation size?

It is enough for a bounded infrastructure smoke to check that evaluation works,
but too small for a reliable model-quality comparison. The corrected Part 5 pilot
used **128 fixed evaluation prompts**. A later comparison should align the saved
subset, generation limits and grading protocol, and retain uncertainty and seed
limitations. Smoke accuracy is not directly comparable to that pilot.

## Do we reuse Part 5's prompt and reward function?

Yes. `math_workload.py` preserves the rendered Part 5 prompt and calls its
`r1_zero_reward_fn` through a thin reward callback. Prepared bundles include exact
reference snapshots so cloud workers do not depend on the Part 5 environment.
The training reward is binary: a response must be correct and correctly formatted.
Rendered-text equality passed CPU checks, and the adapters were exercised in
cloud generation, grading and evaluation. Full token-ID/response-mask equivalence
to Part 5 has not been established; a successful smoke does not prove that parity.

## Why convert JSONL to Parquet? Does verl require it?

Parquet is a supported, convenient format for the structured records our adapter
produces: prompt messages, ground truth, dataset identity and source metadata.
The selected verl dataset loader also supports JSON input; Parquet is our chosen
interchange format, not an algorithm requirement. The important contract is the
record schema and its interpretation. Conversion also creates a portable bundle
with explicit provenance and a saved evaluation selection.

## What does verl coordinate, and what do Ray, FSDP2 and vLLM do?

verl coordinates the RL workflow: data batches, response generation, reward
calculation, advantages, policy updates, weight synchronization, evaluation and
checkpointing. Ray manages processes and resource placement; PyTorch FSDP2 shards
training state; vLLM generates responses. verl includes algorithm implementations
and extension points as well as orchestration. Ray Train does not wrap our verl
workload; it is reserved for optional SFT later in Part 7.

## Did we implement FSDP2 or leave vLLM configuration to verl?

We did not implement either engine. Our YAML selects FSDP2, two-rank sharding,
mixed precision and parameter/optimizer offloading. PyTorch implements FSDP2,
and verl integrates it into its workers. Diagnostics check actual sharding;
configuration flags alone were not our completion evidence.

The YAML's `rollout` section explicitly selects vLLM and sets parallelism,
GPU-memory utilization, sequence limits, sampling and evaluation. verl handles
worker coordination and model-weight synchronization; vLLM handles generation
and KV-cache management. Unspecified settings inherit the pinned framework's defaults.

## What changes when we use more GPUs or multiple instances?

Revisit training node/GPU counts, FSDP sharding, global/microbatch sizes, rollout
parallelism and memory limits together. Multiple machines also need Ray cluster
connectivity, worker placement and durable model/data access. Increasing a rollout
parallelism value alone does not configure multi-node training. Choose the topology
for the model and hardware, then verify actual placement, sharding and performance.

## Why does `math_workload.py` barely mention verl?

It handles data, prompts and rewards without importing the training framework,
so those responsibilities can be checked with lightweight CPU tests.
`r1_agent.py` extends verl's single-turn generation loop to preserve stopping and
sampling settings. `run_7a.py` and the YAML files connect these adapters to the full
framework. Both adapters were used in the verified cloud runs; their small size
reflects delegation to the framework.

## Why call `run_ppo` when we want GRPO? Is it a placeholder?

No. `run_ppo` is the selected verl release's shared training entry point, including
for GRPO. Our YAML sets `algorithm.adv_estimator: grpo` and generates four responses
per prompt. Advantages use group-relative rewards, while policy updates use a
clipped objective. GRPO shares this objective machinery with PPO, but does not
require a learned value critic. The entry-point name does not select the algorithm.

## Are we running Part 5's custom GRPO implementation?

We reuse its workload and explicitly map its selected algorithm settings to
**verl's implementation**; we do not execute Part 5's training loop. The mapping
includes group normalization, clipping, loss reduction and optimizer settings.
This is not an untouched default preset or an exact reproduction: numerical
precision, tensor layouts, sampling seeds and distributed execution can differ.
`part7_grpo` is our diagnostic loss wrapper; it checks frozen-policy agreement and
then delegates the objective to verl's existing vanilla clipped loss.

## Can we implement a customized GRPO algorithm within verl?

Yes. The Part 5 algorithm's mathematical components can be adapted to verl's
interfaces. Supported settings belong in configuration; custom rewards use
callbacks; custom advantages and policy objectives can use registered functions.
verl can retain responsibility for rollout, distributed execution, synchronization
and checkpointing while those functions define the algorithm.

This is adaptation, not loading the entire Part 5 script unchanged. Custom code
must respect batch shapes, response masks, loss reduction and distributed gradient
scaling. Exact FP32 log-softmax or stable microbatch behavior may require engine
or worker changes. More extensive schedule/backend changes involve deeper
integration. The framework is flexible, but not every choice is a YAML switch.
No custom Part 5 algorithm port is currently requested or implemented.

## Why are there separate environment, configuration and runner files?

| Files | Purpose |
| --- | --- |
| `infrastructure/7a_verl/part7A.sh` | Single cloud entry point for setup, preparation, training and optional Hub retention. |
| `infrastructure/7a_verl/environment/requirements.txt` | Deliberately selected direct library versions and exact verl commit. |
| `infrastructure/7a_verl/environment/requirements-cu130.in` | Official CUDA wheel URLs used during resolution. |
| `infrastructure/7a_verl/environment/requirements.lock` | Resolved direct and indirect application dependency versions. |
| `infrastructure/7a_verl/environment/setup_7a.sh` | Isolated cloud installation of matching CUDA 13 wheels and import checks. |
| `infrastructure/7a_verl/config/grpo_smoke.yaml` | Algorithm, batching, FSDP2/vLLM resources, evaluation and checkpoint settings. |
| `infrastructure/7a_verl/config/r1_agents.yaml` | Registration of our generation adapter with verl. |
| `infrastructure/7a_verl/run_7a.py` | Configuration/path assembly, bundle checks, evidence capture and framework launch. |
| `infrastructure/7a_verl/verl_diagnostics.py` | Numerical checks and actual rank/sharding evidence during cloud execution. |
| `infrastructure/7a_verl/hf_artifacts.py` | Authentication/account checks, private repository naming, verl model export, direct uploads and commit receipts. |

This separation keeps installation choices, experiment settings and launch logic
independently reviewable. The runner delegates training to verl rather than
implementing another training loop.

## Do checkpoints have to pass through my laptop?

No. A fresh smoke downloads the original base model automatically and needs no
previous checkpoint. To reuse trained weights, the cloud runner can export/upload
a private Hugging Face model and a later instance can download it by repository
ID and commit revision. Full optimizer/training state is needed only for exact
continuation; its upload is optional and goes directly from cloud to the Hub.
Logs/configuration can be stored independently in a private artifact repository.
The runbook documents these options. Hub file presence, model reload and verified
training recovery are distinct checks; recovery remains part of 7B.

For the fewest manual steps, use `bash infrastructure/7a_verl/part7A.sh --upload-hf`.
It reuses Part 6-style saved Hub authentication or prompts once for a write token,
verifies its account and chooses private repository names automatically. A token
provided through instance secrets makes the flow noninteractive.

## Do I need separate CLI authentication or manual environment setup?

No separate login command is required with `--upload-hf`: the runner prepares its
isolated environment, reuses a saved Hugging Face login or `HF_TOKEN`, or prompts
once for a write token. It verifies the token's account before training and chooses
private destinations automatically. This uses the same Hub authentication as
Part 6. A new VM needs a token again unless credentials are supplied through its
secret configuration. Source must contain the latest changes before running.

## How are uploads named, and can I go back to an earlier model?

The default model repository is
`ACCOUNT/part7-7a-smoke-execution-UTC_TIMESTAMP-RANDOM_SUFFIX`; the evidence
repository adds `-artifacts`. Each default execution has a distinct name.
`--output results/7a/grpo-smoke-001` instead uses
`ACCOUNT/part7-7a-grpo-smoke-001`; choose a new name for each run. Repository
variables can override automatic destinations. `hf_upload.json` records the
repository IDs and exact commit revisions.

Use `--model-id REPOSITORY --model-revision FULL_COMMIT_HASH` to start fresh
training from a selected model's weights. Explicitly uploading again to the same
repository creates a new commit; earlier revisions can still be selected while
retained. This is weights-only restart, not restoration of optimizer or progress.

## Should model and artifact repositories be public?

The current runner enforces private repositories; public model uploads are not
implemented. Public release may suit selected useful models with model cards,
while raw operational evidence and training state can remain private. Your account
dashboard showed more public than private storage, but public capacity should not
be treated as unrestricted backup space. Storage is shared across repositories;
creating more repositories does not increase the account allowance. Check current
usage and the [Hub storage policy](https://huggingface.co/docs/hub/storage-limits)
before retaining many versions. Default 7A artifact uploads exclude full checkpoints; the 7B recovery
experiment retains one complete checkpoint directly on the Hub.

## What does the 7B runner add to the Ray already used in 7A?

7A already runs verl's Ray workers. 7B adds a baseline/controlled-interruption/
pinned-Hub-restore experiment, explicit placement records and comparisons of
restored model/Adam/scheduler/worker RNG and trainer data/step state before further
training. Its manifest inventories checkpoint files and hashes; it is not a Ray
configuration file. YAML controls verl/Ray execution, while the manifest verifies
the resume payload. The reviewed two-A100-80GB experiment now proves exact
restoration of the audited training state and successful step-3 continuation in
a fresh Ray runtime. Driver/vLLM RNG and replacement-host recovery are not claimed.

## How do 7C's manifests, Docker image, Kubernetes, KubeRay, RayJob and verl fit together?

The **7C RayJob manifest contains both Kubernetes pod configuration and Ray
cluster configuration**. It is submitted to the Kubernetes API. The installed
KubeRay operator watches the RayJob and creates/manages the requested Ray cluster;
Kubernetes schedules its pods, Ray schedules actors inside them, and verl
coordinates training through those actors.

```text
7C RayJob manifest
  Contains:
  • Kubernetes settings: image, resources, ports, storage, Secrets
  • Ray settings: head/worker groups, Ray startup parameters
  • Job settings: execution command, deadline, cleanup
                │
                ▼
         Kubernetes API
         Stores the RayJob
                │
                ▼
         KubeRay operator
         Reads the RayJob and manages its Ray cluster
                │
                ▼
         Kubernetes schedules the pods
                │
                ├── CPU head pod
                └── Worker pod with two GPUs
                           │
                           ▼
                    Ray runtime
                    Schedules actors
                           │
                           ▼
                    verl framework
                    Coordinates GRPO
                           │
                           ├── FSDP2 / PyTorch: training
                           └── vLLM: rollout generation
```

The earlier configuration and verification files remain part of the workload:

```text
7A training configuration ─────────────► verl
   Model, data, GRPO, FSDP2, vLLM settings

7A data manifest ───────► Our launcher verifies data hashes
                                      │
                                      ▼
                                     verl

7B checkpoint manifest ► Our recovery code verifies saved state
                                      │
                                      ▼
                            verl restores and resumes
```

The word *manifest* therefore has different uses. The 7A data manifest describes
prepared files/provenance/hashes; the 7B checkpoint manifest describes saved
training state, topology and integrity; our code verifies both. The 7C RayJob
manifest declares a deployment and execution command; Kubernetes and KubeRay
consume it. A RayJob is a Kubernetes resource, not the operator or runtime itself.
KubeRay is the operator; Ray is the runtime; verl is the training framework.

The Dockerfile builds the image containing our code and pinned dependencies.
`render.py` normally runs outside that image, after image publication, to generate
a manifest with its registry digest. `preflight.py` runs inside deployed containers
to verify GPU placement, CUDA computation, imports and Hub Secret authentication;
it loads no model. `run_7c.py` then connects to the existing Ray cluster for native
verl training/interruption/resume. Neither renderer nor preflight runs during the
image build. The intended sequence is build/publish → render → submit → preflight
→ training/interruption → fresh-RayJob resume → evidence retention and cleanup.

Worker groups describe groups of Ray worker pods, not automatic training/inference
roles. Our `training` group supplies both FSDP2 training and vLLM rollout generation.
The separately deployed inference service is planned for 7D.

The RayJob's pod templates already define image, resource requests/limits, ports,
node selection, shared memory, volume mounts and Secret mounts. Separate cluster
prerequisites—Kubernetes nodes, NVIDIA device plugin, storage classes, KubeRay
installation and creation of the referenced namespace/PVC/Secrets—are not supplied
by the renderer. A Helm chart packages templates/values to generate manifests and
manage releases; our Python renderer directly generates a RayJob. Helm can install
KubeRay separately from submission of these workload manifests.

See the [generated RayJob example](examples/7C_RAYJOB_EXAMPLE.yaml),
[manifest comparison guide](MANIFEST_GUIDE.md), [7C plan](7C_PLAN.md) and
[runbook](7C_RUNBOOK.md). The example uses placeholder image/node values and is
for reading. The bounded 7C fresh-install E2E run is now reviewed and complete,
including actual GPU training, pinned Hub restore and job cleanup; see 7C_PLAN
for tested settings and same-node recovery limits.

### Why does an Ubuntu Docker rental differ from an Ubuntu GPU VM for 7C?

Ubuntu describes the userspace; it does not imply host administrative access.
The checked Vast rental was an unprivileged Docker container lacking the required
system/network capabilities. The checked Lambda rental was a KVM VM with systemd,
sudo and two H100 GPUs. An initial VM was deleted after hardware checks; a later VM completed fresh
Kubernetes installation and the bounded E2E experiment. `part7C.sh setup --cloud --with-hf-secret` now prepares
a fresh suitable VM; `--check` only previews its configuration. The setup does not
replace the host CUDA toolkit. CUDA 13.x comes from the separate training image.
Its same-node PVC survives pod replacement, while Hub backup is needed before VM
deletion. Fresh setup, GPU preflight and the bounded recovery experiment have
passed cloud review; replacement-node recovery remains outside that result.

### Do I need manual stage-by-stage commands or to copy data from my laptop?

No. `part7C.sh e2e --cloud` now coordinates setup/readiness, prepares data from the
Part 5 files already in the cloud Git clone, stages to PVC and runs the complete
training/recovery experiment. The portable generated bundle is Git-ignored, but
its original train/validation JSONL sources are tracked. Copying an existing bundle
is an optional shortcut. On a clean VM, `--fresh-setup` requires the real setup path;
on an existing cluster it refuses rather than presenting reuse as clean-install
evidence. Diagnostics may still require individual commands when a stage fails.
CPU tests simulate external operations; real fresh setup, GPU preflight and full
training-state recovery each require their respective cloud evidence.


### What do the 7C test stages verify, and where does Hub recovery get its checkpoint metadata?

Staging copies the prepared bundle from the cloud VM to the shared PVC. GPU
preflight checks GPU assignment, CUDA computation, imports and Hub authentication
without training. Baseline, interruption and recovery are actual short training
runs: baseline provides an uninterrupted three-step reference; interruption saves
complete training state at step 2 and deliberately stops; recovery restores that
state in a new RayJob and continues at step 3. This tests a controlled checkpoint
boundary, not unexpected VM failure or replacement-node recovery.

Recovery uses three kinds of metadata:

| Metadata | Storage | Purpose |
| --- | --- | --- |
| Checkpoint manifest | PVC, then Hugging Face alongside the complete checkpoint | Records the saved step, configuration identities, file inventory and hashes. |
| Hub revision | Hugging Face repository history | Identifies the exact uploaded checkpoint commit. |
| `hf_checkpoint.json` upload receipt | Run directory on PVC, then controller VM evidence | Records repository ID, revision and saved step for the recovery job. |

```text
Upload complete checkpoint + manifest to private HF dataset repository
    |
Resolve uploaded commit and save repository/revision in hf_checkpoint.json
    |
Experiment controller reads receipt and configures a new recovery RayJob
    |
Download exact HF revision into a fresh PVC path
    |
Verify manifest/file hashes -> restore native training state -> audit continuation
```

The checkpoint and its manifest are retained together on Hugging Face. Recovery
therefore tests upload, pinned download, integrity and restoration of model,
optimizer, scheduler and recorded RNG/data state; uploading files alone does not
prove recovery. The current upload receipt is not itself included in that
checkpoint upload. If the VM/PVC is lost, select the repository and revision from
retained evidence or HF history; automatic discovery after controller loss is not
implemented. The bounded 7C experiment has now passed and its synced evidence
was reviewed locally, including a rerun of the recovery checks. This verifies
same-node restoration, not replacement-node or bitwise equivalence.

See [checkpoint transfer code](../infrastructure/7c_kuberay/checkpoint_transfer.py)
and [experiment orchestration](../infrastructure/7c_kuberay/experiment.py).


### How does the E2E command execute run_7c.py inside the image, and where is Ray imported in 7C versus 7D?

The Docker image packages Python, Ray, verl, vLLM and our source files. Copying
`run_7c.py` into the image does not execute it. In the normal workflow, the E2E
controller submits a RayJob whose application command runs that file after
KubeRay has started the Ray cluster. `run_7c.py` does not provision Kubernetes or
pull the image, and `render.py` does not modify the image's startup configuration.

```text
ON THE CLOUD VM
part7C.sh e2e --cloud --fresh-setup
    |
    +--> e2e.py
           |
           +--> setup_cluster.py: install Kubernetes/KubeRay
           +--> prepare data
           +--> experiment.py: coordinate the test phases
                    |
                    +--> render.py: generate each RayJob manifest
                    |      - which image?
                    |      - which resources?
                    |      - which entrypoint command?
                    |
                    +--> job_control.py: submit manifest to Kubernetes
                                  |
                     Kubernetes + KubeRay start the pods
                                  |
INSIDE THE CONTAINERS             |
                     +------------+-------------+
                     |                          |
              preflight.py                 run_7c.py
              GPU checks                   baseline / interrupt / resume
                                                |
                                                +--> restore files if requested
                                                +--> compose/validate configuration
                                                +--> connect to existing Ray
                                                +--> execute native verl
```

There are two distinct startup settings:

| Setting | Meaning |
| --- | --- |
| Dockerfile `ENTRYPOINT` / `CMD` | Default process when a container starts. Our image clears ENTRYPOINT and defaults CMD to bash; KubeRay supplies Ray startup commands in the Kubernetes pod specification. The stored image remains unchanged. |
| RayJob `spec.entrypoint` | Application command submitted to the already running Ray cluster, such as `python /opt/part7/infrastructure/7c_kuberay/run_7c.py ...`. It is not the Docker ENTRYPOINT. |

`render.py` selects `preflight.py` for the preflight phase and `run_7c.py` for
training phases. The latter imports Ray, calls `ray.init(address="auto", ...)`,
prepares/validates the workload and invokes verl. During resume it downloads and
verifies checkpoint files; verl performs native training-state restoration and
training. Ray runs the application as a process inside the existing cluster;
no extra shell entrypoint in the Dockerfile is required for that application.

In **both 7C and 7D, Ray integration executes inside containers**. In 7C our
launcher explicitly imports Ray and invokes verl. The prepared 7D configuration
instead names `ray.serve.llm:build_openai_app`; Ray Serve imports and executes that
installed framework builder inside the cluster. The renderer runs on the cloud
VM, but the builder it references does not. The serving image must still contain
Ray Serve LLM, vLLM and all required application/staging code. That 7D integration
is source-prepared and remains cloud-unverified.

Although the image contains the infrastructure source tree, the normal workflow
runs `e2e.py`, setup, renderer, job controller and experiment coordinator on the
VM. Preflight, the training launcher and verl/Ray workers run in Kubernetes
containers. The VM also uses a separate temporary Docker container to prepare
data; that is not a training pod.

Code references:

- [E2E shell dispatch](../infrastructure/7c_kuberay/part7C.sh),
  [E2E orchestration](../infrastructure/7c_kuberay/e2e.py) and
  [phase coordination](../infrastructure/7c_kuberay/experiment.py).
- [Source packaging](../infrastructure/7c_kuberay/Dockerfile) and
  [RayJob application command](../infrastructure/7c_kuberay/render.py).
- [Training Ray import](../infrastructure/7c_kuberay/run_7c.py),
  [cluster connection](../infrastructure/7c_kuberay/run_7c.py) and
  [remote coordinator/native verl invocation](../infrastructure/7c_kuberay/run_7c.py).
- [GPU-probe Ray import](../infrastructure/7c_kuberay/preflight.py) and
  [7D native Serve builder reference](../infrastructure/7d_serve/render.py).


### How do Kubernetes, KubeRay and the Ray cluster relate, and how do head/worker pods use the same image and resources?

Kubernetes manages machines, pods and containers. A Ray cluster is a distributed
runtime formed by Ray head and worker processes running inside those containers;
it is not another machine cluster that replaces Kubernetes. Our bounded 7C
configuration uses two Ray pods on one GPU VM:

```text
Cloud VM — one Kubernetes node
└── Kubernetes cluster (K3s)
    ├── KubeRay operator pod
    │     Creates and manages Ray pods
    │
    ├── Ray head pod
    │   └── Container running Ray head processes
    │
    └── Ray worker pod
        └── Container running Ray worker processes
            ├── Preparation actor (temporary)
            └── Training workers (after preparation)

        Head + worker processes together form ONE Ray cluster
```

A Ray cluster can span multiple pods and machines. A Kubernetes cluster can host
multiple independent Ray clusters. In 7C, each phase's RayJob creates its own
RayCluster; 7D's RayService is configured to manage a serving RayCluster. The
Kubernetes `RayCluster` object is the desired configuration, not the runtime:

```text
RayCluster object stored in Kubernetes
    ↓ watched by
KubeRay operator
    ↓ creates
Head and worker pods
    ↓ run
Ray processes forming the actual Ray cluster
```

Our fresh-VM setup installs the KubeRay operator through its Helm chart first.
Submitting a RayJob/RayService does not install the operator. The operator uses
its own KubeRay image; both application pods use our shared workload image:

```text
setup_cluster.py
    └── Installs KubeRay operator using its operator image

RayJob / RayService manifest
    └── Existing KubeRay operator reads it
          ├── Head pod   → our image → ray start --head …
          └── Worker pod → our image → ray start --address=HEAD_ADDRESS …
```

The startup command selects head versus worker behavior. Pod resource settings
also differ: our 7C head is CPU-only, and its worker requests two GPUs. The image
provides the software, while KubeRay supplies the role-specific startup commands.
Different head/worker images are possible; this deployment shares one to keep
dependencies consistent. The head coordinates the Ray runtime and provides the
dashboard/job API; workers execute scheduled application work. This is our chosen
placement, not a rule that head processes can never execute application tasks.

Within `run_7c.py`, the reviewed lines have these distinct responsibilities:

| Location | Meaning |
| --- | --- |
| Line 105 | `ray.get_runtime_context().get_node_id()` records the preparation actor's current Ray node ID; it is not Kubernetes context or the entire cluster configuration. |
| Line 144 | `ray.init(address="auto", runtime_env=runtime)` connects the launcher to the existing Ray cluster and supplies runtime settings for child tasks/actors; it does not connect to the Kubernetes controller. |
| Line 153 | `@ray.remote(num_cpus=1, num_gpus=2)` defines the preparation actor's logical resource requirements. Declaration alone does not allocate them. |
| Lines 160–162 | `.remote()` requests actor creation on the chosen worker; preparation executes, its result is collected, then the actor is killed to release resources before training. |
| Lines 183–186 | Configure the remote training coordinator (one CPU, zero GPUs), then invoke native verl, which creates/uses the coordinator and its distributed GPU workers. |

```text
Connect to existing Ray cluster
    ↓
Create preparation actor requesting both GPUs
    ↓
Prepare/validate model and record worker context
    ↓
Release preparation actor's resources
    ↓
Invoke verl with the configured remote coordinator
    ↓
verl creates and coordinates GPU training workers
```

There are two scheduling layers: Kubernetes has already assigned GPUs to the
worker pod; Ray schedules actors against that pod's advertised logical resources.
The preparation actor does not request additional GPUs from Kubernetes or reserve
quota in a shared-cluster queue. Ray resource requests/placement groups do not
replace Kubernetes placement or Kueue admission, and do not by themselves provide
GPU-memory isolation.

See [training launcher](../infrastructure/7c_kuberay/run_7c.py),
[pod/cluster renderer](../infrastructure/7c_kuberay/render.py) and
[platform setup](../infrastructure/7c_kuberay/setup_cluster.py).
