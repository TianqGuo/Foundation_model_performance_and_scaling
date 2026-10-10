# 7B — Ray operations and training-state recovery

Started: 2026-10-09. User authorized continuing to 7B. This first increment defines
implementation and acceptance. Checkpoint integrity/transfer helpers and native
resume preview are now prepared; runtime audits and cloud verification remain
pending. See [the runbook](7B_RUNBOOK.md).

## Initial topology

Reuse the verified Qwen2.5-Math-1.5B/MATH workload, pinned 7A environment and
single cloud node with two training ranks. Record Ray nodes, actor/placement-group
placement, logical resources, physical GPU identities and actual sharding.
Use built-in Ray metrics and worker logs first. No Kubernetes, Kueue, JobSet,
TP/PP or training-backend change is introduced. The future shell runner and
implementation belong under `infrastructure/7b_ray/`; do not copy the training loop.

## Checkpoint contract

The pinned [FSDP checkpoint manager](https://github.com/verl-project/verl/blob/bec9ef74768dd201881cd4e54cd0385e87caae27/verl/utils/checkpoint/fsdp_checkpoint_manager.py)
saves/loads per-rank model/optimizer files and extra scheduler/RNG state. The
pinned [trainer](https://github.com/verl-project/verl/blob/bec9ef74768dd201881cd4e54cd0385e87caae27/verl/trainer/ppo/ray_trainer.py)
saves `data.pt`, identifies the global step and supports `resume_mode=resume_path`
with `resume_from_path`. Missing data state falls back to a fresh loader; 7B must
reject that incomplete checkpoint. Native trainer HDFS resume is not implemented.
Reconfirm behavior against the exact installed checkout before adding hooks.

| State | Required evidence |
| --- | --- |
| Model/Adam | Complete two-rank shards; runtime restore audit before updates; nonempty optimizer moments and step continuity. |
| Scheduler | Restored state and LR; constant LR alone is insufficient proof. |
| Global progress | Saved/restored step and next executed step; total-step cap exceeds saved step. |
| Data | Required `data.pt`, restored loader progress and next source indices matching a continuation reference. |
| RNG | Saved trainer Python/NumPy/Torch/device state audited before subsequent consumption. |
| Rollout runtime | Disclose vLLM/server RNG, cached requests and in-flight generations not proven restorable. |

7A's BF16 export is weights only. Its full checkpoints were intentionally discarded;
7B needs a newly generated full checkpoint, not the exported model as resume state.

## Durable retention and experiment

Use a private Hugging Face artifact repository and direct cloud transfers. Manifest
both ranks, extra state, data progress, topology, versions and file hashes. Record
the Hub commit. Logs/receipts may be copied locally; full state need not pass through
a laptop. Keep one bounded recovery checkpoint within available private storage.

1. Run a bounded baseline; save a checkpoint before its final step and retain a
   continuation reference with state audits and next source indices.
2. Run a controlled-interruption trial. Interrupt the driver/job after a complete
   checkpoint is available, using framework completion evidence rather than an
   arbitrary sleep. Distinguish planned interruption from unexpected failure.
3. Upload the complete checkpoint directly to the Hub. The 7A uploader requires
   job success; interrupted-run retention needs an explicit checkpoint-completeness
   path rather than blindly reusing that success-only helper.
4. Start a fresh Ray runtime on the same node. Download the pinned artifact into
   a fresh directory, verify hashes and two-rank compatibility, and resume from
   that restored copy rather than the original checkpoint on disk.
5. Invoke verl's native full-state resume, audit restoration before new updates,
   then complete another rollout/update/evaluation/checkpoint cycle.

This establishes fresh-runtime recovery on the same host, not replacement-node
recovery. The latter requires a separate cloud allocation. An uninterrupted
continuation is a state/data reference, not a promise of identical sampled outputs.

Measure checkpoint/upload, download/integrity, startup, state-load completion and
first resumed update separately. State timing boundaries and cache conditions.
Durable artifacts must remain readable after producer exit.

## Implementation sequence

1. **Design/source review:** complete at planning scope in this increment.
2. **CPU preparation:** thin runner/configuration, manifest/restore validation and
   minimum cloud-only state-audit hooks around native save/load. Check synthetic
   missing shards/data, hash changes, topology mismatch and resume configuration.
   Never load model checkpoint tensors locally; preserve the 7A runner.
3. **Cloud verification:** one shell entry point for environment reuse/setup,
   authentication, baseline/interruption, retention, restore and evidence capture.
   Agree the live hardware offer and spending cap before renting; an old proposed
   cap does not authorize provisioning. No new instance is needed for CPU work.
4. **Review/reassess:** verify each restoration claim and placement from evidence,
   document unsupported state, update README/PLAN and reassess before 7C.

7B remains incomplete until cloud recovery audits pass. A resume flag, load message
or file listing alone is insufficient. Capture image/model/data/artifact identity,
exceptions and limitations. No resources are provisioned and no commits/pushes
are authorized by this document.
