# Part 7 — ML Infrastructure

## Scope and authorization

- Read `PLAN.md` before work. The accepted roadmap comes from Stage 4 of
  `../part5-alignment/PART5_MODERNIZATION_PLAN.md`; Part 5 working conventions
  are in `../part5-alignment/AGENTS.md`.
- Core scope is 7A–7D; 7E is optional. Part 5 modernization is complete;
  GSPO tuning is deferred and does not block Part 7.
- Preserve Parts 1–6, their environments, local edits and historical artifacts.
  Use them as references; do not adapt Part 5 training to the new framework.
- 7A steps 1–3 were authorized on 2026-10-04. Consult
  `documents/7A_ENVIRONMENT.md`, `documents/7A_GRPO_MAPPING.md` and
  `documents/7A_RUNBOOK.md`. Cloud execution remains pending. Agree hardware and
  a spending cap before paid provisioning.
- Do not stage, commit or push without explicit user approval.

## Execution and code

- Start with Qwen2.5-Math-1.5B and MATH, while keeping model identity/path,
  tokenizer/prompt handling and resource settings configurable. Larger models
  are a future practice goal; keep workload-specific adapters separate from
  infrastructure and validate compatibility/memory before each model switch.
- All model loading, inference, serving, GPU smoke runs and training execute on
  cloud GPUs. Local checks are static or lightweight CPU checks without loading
  model checkpoints. Detect hardware and dependencies; do not assume availability.
- Use verl's coordination and synchronization; Ray Train does not wrap verl.
  Pin a compatible isolated stack; training and serving may use different images.
- Preserve the MATH adapter's explicit exclusions and provenance. Use the bundled
  reference grader on cloud; do not silently swap reward or chat-template semantics.
- Prefer thin adapters, framework configuration and existing helpers over copied
  training loops. Add cohesive modules only when their responsibility is needed.
- Keep milestone runners/configuration/environments under `infrastructure/`;
  7A lives in `infrastructure/7a_verl/`. Keep reusable dataset/reward/generation
  adapters in `workloads/`; add future milestone directories when needed.
- Use descriptive names, `argparse` for custom CLIs, project-root-derived paths,
  LF endings and runners beside their implementation with usage in headers.
  Discover usual model/data locations; path overrides should be optional.
- Keep datasets, caches, checkpoints and credentials out of source commits.
  Store run evidence in descriptive `results/` subdirectories and preserve cloud
  artifacts before resource cleanup. Never store or print API keys.

## Verification and documentation

- Work in roughly one-hour, reviewable increments; reassess after 7A and 7B.
- Run relevant CPU checks before cloud execution. Supply runnable cloud commands
  and explain how uncommitted code reaches the instance without requiring a push.
- Record resolved configuration, dependency/model/data identities, placement,
  metrics, logs and limitations. Mark GPU verification pending until reviewed.
- Prove two actual training ranks and sharded state; flags alone are insufficient.
  Distinguish weights-only restart from verified training-state recovery.
- Keep READMEs concise: capabilities, measured results, settings and interpretation
  caveats. Put setup, commands and troubleshooting in a runbook; track progress
  and validation limits in `PLAN.md`. Do not claim unmeasured scalability or speedups.
- Current user instructions take precedence over historical guidance.
