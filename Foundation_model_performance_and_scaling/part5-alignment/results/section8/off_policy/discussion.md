# §8.5 — Off-Policy GRPO: Discussion

---

## §8.5.2 — Off-Policy Hyperparameter Sweep

### Sweep rationale

`rollout_batch_size=256` was fixed throughout. The sweep varied two dimensions:
- `epochs_per_rollout_batch` (1 vs 4): how many passes over each rollout batch
- `train_batch_size` (256 vs 128): mini-batch size within each epoch

These two parameters determine gradient updates per GRPO step = (rollout_batch_size / train_batch_size) × epochs:
- on-policy (e1, bs256): 1 update/step
- off-policy (e4, bs256): 4 updates/step
- off-policy (e4, bs128): 8 updates/step

A 50-step broad sweep was run first to quickly identify the best direction before committing to full 200-step runs.

### Broad sweep results (50 steps)

| Config | Peak Acc | Final Acc | Final Entropy |
|--------|----------|-----------|---------------|
| on-policy (e1, bs256) | 43.9% | 41.4% | 0.184 |
| off-policy (e4, bs256) | 48.6% | 48.6% | 0.112 |
| off-policy (e4, bs128) | 50.1% | 50.1% | 0.148 |

Both off-policy configs outperform on-policy within 50 steps, confirming that amortising vLLM generation cost across multiple gradient updates is beneficial. `epochs=4, bs=128` wins: 8 gradient updates per GRPO step delivers a further +1.5 pt over bs=256 (4 updates/step).

### Focused runs (200 steps)

| Config | Peak Acc | Peak Step | Final Acc | Final Entropy | Grad Norm (max) | Clip Frac (final) |
|--------|----------|-----------|-----------|---------------|-----------------|-------------------|
| on-policy (e1, bs256) | 45.7% | 80 | 40.1% | 0.265 | 47,104 | 0.255 |
| off-policy (e4, bs128) | 54.6% | 90 | 52.5% | 0.035 | 7,045,248 | 0.622 |

**Off-policy advantage (+9 pt peak):** more gradient updates per rollout translates directly to faster policy improvement. The on-policy baseline has to re-roll a new batch after every single update, wasting the information already present in the rollout data. Off-policy reuses each rollout batch 8 times, extracting more signal per generation cycle.

**Both runs peak early and decline:** accuracy peaks around step 80–90 for both configs and then regresses. This is a known phenomenon in GRPO/PPO training: as the policy learns to solve easier examples first, the gradient signal becomes increasingly concentrated on harder problems, and multiple gradient steps on stale rollouts begin to over-fit the reward on those specific rollout sequences rather than generalising.

**Gradient norm explosion:** both runs exhibit catastrophic gradient norm growth in later steps (47K for on-policy, 7M for off-policy). On-policy shows a high clip fraction (25.5%) despite epochs=1 — within the single gradient step, the policy moves far enough that subsequent mini-batches within the same step hit the clip boundary. Off-policy clip fraction reaches 62.2%, confirming severe policy drift by epochs 3–4.

**Entropy collapse in off-policy:** off-policy entropy collapses to 0.035 (near-deterministic), while on-policy stays at 0.265. The 8× gradient update multiplier accelerates the narrowing of the output distribution — the model rapidly converges to a small set of high-reward response patterns and stops exploring alternatives.

**Comparison to Expert Iteration:** EI entropy was stable or slowly declining across EI steps, reflecting the supervised-learning update at each step which preserves diversity. GRPO off-policy entropy collapses sharply and persistently, driven by the direct policy gradient pushing toward reward-maximising responses. This is a fundamental difference between SFT-based (EI) and RL-based (GRPO) training dynamics.

---

## §8.5.3 — Clip Ablation

| Config | Peak Acc | Peak Step | Final Acc | Final Entropy | Grad Norm (max) | Format Rate (final) |
|--------|----------|-----------|-----------|---------------|-----------------|---------------------|
| grpo_clip | 54.6% | 90 | 52.5% | 0.035 | 7,045,248 | 95.1% |
| grpo_no_clip | 48.1% | 160 | 44.5% | 0.166 | 11,454,022,541,186 | 77.7% |

**Clipping is essential for off-policy stability.** Without the importance weight bound, as the policy drifts across 4 epochs of gradient updates, the ratio π_θ/π_θ_old can grow arbitrarily large for tokens where the current policy assigns much higher probability than the old policy. These large ratios produce gradient updates orders of magnitude larger than the clipped version, destabilising training.

**Gradient norm catastrophe:** `grpo_no_clip` reaches a gradient norm of 11.5 trillion — six orders of magnitude worse than `grpo_clip`'s 7 million. This is not a small quantitative difference; it reflects completely uncontrolled policy updates.

**Format regression in no_clip:** the format rate degrades from 81.5% → 77.7% under `grpo_no_clip`, meaning the model is losing format compliance it had already acquired. This does not occur with clipping. The large unconstrained gradient updates are overwriting previously learned format structure, a form of catastrophic forgetting within the RL loop. `grpo_clip` format rate rises monotonically to 95.1%.

**Accuracy and peak timing:** `grpo_no_clip` achieves only 48.1% peak accuracy (vs 54.6%), and its peak comes much later at step 160. The unstable updates slow convergence — the policy makes progress in some steps but regresses in others, producing a slower, noisier trajectory toward the optimum.

**Entropy:** `grpo_no_clip` maintains higher entropy (0.166 vs 0.035). This is not genuine exploration — it is the natural consequence of destructive updates that prevent the policy from converging to any consistent strategy. The apparent diversity reflects incoherence, not breadth.

**Conclusion:** PPO-style clipping is not a minor regularisation tweak in the off-policy setting — it is what makes multi-epoch training viable. Without it, the unconstrained importance weights turn additional gradient updates from a benefit into a liability.