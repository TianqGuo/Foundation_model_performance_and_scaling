# §8.1 — Learning Rate Sweep: Discussion

**Best learning rate:** `1e-5` — peak validation accuracy 50.6% (step 145), final 47.3%, well above the ≥25% target.

**Other observed trends:**

At `lr=1e-4`, the gradient norm reached 47.5 and token entropy collapsed to 0.06 nats by the end of training, consistent with policy degeneration to near-deterministic (likely repetitive) outputs after an initial accuracy spike — the model "memorizes" a fixed response pattern rather than reasoning. At `lr=3e-5`, the opposite failure mode appeared: entropy rose to 0.38 nats while accuracy steadily degraded, indicating the policy drifted toward near-uniform output distributions and lost the reasoning format it briefly learned in the first few steps.
