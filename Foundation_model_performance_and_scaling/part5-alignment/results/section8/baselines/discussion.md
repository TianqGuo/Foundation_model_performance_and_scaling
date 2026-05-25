# §8.2 — Effect of Baselining: Discussion

**Best loss type:** `reinforce_with_baseline` — final validation accuracy 47.3% vs 32.1% for `no_baseline`.

**Other observed trends:**

The gradient norm for `no_baseline` (0.34) was roughly 20× lower than for `reinforce_with_baseline` (6.78), confirming that raw rewards provide a much weaker training signal — without mean centering, high-reward and low-reward rollouts within the same group receive similar-magnitude updates, leaving the policy nearly stationary. Token entropy for `no_baseline` remained consistently higher (0.258 vs 0.158 at the final step) and showed no downward trend across 200 steps, indicating the policy never converged on a consistent reasoning format due to the high-variance, uncentered gradient estimates.