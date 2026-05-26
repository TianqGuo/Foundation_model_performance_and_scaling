# §8.4 — Effect of Group Standard Deviation Normalization: Discussion

**Best setting:** `with_std` (standard GRPO) — peak accuracy 50.6% vs 48.1%, final accuracy 47.3% vs 46.5%.

## Commentary on findings

**Gradient norm:** The most noticeable trend is in gradient norm — `with_std` produces a final grad norm of 6.78 versus 3.42 for `no_std` (roughly 2×). This is expected: dividing the advantage by the group std scales up gradients for questions where rollout rewards are spread out (high std), and scales down gradients for questions where all rollouts score similarly (low std). The net effect is larger gradient magnitudes overall. `no_std` keeps advantages in a narrower range (reward − mean only), leading to smaller, more uniform updates.

**Accuracy:** `with_std` outperforms `no_std` by ~1 point on both peak and final accuracy. The Dr. GRPO paper argues that std normalization introduces bias on "easy" groups (all correct) and "hard" groups (all wrong), since dividing by a near-zero std artificially amplifies gradients on uninformative batches. In practice here, the amplification appears beneficial — the stronger gradient signal on high-variance groups (mixed correct/incorrect rollouts) provides better learning signal than it loses from the occasional uninformative-group amplification.

**Entropy:** `no_std` converges to lower entropy (0.097 vs 0.158), indicating a tighter, more deterministic output distribution. This is consistent with the smaller gradient updates — the policy moves less per step and settles into a narrower mode. `with_std`'s slightly higher entropy suggests it continues exploring reasoning strategies for longer, which may contribute to its higher peak accuracy.

**Conclusion:** `with_std` (standard GRPO) is the better setting on this task and is used for §8.5+. The Dr. GRPO motivation (avoiding amplification of uninformative groups) is theoretically sound but did not improve accuracy here, likely because the MATH dataset has enough diversity that most groups contain mixed outcomes and the std signal is informative rather than noisy.
