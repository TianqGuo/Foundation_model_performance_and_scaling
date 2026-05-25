# §8.3 — Length Normalization: Discussion

## §8.3.1 Conceptual Analysis

`masked_mean` averages per-token loss over the response tokens of each sequence, giving every token equal weight *within* its sequence but up-weighting tokens in shorter responses relative to longer ones (a token in a 4-token response gets gradient weight 1/4; a token in a 7-token response gets 1/7). `masked_normalize` divides the token loss sum by a fixed constant (e.g. `max_response_tokens=1024`), so every token receives the same gradient weight (1/1024) regardless of which response it came from — but longer responses contribute a proportionally larger per-example loss to the batch mean.

**Pros and cons:**

- `masked_mean` is simple and keeps per-sequence loss in a consistent range regardless of length, making batch-level gradients stable. However, it implicitly up-weights tokens in short responses, which may bias learning toward brevity — a short correct answer gets stronger per-token reinforcement than a long correct answer.
- `masked_normalize` equalises per-token gradient weight across all response lengths, which is fairer to longer reasoning chains. The trade-off is that the per-example loss scale now varies with response length, introducing gradient variance across the batch whenever responses differ substantially in length.

**When each is preferable:** `masked_normalize` is conceptually fairer for long reasoning RL — each token receives equal credit regardless of which sequence it came from, so longer correct reasoning chains get proportional reinforcement. However, this advantage only holds when response lengths are relatively uniform across the batch. In GRPO rollouts, response lengths vary widely (e.g. 50 to 900 tokens), meaning `masked_normalize` can give a long response ~18× more gradient weight than a short one. This length imbalance dominates the update signal and introduces the instability observed empirically (higher grad norm, entropy collapse). `masked_mean` sacrifices per-token fairness for sequence-level stability, which is the more robust choice when rollout lengths are unpredictable.

## §8.3.2 Empirical Findings

**Best length normalization:** `masked_mean` — peak accuracy 50.6% vs 48.5%, final accuracy 47.3% vs 46.2%.

**Other observed trends:**

Token entropy for `masked_normalize` rose sharply to 0.681 nats by the final step compared to 0.158 for `masked_mean`, indicating the policy failed to converge on a consistent reasoning format — consistent with the noisier gradient signal introduced by length-dependent loss scaling. The gradient norm was also higher for `masked_normalize` (10.56 vs 6.78), reflecting larger and more variable per-example losses, and average response length was paradoxically shorter (214.8 vs 234.5 tokens), suggesting the length incentive did not promote richer reasoning but instead destabilised training.