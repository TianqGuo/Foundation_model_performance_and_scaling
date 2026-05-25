# §8.3 — Length Normalization: Discussion

## §8.3.1 Conceptual Analysis

`masked_mean` averages per-token loss over the response tokens of each sequence, giving every token equal weight *within* its sequence but up-weighting tokens in shorter responses relative to longer ones (a token in a 4-token response gets gradient weight 1/4; a token in a 7-token response gets 1/7). `masked_normalize` divides the token loss sum by a fixed constant (e.g. `max_response_tokens=1024`), so every token receives the same gradient weight (1/1024) regardless of which response it came from — but longer responses contribute a proportionally larger per-example loss to the batch mean.

**Pros and cons:**

- `masked_mean` is simple and keeps per-sequence loss in a consistent range regardless of length, making batch-level gradients stable. However, it implicitly up-weights tokens in short responses, which may bias learning toward brevity — a short correct answer gets stronger per-token reinforcement than a long correct answer.
- `masked_normalize` equalises per-token gradient weight across all response lengths, which is fairer to longer reasoning chains. The trade-off is that the per-example loss scale now varies with response length, introducing gradient variance across the batch whenever responses differ substantially in length.

**When each is preferable:** `masked_normalize` is conceptually fairer for long reasoning RL — each token receives equal credit regardless of which sequence it came from, so longer correct reasoning chains get proportional reinforcement. However, this advantage only holds in the right regime: long responses, high length variance within a batch, and a reward that interacts with reasoning depth. In GRPO rollouts on short math answers, response lengths are short (~200–280 tokens) and the reward is binary (correct/incorrect), so there is little length signal for the aggregation choice to act on differently. In this regime `masked_normalize`'s latent advantage has nothing to bite on, and the only remaining distinction is gradient stability — where `masked_mean` wins. `masked_mean` sacrifices per-token fairness for sequence-level stability, which is the more robust choice when rollout lengths are unpredictable or when the task does not reward longer reasoning chains specifically.

**Caveat on this result:** this comparison is a single run per method. RL training is stochastic enough that entropy instability can appear on one seed and not another. The 4× entropy gap (0.681 vs 0.158) is large enough to be meaningful, but the accuracy difference is small (~1 point), limiting practical significance in this regime. To firmly establish `masked_normalize` as less stable would require multiple seeds. The regime where the two methods are expected to diverge more clearly — long chain-of-thought, high length variance, reward sensitive to reasoning depth — was not present in this experiment.

## §8.3.2 Empirical Findings

**Best length normalization:** `masked_mean` — peak accuracy 50.6% vs 48.5%, final accuracy 47.3% vs 46.2%.

**Other observed trends:**

Token entropy for `masked_normalize` rose sharply to 0.681 nats by the final step compared to 0.158 for `masked_mean`, indicating the policy failed to converge on a consistent reasoning format — consistent with the noisier gradient signal introduced by length-dependent loss scaling. The gradient norm was also higher for `masked_normalize` (10.56 vs 6.78), reflecting larger and more variable per-example losses, and average response length was paradoxically shorter (214.8 vs 234.5 tokens), suggesting the length incentive did not promote richer reasoning but instead destabilised training.