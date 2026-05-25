# §8.3 — Length Normalization: Discussion

## §8.3.1 Conceptual Analysis

`masked_mean` averages per-token loss over the response tokens of each sequence, giving every token equal weight within its sequence and every sequence equal weight in the batch regardless of length. `masked_normalize` divides the token loss sum by a fixed constant (e.g. `max_response_tokens`), so a longer response produces a proportionally larger per-example loss — assigning more gradient credit to sequences with more tokens.

**Pros and cons:**

- `masked_mean` produces stable, length-neutral gradient estimates. It is preferable when response correctness is independent of length, and when batch composition varies significantly in response length (which is typical in GRPO rollouts where some responses are cut short and others use the full budget).
- `masked_normalize` implicitly rewards longer reasoning chains with more gradient signal, which could help when extended chain-of-thought is genuinely needed for correctness. However, it introduces gradient variance proportional to length variation within the batch, and risks incentivising verbosity or padding to inflate the per-example loss magnitude.

**When each is preferable:** `masked_normalize` could be beneficial in a setting where all responses are near the maximum length and the task clearly rewards detailed step-by-step work. `masked_mean` is preferable — and safer — in general settings where response length varies unpredictably, as it keeps the gradient scale consistent across examples.

## §8.3.2 Empirical Findings

**Best length normalization:** `masked_mean` — peak accuracy 50.6% vs 48.5%, final accuracy 47.3% vs 46.2%.

**Other observed trends:**

Token entropy for `masked_normalize` rose sharply to 0.681 nats by the final step compared to 0.158 for `masked_mean`, indicating the policy failed to converge on a consistent reasoning format — consistent with the noisier gradient signal introduced by length-dependent loss scaling. The gradient norm was also higher for `masked_normalize` (10.56 vs 6.78), reflecting larger and more variable per-example losses, and average response length was paradoxically shorter (214.8 vs 234.5 tokens), suggesting the length incentive did not promote richer reasoning but instead destabilised training.