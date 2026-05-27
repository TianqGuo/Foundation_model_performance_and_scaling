# §8.6 — Prompt Ablation: Discussion

**Best setting:** `question_only` — peak accuracy 71.1% vs 54.6%, final accuracy 69.3% vs 52.5%.

---

## Results summary

| Metric | r1_zero | question_only |
|--------|---------|---------------|
| Starting accuracy | 36.0% | 60.7% |
| Peak accuracy | 54.6% @ step 90 | 71.1% @ step 180 |
| Final accuracy | 52.5% | 69.3% |
| Entropy (first → final) | 0.238 → 0.035 | 0.102 → 0.095 |
| Grad norm (max) | 7,045,248 | 0.186 |
| Clip fraction (final) | 62.2% | 0.001 |
| Response length (final) | ~226 tokens | ~530 tokens |

---

## Why question_only outperforms so decisively

### Pretraining distribution alignment

Qwen 2.5 Math 1.5B was continually pretrained on high-quality synthetic math data in natural format: a question followed by a structured solution, without any special reasoning tags. The `question_only` prompt presents the input in exactly this way — a raw question — so the model's pretrained weights already encode a strong prior for producing correct, well-structured mathematical solutions.

The `r1_zero` prompt wraps the problem in a conversational template and requires the model to produce output within `<think>…</think><answer>…</answer>` tags. This format does not appear in the model's pretraining distribution. Before RL can improve mathematical reasoning, it must first teach the policy to reliably produce the tag structure — these two objectives compete during early training, explaining the lower starting accuracy (36.0% vs 60.7%) and the high format rate variance in early `r1_zero` steps.

### Starting accuracy gap (+24.7 pt)

The `question_only` model starts RL already at 60.7% accuracy — a strong base that reflects how well the pretrained model handles math in its natural format. The `r1_zero` model starts at 36.0% because many of those early failures are format failures rather than reasoning failures: the model generates content but outside the expected tag structure, causing the reward function to return 0.

### Training stability (grad norm: 0.186 vs 7 million)

Because `question_only` begins close to the pretrained distribution, the RL policy barely needs to move to improve further. Each gradient update is small and well-conditioned — maximum grad norm of 0.186, clip fraction near 0 (0.001) throughout all 200 steps. The policy is nudging itself toward better reasoning within a familiar output space.

`r1_zero` must make large policy changes to acquire the tag format simultaneously with improving reasoning. These large updates produce the gradient explosion seen in later steps (7 million max grad norm), and the high clip fraction (62.2%) reflects how far the policy has drifted from the rollout distribution within each set of 8 gradient updates.

### Entropy: collapse vs stability

`r1_zero` entropy collapses from 0.238 to 0.035 — the policy rapidly converges on a narrow mode (the tag structure) and loses diversity. `question_only` entropy starts lower (0.102) and stays flat throughout (0.095 final) — the model was already operating in a well-calibrated, confident regime for natural math output and simply refines within it. The stability of the entropy trajectory under `question_only` is a direct reflection of the pretraining alignment.

### Response length and format

`question_only` generates ~530-token responses — detailed multi-step solutions written as natural mathematical prose, matching the model's pretraining style. `r1_zero` responses are ~226 tokens, constrained by the tag structure that limits how much reasoning can naturally flow within the format. The longer responses in `question_only` likely contribute to higher accuracy: more working space means the model can resolve ambiguous reasoning steps that shorter responses truncate.

### Peak timing and convergence

`question_only` peaks at step 180 (near the end of training) while `r1_zero` peaks at step 90 and then declines. Starting close to the optimum, `question_only` makes steady, stable progress throughout. `r1_zero` makes fast initial progress (the policy quickly learns the format), but then over-optimises and regresses — the aggressive gradient updates in the off-policy setting push the already format-locked policy past a good optimum.

---

## Broader implication: RL and pretraining distribution

This result illustrates a general principle: **the effectiveness of RL fine-tuning is bounded by the distance between the base model's pretrained distribution and the target behaviour**.

When the prompt aligns with pretraining:
- The base model starts near the reward-maximising policy
- RL updates are small, stable, and targeted
- The model can reach a higher performance ceiling with less instability

When the prompt does not align with pretraining:
- The base model must simultaneously learn a new output structure and improve task performance
- RL updates are large and noisy, risking instability and catastrophic gradient behaviour
- The model's ceiling is lower because part of the policy capacity is consumed by format learning

This is the same principle that motivates SFT warm-starting before RL in production systems (e.g., InstructGPT, DeepSeek R1): SFT first aligns the model to the target output format, then RL improves the quality within that format. Here, the Qwen 2.5 Math pretraining serves as an implicit SFT warm-start for `question_only` — the model already "knows" the output format, leaving RL free to focus entirely on mathematical correctness.