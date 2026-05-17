# Section 3: Zero-Shot MATH Baseline — Analysis

## Setup

- **Model:** Qwen 2.5 Math 1.5B Base (`Qwen/Qwen2.5-Math-1.5B`)
- **Prompt:** r1_zero (`cs336_alignment/prompts/r1_zero.prompt`)
- **Reward function:** `cs336_alignment.drgrpo_grader.r1_zero_reward_fn`
- **Local smoke test:** 10 examples from GSM8K (`data/gsm8k/test.jsonl`)
- **Full evaluation:** 5K examples from MATH validation set (cluster only)
- **Raw results:** `zero_shot_eval.jsonl`

---

## (b) Category Breakdown and Root Cause Analysis

### Counts (10-example GSM8K smoke test)

| Category | Count | % |
|---|---|---|
| Correct (format=1, answer=1) | 0 | 0% |
| Format ok, wrong answer (format=1, answer=0) | 1 | 10% |
| No format (format=0, answer=0) | 9 | 90% |

### Format=0 cases: parser issue or model issue?

Two distinct failure modes were observed:

**Parser strictness (Examples 1, 2):** The model generates the correct *structure* but
uses `\n` between `</think>` and `<answer>` instead of a space. The grader requires the
exact string `"</think> <answer>"`, so `"</think>\n<answer>"` fails format check.

Example (GT: 18, response truncated):
```
... = $18. </think>\n<answer> <block>18 </block> </answer>
```
The model reached the correct answer and attempted the right format — rejected on a
whitespace technicality. This is a parser issue, not a reasoning failure.

**Base model behavior (Examples 3–7, 9–10):** The model ignores the r1_zero format
entirely, defaulting to patterns from its math pretraining. Two common sub-patterns:

- Generates Python code + `\boxed{}` output (Examples 3, 7, 10) — correct answer,
  wrong format. Examples 7 and 10 actually reach the correct answer (260 and 460
  respectively) inside `\boxed{}` but are counted as 0.
- Hallucinates unrelated problems mid-response (Examples 3, 4, 7, 10), continuing
  to generate new questions after the first one is answered.
- Generates inline prose with no tags at all (Examples 5, 6, 9).

### Format=1, wrong answer: what is going wrong?

Example 8 (GT: 160): The model produces valid `</think> <answer>...</answer>` tags but
hallucinates a completely different problem (file sizes, 200 GB). The answer content is
also prose rather than a number:
```
</think> <answer>We first calculate the size of the file after 40% of the process is
completed, which is <0.4*200=80GB</answer>
```
This is a base model issue: the model generates a plausible-looking math response to an
imagined problem. The format is technically compliant but the reasoning is disconnected
from the actual question.

### Summary

Most format=0 failures are **base model behavior**, not parser bugs. The base Qwen 2.5
Math 1.5B model was pretrained on math Q&A pairs that use `\boxed{}` for answers, so it
defaults to that format rather than the `<answer>` tags the r1_zero prompt requests. This
is consistent with what the requirements doc notes: *"the r1_zero prompt is not the best
choice... because of a mismatch between the prompt and how Qwen 2.5 Math 1.5B was
pretrained."*

The two parser-strictness failures (newline vs. space) suggest that relaxing the format
check slightly (e.g., treating `\s+` between `</think>` and `<answer>` as valid) would
recover some true positives, but this is intentionally strict to train format compliance
via RL.

---

## (c) Zero-Shot Baseline Performance

**Local smoke test (10 examples, GSM8K):** 0% end-to-end accuracy.

This is expected — see (b) for why. The majority of failures are format failures rather
than reasoning failures; several responses contain the correct answer in `\boxed{}` format.

**On the full MATH validation set (cluster):** To be filled in after cluster run.
Expected: low but non-zero accuracy, likely in the range reported for base models with
the r1_zero prompt before RL training. The `question_only` prompt is expected to perform
significantly better on this base model (per Liu et al., 2025), and will be compared
directly in a later section. RL training (Sections 5–6) will improve both format
compliance and answer correctness substantially.