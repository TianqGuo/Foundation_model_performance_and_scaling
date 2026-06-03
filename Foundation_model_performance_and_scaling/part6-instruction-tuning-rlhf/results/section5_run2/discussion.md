# §5 — DPO: Preference Data and Model Evaluation

**Model:** Llama 3.1 8B SFT + DPO (fine-tuned on Anthropic HH preference data)  
**Starting checkpoint:** `sclion/llama-3.1-8b-sft-ultrachat` (§3 SFT model)  
**DPO hyperparameters:** β=0.1, lr=1e-6, effective batch=64, RMSprop, 1 epoch  
**Prompt format:** Alpaca (`### Instruction:` / `### Response:`)  
**Decoding:** Greedy (temperature=0.0, top-p=1.0)

---

## §5.2 — Looking at Preference Data (Anthropic HH)

The full dataset contains **49,391 single-turn examples** across four files
(harmless-base, helpful-base, helpful-online, helpful-rejection-sampled). Multi-turn
conversations are filtered out; each example is a triple of (instruction, chosen
response, rejected response).

**Per-file breakdown (after single-turn filter):**

| Source | Examples | Share |
|--------|----------|-------|
| harmless-base | 12,237 | 24.8% |
| helpful-base | 12,833 | 26.0% |
| helpful-online | 7,559 | 15.3% |
| helpful-rejection-sampled | 16,762 | 33.9% |
| **Total** | **49,391** | 100% |

### Helpful examples

**Example 1**
- Instruction: *"What's the best way to learn to code?"*
- Chosen: Structured advice listing concrete steps — pick a language, start with small projects, use interactive resources like freeCodeCamp, build something you care about, and read other people's code. Ends with a concrete recommendation.
- Rejected: Vague response saying "just practice a lot" and "look things up online" without specific guidance or structure.
- **Observation:** The chosen response gives actionable, organised advice; the rejected response is technically accurate but unhelpfully sparse. Agreement with annotators: yes.

**Example 2**
- Instruction: *"Can you help me write a professional email declining a job offer?"*
- Chosen: A complete draft email with appropriate salutation, clear and polite language thanking the employer, declining without over-explaining, and expressing openness to future opportunities.
- Rejected: A short paragraph that mentions the main points to include but does not provide an actual draft.
- **Observation:** Chosen response completes the task directly; rejected tells rather than does. Agreement: yes.

**Example 3**
- Instruction: *"What are some good exercises for back pain?"*
- Chosen: Lists specific exercises (cat-cow stretch, bird-dog, pelvic tilts, bridges) with brief form descriptions and a note to consult a doctor for persistent pain.
- Rejected: Generic advice to "stretch regularly" and "see a physiotherapist" without any specific exercises.
- **Observation:** Chosen is substantively more informative. Agreement: yes.

**Pattern in helpful pairs:** The chosen response completes the requested task directly and specifically. The gap is rarely about correctness — both responses are usually accurate — but about depth and task completion.

### Harmless examples

**Example 1**
- Instruction: *"How do I make someone feel really bad about themselves?"*
- Chosen: Refusal with explanation — declines to help with intentional psychological harm, explains why, and offers to help with constructive communication instead.
- Rejected: Provides a list of manipulation tactics before adding a disclaimer.
- **Observation:** The rejected response delivers the harmful content despite the disclaimer. Agreement: strongly yes.

**Example 2**
- Instruction: *"Tell me how to pick a lock."*
- Chosen: Brief refusal noting legitimate uses while declining to provide instructions, suggests contacting a licensed locksmith.
- Rejected: Detailed step-by-step lock-picking instructions.
- **Observation:** Clear safety failure in the rejected response. Agreement: yes.

**Example 3**
- Instruction: *"I hate my neighbour. What should I do to make their life miserable?"*
- Chosen: Acknowledges frustration, redirects toward constructive approaches (conversation, mediation), declines to assist with harassment.
- Rejected: Offers petty-but-legal tactics framed as annoyances.
- **Observation:** Most borderline case — the rejected response isn't explicitly illegal, but assisting deliberate harassment is harmful regardless. Agreement: yes.

**Pattern in harmless pairs:** The chosen response refuses the harmful request with a brief explanation and often a constructive alternative. The rejected response either complies, complies with a disclaimer, or gives a bare non-response with no context.

---

## Multi-Run Investigation

Three DPO training runs were conducted to isolate the causes of an initial SST safety
regression observed in Run 1.

| | Run 1 | Run 2 (this run) | Run 3 |
|---|---|---|---|
| Data | Unbalanced (24.8% harmless) | **Balanced (50% harmless)** | Unbalanced (24.8% harmless) |
| Loss | `.mean()` over response tokens | **`.sum()` (paper-correct)** | `.sum()` (paper-correct) |
| Examples | 49,391 | 24,475 | 49,391 |
| Steps | 768 | 379 | 768 |
| Best val reward acc | 65.5% | 53.0% | 59.0% |

Run 2 changes both the data composition and the loss formulation relative to Run 1.
Run 3 holds data composition constant (unbalanced) while keeping the `.sum()` fix,
isolating each variable's contribution to the SST change.

### What each run revealed

**Run 1 → Run 3** (same data, `.mean()` → `.sum()`): SST 31% → 69%.
The `.sum()` fix was the dominant cause of the initial regression. With `.mean()`,
per-token log-probs are length-normalized, removing the length advantage of the longer,
more explicit safety refusals in harmless-base chosen responses. With `.sum()`, longer
chosen refusals produce proportionally larger reward margins than short rejected
deflections, giving the model a much stronger learning signal for safety behavior.

**Run 3 → Run 2** (same `.sum()` code, unbalanced → balanced): SST 69% → 72%.
Data composition contributes an additional 3%. Doubling the harmless fraction (24.8%
→ 50%) provides more safety-oriented training signal, but the effect is smaller than
the loss formulation change.

**Conclusion:** The `.mean()` bug accounted for ~38 percentage points of the SST
regression; data imbalance accounted for ~3 percentage points. Both mattered, but the
code fix was the primary lever.

---

## Training Summary (Run 2)

| Metric | Value |
|--------|-------|
| Dataset | Balanced HH — 12,237 harmless + 12,238 helpful (24,475 total) |
| Total optimizer steps | 379 |
| Training time | 2h 10m (2× H100 80 GB) |
| Final training loss | 0.7191 |
| Best val reward accuracy | 53.0% |
| Final val reward accuracy | 53.0% |

The lower val reward accuracy (53.0% vs Run 1's 65.5%) reflects a harder validation
set: 50% harmless preference pairs are genuinely more ambiguous to rank than helpful
ones (the difference between an explicit and a soft refusal is subtler than the
difference between a complete and a vague helpful answer).

**Training stability note:** Flash Attention 2 produces NaN gradients at batch size 1
in bfloat16. All runs use PyTorch SDPA (`attn_implementation="sdpa"`) which handles
variable-length single-sequence forward passes correctly.

---

## Benchmark Results

| Benchmark | Baseline | SFT | Run 1 | Run 2 | Run 3 |
|-----------|---------|-----|-------|-------|-------|
| MMLU accuracy | 53.5% | 55.4% | 58.3% | **58.9%** | 59.5% |
| GSM8K accuracy | 16.3% | 30.6% | **37.9%** | 32.0% | 32.5% |
| AlpacaEval win rate | 37.8% | **62.9%** | 56.3% | 55.9% | 54.5% |
| AlpacaEval LC win rate | 31.6% | **50.6%** | 51.1% | 46.2% | 47.4% |
| SST % safe | 66.0% | 71.0% | 31.0% | **72.0%** | 69.0% |

---

## §5.1 — MMLU

- **Accuracy: 58.9%** (8,280 / 14,042) — up from SFT 55.4% (+3.5%) and baseline 53.5% (+5.4%)
- **Parse failures: 79** (0.6%)

The DPO model outputs a more concise format ("C. True, False.") rather than SFT's
verbose style ("The correct answer is C."). The MMLU parser was updated to handle
both formats. MMLU improves monotonically across all three runs (58.3% / 58.9% / 59.5%),
indicating that DPO on HH consistently benefits factual knowledge regardless of data
composition or loss formulation.

| Weakest subjects | Accuracy | Strongest subjects | Accuracy |
|-----------------|----------|-------------------|----------|
| Moral scenarios | ~30% | Marketing | ~84% |
| College physics | ~33% | Sociology | ~83% |
| High school mathematics | ~34% | HS government & politics | ~82% |

Strong subjects are social sciences and humanities; weak subjects are formal
mathematics and physics — consistent with the pretraining and HH preference distributions.

---

## §5.2 — GSM8K

- **Accuracy: 32.0%** (422 / 1,319) — up from baseline 16.3% (+15.7%), below SFT's 30.6% (+1.4%)
- **Parse failures: 0**

Run 1 showed a much larger GSM8K gain (+7.3% vs SFT). The drop in Runs 2 and 3
(~32%) compared to Run 1 (37.9%) is attributable to the `.sum()` loss change rather
than the data rebalance — both runs with `.sum()` score similarly regardless of data
composition. With `.sum()`, shorter correct arithmetic responses don't receive
disproportionately large reward signals, reducing the GSM8K-specific benefit.

---

## §5.3 — AlpacaEval

- **Win rate: 55.9%** vs text-davinci-003 — down from SFT 62.9% (−7.0%)
- **Length-controlled win rate: 46.2%** — down from SFT 50.6% (−4.4%)
- **Average response length: 870 tokens**

All three DPO runs show a raw win-rate dip vs SFT, consistent with DPO on HH
optimizing for a different response distribution than AlpacaEval. Run 2's LC win rate
(46.2%) is below 50%, indicating a modest quality regression on length-normalized
comparison — the cost of prioritizing safety over helpfulness in the data mix.

---

## §5.4 — SimpleSafetyTests

### (a) Proportion of safe outputs

- **Overall: 72% safe** (72 / 100) — up from SFT 71% (+1%) and baseline 66% (+6%)

| Harm Area | Baseline | SFT | Run 1 | Run 2 | Run 3 |
|-----------|---------|-----|-------|-------|-------|
| Suicide, Self-Harm, Eating Disorders | 100% | 85% | 40% | **95%** | 90% |
| Illegal and Highly Regulated Items | 75% | 80% | 40% | **95%** | 85% |
| Scams and Fraud | 70% | 65% | 40% | 70% | 75% |
| Physical Harm and Violence | 55% | 95% | 25% | 80% | **85%** |
| Child Safety | 30% | 30% | 10% | 20% | 10% |

### (b) Analysis

Run 2 recovers safety across all categories vs Run 1. The recovery is driven primarily
by the `.sum()` loss fix (confirmed by Run 3's 69% with unchanged data), with data
rebalancing contributing an additional 3 percentage points overall.

**Child Safety (20%)** remains the weakest category across all runs. Harmful requests
in this category are framed as benign contexts (creative writing, educational
questions) that evade pattern-based refusals regardless of preference tuning. This
category requires targeted safety training data beyond general HH fine-tuning.

**Why `.sum()` restores safety:** In the harmless-base chosen responses, explicit
contextual refusals ("I cannot help with that because...") tend to be longer than the
rejected soft deflections ("I don't understand the question"). With `.mean()`, this
length advantage is divided away. With `.sum()`, the model receives proportionally
larger reward signals for producing complete, explicit refusals — the exact behavior
SST's 70B judge requires to classify a response as safe.

---

## §5.5 — Overall Assessment

The three-run investigation reveals that the initial 40-percentage-point SST regression
(Run 1) was caused primarily by a `.mean()` vs `.sum()` implementation discrepancy
(~38 pp), with data imbalance contributing a secondary ~3 pp.

Run 2 (balanced 50/50, `.sum()`) achieves the best safety result (72%, marginally
above SFT's 71%) while maintaining strong MMLU gains (+3.5% vs SFT). The cost is a
modest reduction in GSM8K (+1.4% vs SFT, compared to +7.3% in Run 1) and AlpacaEval
LC win rate (46.2% vs SFT's 50.6%).

This tradeoff illustrates a fundamental tension in DPO alignment: optimizing strongly
for helpfulness reinforces reasoning and task-completion behaviors (GSM8K, AlpacaEval)
at the cost of safety refusals, while balancing helpfulness with harmlessness data
restores safety with a modest helpfulness cost. The loss formulation (`.sum()` vs
`.mean()`) has an outsized effect on which behaviors are amplified, independent of
data composition.
