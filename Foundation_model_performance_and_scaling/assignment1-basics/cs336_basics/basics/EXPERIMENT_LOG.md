# Experiment Log - CS336 Assignment 1

This document tracks all experiments conducted for Assignment 1, Section 7.

## Log Format

For each experiment, document:
- **Date**: When the experiment was run
- **Experiment Name**: Identifier used in tracking system
- **Hypothesis/Motivation**: What are you testing and why?
- **Configuration Changes**: What changed from baseline?
- **Results**: Key metrics (loss, perplexity, etc.)
- **Loss Curves**: Links to generated plots
- **Observations**: What did you learn?
- **Next Steps**: What to try next based on results

---

## Experiment 1: Baseline Model (17M Parameters)

**Date**: [To be filled]

**Experiment Name**: `baseline_17m`

**Hypothesis/Motivation**:
Establish a baseline for the 17M parameter model on TinyStories. This will serve as the reference point for all subsequent ablations and modifications.

**Configuration**:
- Model: 12 layers, 768 hidden dim, 12 heads, 3072 FFN dim (~17M params)
- Training: 10,000 iterations, batch size 64, context length 512
- Learning rate: 6e-4 (peak), 6e-5 (min), 500-step warmup, cosine decay
- Optimizer: AdamW (β1=0.9, β2=0.999, weight decay=0.1)
- Regularization: Gradient clipping at 1.0
- Dataset: TinyStories (10K vocabulary)

**Results**:
- Training loss (final): [To be filled]
- Validation loss (final): [To be filled]
- Best validation loss: [To be filled] at step [X]
- Training time: [To be filled] hours
- Throughput: [To be filled] tokens/sec

**Loss Curves**:
- See: `assignment_experiments/runs/baseline_17m/loss_curves.png`
- By gradient steps: [Link or screenshot]
- By wallclock time: [Link or screenshot]

**Observations**:
[To be filled after running]
- Did the loss converge smoothly?
- Any signs of overfitting?
- How does validation loss compare to training loss?
- Is throughput satisfactory?

**Next Steps**:
[To be filled]

---

## Experiment 2: [Example Template]

**Date**: YYYY-MM-DD

**Experiment Name**: `experiment_name`

**Hypothesis/Motivation**:
What are you testing and why? What do you expect to happen?

**Configuration Changes**:
List only what changed from baseline:
- Parameter X: changed from Y to Z
- Added/removed component ABC

**Results**:
- Training loss (final): X.XX
- Validation loss (final): X.XX
- Best validation loss: X.XX at step NNNN
- Training time: X.XX hours
- Comparison to baseline: +/- X% validation loss

**Loss Curves**:
- Path: `assignment_experiments/runs/experiment_name/loss_curves.png`
- Notable patterns: [Describe any interesting behavior]

**Observations**:
- What happened?
- Did it match your hypothesis?
- Any unexpected behaviors?
- What insights did you gain?

**Next Steps**:
What should be tried next based on these results?

---

## Experiment Summary Table

| Exp # | Name | Date | Key Change | Val Loss (Final) | Val Loss (Best) | Train Time (hrs) | Notes |
|-------|------|------|------------|------------------|-----------------|------------------|-------|
| 1 | baseline_17m | - | Baseline | - | - | - | Reference model |
| 2 | - | - | - | - | - | - | - |
| 3 | - | - | - | - | - | - | - |

---

## Key Findings and Insights

### Learning Rate Sensitivity
[Document findings about learning rate effects]

### Architecture Ablations
[Document findings about architectural choices]

### Training Dynamics
[Document findings about training behavior, convergence, etc.]

### Efficiency Observations
[Document findings about throughput, memory usage, etc.]

---

## Recommendations

Based on all experiments, what are the key recommendations for:

1. **Model Architecture**:
   - [What architectural choices work best?]

2. **Hyperparameters**:
   - [What hyperparameter settings work best?]

3. **Training Procedure**:
   - [What training procedures work best?]

4. **Future Work**:
   - [What should be explored next?]

---

## Appendix: Experiment Configurations

### How to Reproduce

Each experiment can be reproduced with:

```bash
uv run python -m assignment_experiments.run_experiment \
    --name <experiment_name> \
    --config assignment_experiments/configs/<config_file>.json
```

Configuration files are version controlled in `assignment_experiments/configs/`.

### Computing Environment

Document the computing environment used:
- Hardware: [GPU/CPU model]
- Software: PyTorch version, CUDA version
- System: OS details

This helps explain performance differences between runs.
