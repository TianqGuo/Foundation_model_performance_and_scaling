# Ablation Experiments - Quick Start Guide

## Overview

This guide provides step-by-step instructions for running the Transformer ablation experiments required by the assignment.

## Prerequisites

- Tokenized TinyStories dataset (`cs336_basics/artifacts/datasets/tinystories_*_tokens.npy`)
- GPU recommended (experiments will take ~1 hour each on H100, longer on other GPUs)
- Can also run with `--low-resource` flag for 40M tokens instead of 328M

## Running Experiments

### 1. Baseline (Reference Model)

Run the baseline first to establish reference performance:

```bash
uv run python cs336_basics/assignment_experiments/ablations/run_ablation.py \
    --ablation baseline \
    --learning-rate 3e-4 \
    --batch-size 32
```

**Expected**: Val loss ≤ 1.45, similar to batch_8 experiment

### 2. Layer Norm Ablation (Problem: layer_norm_ablation)

**Important**: This will likely fail at the default learning rate!

#### Step 1: Try default LR (expected to fail/diverge)

```bash
uv run python cs336_basics/assignment_experiments/ablations/run_ablation.py \
    --ablation no_layer_norm \
    --learning-rate 3e-4 \
    --batch-size 32
```

**Expected**: Training instability, loss divergence

#### Step 2: Find stable learning rate

```bash
# Try progressively lower learning rates
for lr in 1e-4 5e-5 2e-5 1e-5; do
    uv run python cs336_basics/assignment_experiments/ablations/run_ablation.py \
        --ablation no_layer_norm \
        --learning-rate $lr \
        --batch-size 32 \
        --output-dir cs336_basics/basics/runs/ablations/no_layer_norm_lr_$lr
done
```

**Expected**: Stability at some lower LR (likely 1e-5 to 5e-5), but higher final loss than baseline

**Deliverables**:
- Learning curve showing instability at 3e-4
- Learning curve at best found learning rate
- Commentary: RMSNorm is critical for stable training; without it, much lower LR is needed but performance degrades

### 3. Post-norm Ablation (Problem: pre_norm_ablation)

```bash
uv run python cs336_basics/assignment_experiments/ablations/run_ablation.py \
    --ablation post_norm \
    --learning-rate 3e-4 \
    --batch-size 32
```

**Expected**: Trains but potentially less stable than pre-norm baseline

**Deliverable**:
- Learning curve comparing post-norm vs baseline (pre-norm)
- Note: Pre-norm became standard because it's more stable

### 4. No Position Embeddings (Problem: no_pos_emb)

```bash
uv run python cs336_basics/assignment_experiments/ablations/run_ablation.py \
    --ablation no_position_emb \
    --learning-rate 3e-4 \
    --batch-size 32
```

**Expected**: Converges but with degraded performance vs. RoPE

**Deliverable**:
- Learning curve comparing NoPE vs RoPE (baseline)
- Shows benefit of explicit position encodings

### 5. SwiGLU vs SiLU (Problem: swiglu_ablation)

```bash
uv run python cs336_basics/assignment_experiments/ablations/run_ablation.py \
    --ablation silu_ffn \
    --learning-rate 3e-4 \
    --batch-size 32
```

**Expected**: Trains normally, performance comparison shows benefit of gating

**Deliverables**:
- Learning curve comparing SiLU vs SwiGLU (baseline)
- Commentary on gating mechanism benefit

## Running All Experiments (Sequential)

```bash
#!/bin/bash
# Run all ablations sequentially

# Baseline
uv run python cs336_basics/assignment_experiments/ablations/run_ablation.py \
    --ablation baseline

# No layer norm experiments (multiple LRs)
for lr in 3e-4 1e-4 5e-5 2e-5; do
    uv run python cs336_basics/assignment_experiments/ablations/run_ablation.py \
        --ablation no_layer_norm \
        --learning-rate $lr \
        --output-dir cs336_basics/basics/runs/ablations/no_layer_norm_lr_$lr
done

# Post-norm
uv run python cs336_basics/assignment_experiments/ablations/run_ablation.py \
    --ablation post_norm

# No position embeddings
uv run python cs336_basics/assignment_experiments/ablations/run_ablation.py \
    --ablation no_position_emb

# SiLU FFN
uv run python cs336_basics/assignment_experiments/ablations/run_ablation.py \
    --ablation silu_ffn

# Analyze results
uv run python cs336_basics/assignment_experiments/ablations/analyze_ablations.py
```

## Low-Resource Mode (For Limited GPU Access)

If you have limited GPU resources, run with reduced token budget:

```bash
uv run python cs336_basics/assignment_experiments/ablations/run_ablation.py \
    --ablation baseline \
    --low-resource  # Uses 40M tokens instead of 328M
```

This is faster but validation loss targets should be adjusted (expect ~1.8 instead of 1.45).

## Analyzing Results

After running experiments, analyze and visualize:

```bash
uv run python cs336_basics/assignment_experiments/ablations/analyze_ablations.py
```

This generates:
- `ablation_comparison.png` - Comparative plots
- `ablation_analysis.txt` - Detailed analysis report

## Manual Analysis

View summary metrics:

```bash
# Compare validation losses
for abl in baseline no_layer_norm post_norm no_position_emb silu_ffn; do
    echo -n "$abl: "
    jq '.statistics.val_loss.best' \
        cs336_basics/basics/runs/ablations/$abl/summary.json 2>/dev/null || echo "not run"
done
```

View training curves:

```bash
# All experiments save loss curves to:
cs336_basics/basics/runs/ablations/{ablation_name}/loss_curves.png
```

## Expected Outcomes Summary

| Ablation | Expected Behavior | Val Loss vs Baseline |
|----------|------------------|---------------------|
| Baseline | Stable training | 1.32 (reference) |
| No Layer Norm | Unstable at 3e-4, needs ~1e-5 | +15-30% degradation |
| Post-norm | Trains, potentially less stable | +0-10% degradation |
| NoPE | Trains normally | +5-15% degradation |
| SiLU FFN | Trains normally | +0-5% degradation |

## Time Estimates

On H100 GPU with 328M tokens:
- Each experiment: ~1-1.5 hours
- Total (baseline + 4 ablations + LR sweep): ~6-8 hours

On consumer GPUs (RTX 3090/4090):
- Each experiment: ~2-4 hours
- Total: ~12-20 hours

With `--low-resource` (40M tokens):
- Each experiment: ~10-20 minutes
- Total: ~1-2 hours

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
--batch-size 16  # or 8

# Or use low-resource mode
--low-resource
```

### Training Diverges (Loss → NaN)

This is expected for `no_layer_norm` at high LR. Try lower LR:

```bash
--learning-rate 1e-5
```

### Import Errors

Make sure you're running from the project root and using `uv run`:

```bash
cd /path/to/assignment1-basics
uv run python cs336_basics/assignment_experiments/ablations/run_ablation.py ...
```

## Deliverables Checklist

For each ablation, you need:

- [ ] **Layer Norm Ablation**
  - [ ] Learning curve at 3e-4 (showing instability)
  - [ ] Learning curve at best stable LR
  - [ ] Commentary on RMSNorm importance

- [ ] **Pre-norm vs Post-norm**
  - [ ] Comparative learning curves
  - [ ] Commentary on difference

- [ ] **No Position Embeddings**
  - [ ] Comparative learning curves (NoPE vs RoPE)
  - [ ] Commentary on position encoding benefit

- [ ] **SwiGLU vs SiLU**
  - [ ] Comparative learning curves
  - [ ] Commentary on gating benefit

All curves and analysis will be automatically saved to the experiment directories.

## Next Steps

After running ablations, you can:
1. Use `analyze_ablations.py` to generate comparative visualizations
2. Write up findings based on the results
3. Compare to baseline from earlier experiments
4. Consider additional experiments if interesting patterns emerge

Good luck!
