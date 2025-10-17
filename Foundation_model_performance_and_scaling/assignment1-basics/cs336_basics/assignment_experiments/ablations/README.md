# Transformer Ablation Experiments

This directory contains implementations and experiments for understanding the impact of various Transformer architectural components through systematic ablations.

## Overview

The ablation experiments help answer fundamental questions about Transformer design:

1. **Layer Normalization Ablation** - How important is RMSNorm for training stability?
2. **Pre-norm vs. Post-norm** - Which normalization placement works better?
3. **Position Embeddings** - Can transformers work without explicit position information (NoPE)?
4. **SwiGLU vs. SiLU** - Does gating in feed-forward networks matter?

## Files

- **`modified_transformer_block.py`** - Flexible Transformer block supporting all ablations
- **`modified_transformer_lm.py`** - Configurable language model for experiments
- **`run_ablation.py`** - Main experiment runner
- **`README.md`** - This file

## Quick Start

### Run All Ablations

```bash
# Baseline (standard pre-norm Transformer with SwiGLU and RoPE)
uv run python cs336_basics/assignment_experiments/ablations/run_ablation.py --ablation baseline

# No layer normalization
uv run python cs336_basics/assignment_experiments/ablations/run_ablation.py --ablation no_layer_norm

# Post-norm architecture
uv run python cs336_basics/assignment_experiments/ablations/run_ablation.py --ablation post_norm

# No position embeddings (NoPE)
uv run python cs336_basics/assignment_experiments/ablations/run_ablation.py --ablation no_position_emb

# SiLU FFN instead of SwiGLU
uv run python cs336_basics/assignment_experiments/ablations/run_ablation.py --ablation silu_ffn
```

### Run with Custom Settings

```bash
# No layer norm with lower learning rate (recommended)
uv run python cs336_basics/assignment_experiments/ablations/run_ablation.py \
    --ablation no_layer_norm \
    --learning-rate 1e-4

# Faster iteration with smaller batch
uv run python cs336_basics/assignment_experiments/ablations/run_ablation.py \
    --ablation baseline \
    --batch-size 64

# Low-resource mode (40M tokens instead of 328M)
uv run python cs336_basics/assignment_experiments/ablations/run_ablation.py \
    --ablation baseline \
    --low-resource
```

## Experiment Descriptions

### 1. Baseline

**Standard pre-norm Transformer with all modern components:**
- Pre-normalization (RMSNorm before sublayers)
- SwiGLU feed-forward networks
- RoPE position embeddings
- d_ff ≈ (8/3) × d_model

This serves as the reference point for all ablations.

### 2. No Layer Normalization (`no_layer_norm`)

**Removes all RMSNorm layers from the model.**

Purpose: Investigate the importance of layer normalization for training stability.

Expected observations:
- Training may become unstable at the baseline learning rate (3e-4)
- May require significantly lower learning rate (e.g., 1e-4 or lower)
- Gradients may explode or vanish without normalization
- Final performance likely to be degraded

Key question: Can stability be recovered with careful learning rate tuning?

### 3. Post-norm Architecture (`post_norm`)

**Switches from pre-norm to post-norm:**

Pre-norm (baseline):
```
z = x + Attention(RMSNorm(x))
y = z + FFN(RMSNorm(z))
```

Post-norm (this ablation):
```
z = RMSNorm(x + Attention(x))
y = RMSNorm(z + FFN(z))
```

Purpose: Compare modern pre-norm (consensus choice) vs. original Transformer post-norm.

Expected observations:
- Post-norm was the original design but pre-norm became standard
- Pre-norm typically trains more stably
- Post-norm may require different hyperparameters
- Different gradient flow properties

### 4. No Position Embeddings (`no_position_emb`)

**Removes RoPE, providing no explicit position information.**

Purpose: Test whether causal attention alone can encode position information.

Background:
- Decoder-only transformers with causal masks theoretically can infer positions
- No Position Embedding (NoPE) has been studied in recent work
- Attention patterns may encode relative positions implicitly

Expected observations:
- Model may still learn reasonable language modeling
- Performance likely degraded compared to RoPE
- Interesting test of implicit vs. explicit position encoding

### 5. SiLU FFN (`silu_ffn`)

**Replaces SwiGLU with simpler SiLU activation:**

SwiGLU (baseline):
```
FFN(x) = W2 × (SiLU(W1 × x) ⊙ (W3 × x))
Parameters: 3 weight matrices, d_ff ≈ (8/3) × d_model
```

SiLU (this ablation):
```
FFN(x) = W2 × SiLU(W1 × x)
Parameters: 2 weight matrices, d_ff = 4 × d_model
```

Purpose: Evaluate the benefit of gating in feed-forward networks.

Note: d_ff is adjusted to approximately match parameter counts:
- SwiGLU: 3 × d_model × d_ff (with d_ff ≈ (8/3) × d_model) ≈ 8 × d_model²
- SiLU: 2 × d_model × d_ff (with d_ff = 4 × d_model) = 8 × d_model²

## Configuration

All experiments use the same base configuration:

```
Model:
  - d_model: 512
  - num_layers: 4
  - num_heads: 16
  - context_length: 256
  - vocab_size: 10,000

Training:
  - Dataset: TinyStories
  - Total tokens: ~328M (or 40M in low-resource mode)
  - Batch size: 32 (default, configurable)
  - Learning rate: 3e-4 (default, should tune for ablations)
  - Warmup: 1% of total iterations
  - Cosine decay to 0.1 × max_lr

Optimizer:
  - AdamW (β1=0.9, β2=0.999)
  - Weight decay: 0.1
  - Gradient clipping: 1.0
```

## Expected Results & Analysis

### Baseline
- Should achieve validation loss ≤ 1.45
- Serves as reference for comparisons

### No Layer Norm
- **Critical:** Likely requires lower learning rate (try 1e-4, 5e-5, or lower)
- May not converge at all with standard learning rate
- If it converges, expect higher final loss
- **Analysis:** Demonstrates importance of normalization for gradient flow

### Post-norm
- May converge but potentially less stably than pre-norm
- Might require adjusted learning rate
- **Analysis:** Shows why pre-norm became the modern standard

### NoPE (No Position Embeddings)
- Should still converge (causal mask provides some position info)
- Expect degraded performance vs. RoPE
- **Analysis:** Quantifies benefit of explicit position encodings

### SiLU FFN
- Should converge normally (both are valid architectures)
- Performance difference shows benefit of gating mechanism
- **Analysis:** Evaluates importance of GLU-style gating

## Output Structure

Each experiment creates:

```
cs336_basics/basics/runs/ablations/{ablation_type}/
├── config.json                 # Experiment configuration
├── checkpoints/
│   ├── best_model.pt          # Best validation checkpoint
│   ├── final_checkpoint.pt    # Final checkpoint
│   └── checkpoint_*.pt        # Periodic checkpoints
├── metrics.csv                # Training/validation metrics
├── metrics.json               # Metrics in JSON format
├── summary.json               # Final statistics
├── loss_curves.png            # Visualization
└── lr_schedule.png            # Learning rate schedule
```

## Parameter Counts

Approximate trainable parameters for each ablation (may vary slightly):

| Ablation | Parameters | Notes |
|----------|-----------|-------|
| Baseline | ~22.7M | Standard configuration |
| No Layer Norm | ~22.5M | Slightly fewer (no norm parameters) |
| Post-norm | ~22.7M | Same as baseline |
| NoPE | ~22.7M | Same (RoPE has no parameters) |
| SiLU FFN | ~22.7M | Matched to SwiGLU |

## Tips for Running Experiments

### 1. No Layer Norm Experiment

The no layer norm experiment is most likely to fail. Try a learning rate sweep:

```bash
# Try decreasing learning rates
for lr in 1e-4 5e-5 1e-5; do
    uv run python cs336_basics/assignment_experiments/ablations/run_ablation.py \
        --ablation no_layer_norm \
        --learning-rate $lr \
        --output-dir cs336_basics/basics/runs/ablations/no_layer_norm_lr_${lr}
done
```

### 2. Quick Testing

For rapid iteration, use smaller token budget and batch size:

```bash
uv run python cs336_basics/assignment_experiments/ablations/run_ablation.py \
    --ablation baseline \
    --total-tokens 10000000 \
    --batch-size 16
```

### 3. Comparing Results

Use the summary.json files to compare final metrics:

```bash
# View validation losses
for abl in baseline no_layer_norm post_norm no_position_emb silu_ffn; do
    echo -n "$abl: "
    jq '.statistics.val_loss.best' \
        cs336_basics/basics/runs/ablations/$abl/summary.json
done
```

## Analysis Script (Coming Soon)

A visualization script will be added to compare all ablations:

```bash
python cs336_basics/assignment_experiments/ablations/analyze_ablations.py
```

This will generate comparative plots and analysis.

## Related Files

- Model components: `cs336_basics/transformer_training/model/`
- Training infrastructure: `cs336_basics/basics/run_experiment.py`
- Data loading: `cs336_basics/data/data_loader.py`

## References

- Layer normalization: Ba et al. (2016)
- Pre-norm vs. Post-norm: Xiong et al. (2020), "On Layer Normalization in the Transformer Architecture"
- RoPE: Su et al. (2021), "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- NoPE: Kazemnejad et al. (2023), "The Impact of Positional Encoding on Length Generalization in Transformers"
- SwiGLU: Shazeer (2020), "GLU Variants Improve Transformer"
