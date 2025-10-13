# Learning Rate Tuning - Quick Reference

## Quick Start

### 1. Run the Sweep (GPU recommended)

```bash
# Full sweep with default learning rates
uv run python -m cs336_basics.assignment_experiments.learning_rate_sweep

# Low-resource mode (CPU/Apple Silicon)
uv run python -m cs336_basics.assignment_experiments.learning_rate_sweep --low-resource --device mps
```

### 2. Analyze Results

```bash
# Generate analysis and plots
uv run python -m cs336_basics.assignment_experiments.analyze_lr_sweep
```

### 3. Document Findings

Update `cs336_basics/basics/EXPERIMENT_LOG.md` with your results.

## Model Configuration (17M Parameters)

Based on assignment specifications:

```python
{
    "vocab_size": 10000,
    "context_length": 256,
    "d_model": 512,
    "d_ff": 1344,        # 8/3 * 512, multiple of 64
    "num_layers": 4,
    "num_heads": 16,
    "rope_theta": 10000.0,
}
```

**Parameter count calculation**:
- Embedding: `vocab_size * d_model = 10000 * 512 = 5.12M`
- Non-embedding: ~17M parameters (from 4 layers × architecture)

## Training Budget

### Full Mode (GPU)
- Total tokens: 327,680,000
- Batch size: 128
- Context length: 256
- Steps: 10,000
- Time per LR: ~30-40 minutes on H100
- **Total sweep time: ~4 hours** (7 learning rates)

### Low-Resource Mode (CPU/MPS)
- Total tokens: 40,000,000
- Batch size: 32
- Context length: 256
- Steps: 5,000
- Time per LR: ~36 minutes on M3 Max (MPS), ~1.5 hours on CPU
- **Total sweep time: ~4-10 hours**

## Default Learning Rates

The sweep tests 7 values spanning 2 orders of magnitude:

```python
learning_rates = [
    1e-4,   # Conservative
    3e-4,   # GPT-2 baseline
    6e-4,   # GPT-3 style
    1e-3,   # Aggressive
    3e-3,   # Near edge
    6e-3,   # Likely diverges
    1e-2,   # Expected to diverge
]
```

## Expected Outcomes

### Successful Run
- **Target loss (full)**: ≤ 1.45
- **Target loss (low-resource)**: ≤ 2.00
- At least one divergent run (to identify edge of stability)
- Clear optimal learning rate

### What Success Looks Like

```
Learning Rate | Status      | Final Loss | Notes
1e-4          | ✅ Converged | 1.62       | Too slow
3e-4          | ✅ Converged | 1.42       | Good!
6e-4          | ✅ Converged | 1.40       | Best
1e-3          | ✅ Converged | 1.45       | Edge
3e-3          | ⚠️  Unstable | 2.10       | Oscillates
6e-3          | ❌ Diverged  | NaN        | Too high
1e-2          | ❌ Diverged  | NaN        | Too high
```

## Deliverables Checklist

### (a) Hyperparameter Sweep

- [ ] Learning curves for multiple LRs (`lr_sweep_comparison.png`)
- [ ] Final loss vs LR plot (`lr_vs_final_loss.png`)
- [ ] Analysis report (`lr_sweep_analysis.md`)
- [ ] Model achieving ≤ 1.45 loss (checkpoint from best LR)
- [ ] Explanation of search strategy in experiment log

### (b) Edge of Stability Analysis

- [ ] Learning curves with at least one divergent run
- [ ] Analysis of divergence point vs optimal LR
- [ ] Discussion of "edge of stability" folk wisdom
- [ ] Convergence rate comparison

## Hyperparameter Search Strategy

### Recommended Approach

1. **Coarse sweep** (this implementation):
   - Test 7 LRs spanning 2 orders of magnitude
   - Identify divergence point
   - Find best performing LR

2. **Fine-tuning** (if needed):
   - Narrow search around best LR
   - Test 3-5 LRs in ±2× range
   - Refine to meet target loss

3. **Validation**:
   - Run best LR multiple times with different seeds
   - Verify stability and reproducibility

### Why This Strategy Works

- **Logarithmic spacing**: Captures exponential relationship between LR and loss
- **Wide range**: Ensures we find divergence point
- **Multiple seeds**: Not needed initially due to deterministic training
- **Efficient**: Single sweep gives actionable results

## Interpreting Results

### Edge of Stability

**Folk wisdom**: Best LR is "at the edge of stability" (just below divergence point)

**What to look for**:
- Divergence LR (first LR that diverges)
- Best LR (lowest final loss among converged runs)
- Ratio: `divergence_lr / best_lr`

**Expected ratio**: 2-4×
- If ratio > 4×: Best LR is conservative, can be more aggressive
- If ratio < 2×: Best LR is near edge, validates folk wisdom
- If ratio ≈ 1×: Best LR is at edge (risky but fast convergence)

### Convergence Patterns

**Good convergence**:
```
Loss: 5.2 → 3.8 → 2.6 → 1.8 → 1.4 (smooth decrease)
```

**Too conservative**:
```
Loss: 5.2 → 4.9 → 4.6 → 4.3 → 4.0 (slow decrease)
```

**Unstable**:
```
Loss: 5.2 → 3.1 → 4.8 → 2.9 → 5.5 (oscillates)
```

**Diverged**:
```
Loss: 5.2 → 8.3 → 45.2 → NaN (explodes)
```

## Optimization Tips

### Speed Up Training

**On CUDA**:
```python
# Enable TF32 (automatic in our code)
torch.set_float32_matmul_precision('high')
```

**On MPS**:
```python
# Don't use TF32 (causes unstable training)
# Use AOT eager compilation
model = torch.compile(model, backend="aot_eager")
```

**On CPU**:
```python
# Use standard compilation
model = torch.compile(model)
```

### Reduce Memory

```python
# Use smaller batch size
"batch_size": 64  # instead of 128

# Use mixed precision (careful: may affect convergence)
"dtype": "float16"  # or "bfloat16"

# Reduce context length
"context_length": 128  # instead of 256
```

## Monitoring During Training

### What to Watch

```bash
# Check tokens/sec (throughput)
Iter 100/10000 | Loss: 4.21 | LR: 1.2e-04 | Tok/s: 8543
#                                                   ^^^^
# Should be 5000-10000 on GPU, 100-500 on CPU
```

### Progress Checks

```bash
# Check summary of running experiment
uv run python -m cs336_basics.basics.analyze_experiments summary \
    --experiments lr_3e_04

# View loss curves mid-training
open cs336_basics/basics/runs/lr_sweep/lr_3e_04/loss_curves.png
```

## Troubleshooting

### "Too slow" (expected ~40 min, taking >2 hours)

**Causes**:
1. Data loading bottleneck → Use memmap (already implemented)
2. No GPU acceleration → Check device with `torch.cuda.is_available()`
3. Validation too frequent → Reduce eval_interval
4. No compilation → Ensure torch.compile is used

**Check**:
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check throughput (should be >5000 tok/s on GPU)
# Look for "Tok/s" in training logs
```

### "Diverged immediately"

**Causes**:
1. Learning rate too high → Try lower rates
2. Gradient clipping too high → Try 0.5 or 0.3
3. Weight initialization issue → Check model initialization
4. Data preprocessing issue → Verify tokenized data is valid

**Fix**:
```python
# Test with very conservative LR first
--learning-rates 1e-5 3e-5 1e-4
```

### "Not reaching target loss"

**Options**:
1. Train longer → Increase max_iterations
2. Tune warmup → Try longer warmup_iters
3. Adjust LR schedule → Try different min_learning_rate
4. Tune other hyperparams → weight_decay, betas

## File Outputs

After running the sweep, you'll have:

```
cs336_basics/basics/runs/lr_sweep/
├── lr_1e_04/
│   ├── config.json                    # Full configuration
│   ├── metrics.csv                    # All metrics (timestamped)
│   ├── summary.json                   # Final statistics
│   ├── loss_curves.png                # Training curves
│   ├── lr_schedule.png                # LR schedule
│   ├── training.log                   # Detailed logs
│   └── checkpoints/
│       ├── best_model.pt              # Best validation loss
│       ├── checkpoint_1000.pt         # Periodic checkpoints
│       └── final_checkpoint.pt        # Final model
├── lr_3e_04/
│   └── ...
├── [... other LRs ...]
├── lr_sweep_comparison.png            # All LRs compared
├── lr_vs_final_loss.png               # Loss vs LR plot
└── lr_sweep_analysis.md               # Detailed analysis
```

## Next Steps After Sweep

1. **Review analysis report**: `lr_sweep_analysis.md`
2. **Document in experiment log**: `cs336_basics/basics/EXPERIMENT_LOG.md`
3. **Use best LR for subsequent experiments**
4. **Save best checkpoint for submission**

## References

- Kingma & Ba (2015): Adam: A Method for Stochastic Optimization
- GPT-2 used lr=2.5e-4 with 512 batch size
- GPT-3 used lr=6e-4 with 3.2M batch size
- "Edge of stability" concept from Cohen et al. (2021)
