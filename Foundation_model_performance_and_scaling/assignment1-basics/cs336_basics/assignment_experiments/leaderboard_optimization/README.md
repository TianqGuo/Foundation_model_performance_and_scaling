# Leaderboard Optimization Strategy

## Overview

This folder contains optimizations for the CS336 Assignment 1 leaderboard competition.

**Goal:** Minimize validation loss on OpenWebText within 1.5 hours on H100 GPU.

**Baseline:** 4.02 validation loss (from our standard OWT experiment)
**Target:** <3.5 validation loss
**Stretch Goal:** <3.0 validation loss

---

## Implemented Optimizations

### 1. Weight Tying ⭐ (Highest Impact)

**What:** Share parameters between input embeddings and output projection layer

**Benefits:**
- Reduces parameters by 36.2% (16.4M parameters saved!)
- Model: 45.2M → 28.8M parameters
- Faster training (fewer parameters to update)
- Better memory efficiency
- Often improves performance on smaller models

**Implementation:** `modifications/weight_tied_transformer.py`

**References:**
- Original Transformer paper (Vaswani et al., 2017, Section 3.4)
- PaLM paper (Chowdhery et al., 2022, Section 2)

**Key Code:**
```python
# Use transposed embedding weights as output projection
logits = torch.matmul(x, self.token_embeddings.weight.T)
```

**Initialization:** Lower std deviation (1/sqrt(d_model)) for stability

---

### 2. Larger Batch Size + Scaled Learning Rate

**What:** Increase batch size from 32 → 64, scale LR accordingly

**Benefits:**
- More tokens per iteration → faster convergence
- Better gradient estimates → more stable training
- Improved GPU utilization on H100

**Configuration:**
- Batch size: 64 (2× baseline)
- Learning rate: 4e-4 (scaled by √2 ≈ 1.33×)
- Shorter warmup: 300 iterations (vs 400)

---

### 3. Aggressive Learning Rate Schedule

**What:** Higher peak LR with faster warmup and cosine decay

**Benefits:**
- Faster initial progress
- More exploration in parameter space
- Better final performance with proper decay

**Configuration:**
- Max LR: 4e-4 (vs 3e-4 baseline, +33%)
- Min LR: 4e-5 (vs 3e-5 baseline)
- Warmup: 300 iterations (vs 400, -25%)
- Cosine decay over full training

---

### 4. Mixed Precision Training (BFloat16)

**What:** Use bfloat16 instead of float32

**Benefits:**
- 2× speedup on H100 with Tensor Cores
- Same memory, ~2× throughput
- Minimal accuracy impact (bfloat16 has same exponent range as fp32)

**Note:** H100 has native bfloat16 support, making this essentially free performance

---

## Configuration Files

### `configs/baseline_1.5hr.json`
- Standard configuration (no optimizations)
- For baseline comparison
- Expected: ~4.0 validation loss

### `configs/optimized_1.5hr.json`
- All optimizations enabled
- Weight tying + larger batch + fast schedule + bfloat16
- Expected: ~3.3-3.5 validation loss

---

## Estimated Performance Improvements

| Optimization | Expected Improvement | Basis |
|--------------|---------------------|-------|
| Weight Tying | -0.2 to -0.4 loss | Literature + fewer params |
| Larger Batch | -0.1 to -0.2 loss | Better gradient estimates |
| Fast LR Schedule | -0.1 to -0.2 loss | More training progress |
| Mixed Precision | Faster (same loss) | Hardware acceleration |
| **Combined** | **-0.4 to -0.8 loss** | Synergistic effects |

**Expected Final Loss:** 3.2-3.6 (vs baseline 4.02)

---

## Testing Strategy

### Phase 1: Quick Validation (TinyStories)
```bash
# Test weight tying on TinyStories (15 min)
uv run python experiments/test_on_tinystories.py --config configs/optimized_1.5hr.json
```

### Phase 2: Short OWT Run (30 min)
```bash
# Test on OpenWebText for 30 minutes
uv run python experiments/test_short_run.py --config configs/optimized_1.5hr.json --time-limit 1800
```

### Phase 3: Final 1.5hr Run
```bash
# Full leaderboard run (1.5 hours)
uv run python experiments/run_leaderboard.py --config configs/optimized_1.5hr.json
```

---

## Time Budget Analysis

**1.5 hours = 5,400 seconds**

Estimated iteration times:
- Standard model (fp32, batch=32): ~0.18s/iter → 30,000 iterations
- Optimized model (bf16, batch=64): ~0.11s/iter → 49,000 iterations

**More iterations = better final loss!**

---

## Additional Optimizations Considered (Future Work)

### Not Implemented (Time Constraints)
1. **Muon Optimizer** - 30-40% faster convergence but requires custom implementation
2. **Flash Attention** - 2-3× speedup but may need CUDA kernel
3. **Gradient Accumulation** - Simulate larger batches without OOM
4. **Model Architecture Changes** - Wider/shallower models for speed

### Why Not Implemented
- Weight tying alone gives 36% parameter reduction
- Batch size + mixed precision give 2× speedup
- These are proven, low-risk optimizations
- More complex changes need extensive testing

---

## File Structure

```
leaderboard_optimization/
├── README.md (this file)
├── configs/
│   ├── baseline_1.5hr.json          # No optimizations
│   └── optimized_1.5hr.json         # All optimizations
├── modifications/
│   └── weight_tied_transformer.py   # Weight-tied model
├── experiments/
│   ├── test_on_tinystories.py       # Quick validation
│   ├── test_short_run.py            # 30-min OWT test
│   └── run_leaderboard.py           # Final 1.5hr run
└── results/
    ├── learning_curves.png          # Generated after run
    ├── final_submission.json        # Leaderboard submission
    └── description.txt              # What we did
```

---

## Expected Results

**Baseline (no optimizations):**
- Validation Loss: ~4.0
- Parameters: 45.2M
- Iterations in 1.5hr: ~30,000
- Perplexity: ~55

**Optimized (weight tying + batch + LR + bf16):**
- Validation Loss: ~3.3-3.5 ✓
- Parameters: 28.8M (-36%)
- Iterations in 1.5hr: ~49,000 (+63%)
- Perplexity: ~27-33

**Improvement: ~0.5-0.7 loss reduction**

---

## Running the Experiments

### Quick Start (Recommended Testing Order)

**Phase 1: Quick Validation on TinyStories (~15 minutes)**
```bash
cd /mnt/d/Repos/MachineLearningNotes/stanford-cs336/assignment1-basics

uv run python cs336_basics/assignment_experiments/leaderboard_optimization/experiments/test_on_tinystories.py \
    --config cs336_basics/assignment_experiments/leaderboard_optimization/configs/optimized_1.5hr.json
```

**Phase 2: Short OpenWebText Test (~30 minutes)**
```bash
uv run python cs336_basics/assignment_experiments/leaderboard_optimization/experiments/test_short_run.py \
    --config cs336_basics/assignment_experiments/leaderboard_optimization/configs/optimized_1.5hr.json \
    --time-limit 1800
```

**Phase 3: Final 1.5-Hour Leaderboard Run**
```bash
uv run python cs336_basics/assignment_experiments/leaderboard_optimization/experiments/run_leaderboard.py \
    --config cs336_basics/assignment_experiments/leaderboard_optimization/configs/optimized_1.5hr.json \
    --time-limit 5400 \
    --output-dir cs336_basics/basics/runs/leaderboard_final
```

This will:
1. Train for exactly 1.5 hours (5400 seconds)
2. Save checkpoints every 4,000 iterations
3. Evaluate every 400 iterations
4. Save best model based on validation loss
5. Generate metrics CSV for plotting

---

## Submission Package

After the final run completes, create submission package with:

```bash
uv run python cs336_basics/assignment_experiments/leaderboard_optimization/experiments/create_submission.py \
    --run-dir cs336_basics/basics/runs/leaderboard_final
```

This generates:
- `final_submission.json` - Final metrics and configuration
- `learning_curves.png` - Wallclock time vs loss plot (required for leaderboard)
- `description.txt` - Complete description of modifications
- Ready for GitHub leaderboard submission

**Submission Requirements:**
- ✓ Validation loss < 1.5 hours wallclock time
- ✓ Learning curve with wallclock time on x-axis
- ✓ Description of modifications made
- ✓ Final validation loss metric

---

## Key Insights

1. **Weight tying is the MVP** - 36% fewer parameters with potential performance boost
2. **Mixed precision is free speed** - 2× faster on H100 with no downside
3. **Larger batches help** - Better gradient estimates, more iterations/sec
4. **Aggressive LR works** - Small models benefit from higher learning rates

---

## References

- **Weight Tying:** Vaswani et al. (2017) "Attention Is All You Need", Section 3.4
- **PaLM:** Chowdhery et al. (2022) "PaLM: Scaling Language Modeling with Pathways", Section 2
- **NanoGPT Speedrun:** https://github.com/KellerJordan/modded-nanogpt
- **Llama 3:** https://github.com/meta-llama/llama3
- **Our Ablations:** See `../ablations/` for component analysis

---

## Status

**Implementation Complete:**
- ✅ Weight-tied model implemented (`modifications/weight_tied_transformer.py`)
- ✅ Optimized configurations created (`configs/optimized_1.5hr.json`)
- ✅ Test scripts created (TinyStories, 30-min, final run)
- ✅ Submission package generator created
- ✅ Strategy documented

**Testing Phases:**
- ⏳ Phase 1: Quick validation on TinyStories (~15 min)
- ⏳ Phase 2: 30-min OpenWebText test
- ⏳ Phase 3: Final 1.5-hour leaderboard run
- ⏳ Phase 4: Create and submit leaderboard package

**Next Step:** Run Phase 1 (TinyStories validation) to verify implementation

---

## Quick Parameter Count Verification

To verify weight tying works correctly, run:

```bash
cd /mnt/d/Repos/MachineLearningNotes/stanford-cs336/assignment1-basics
uv run python cs336_basics/assignment_experiments/leaderboard_optimization/modifications/weight_tied_transformer.py
```

Expected output:
```
Parameter Comparison:
  Standard model:     45,224,448
  Weight-tied model:  28,800,000
  Untied model:       45,224,448

Savings from tying: 16,424,448 parameters
Reduction:          36.3%
```

---

**Ready to test and submit!** 🚀
