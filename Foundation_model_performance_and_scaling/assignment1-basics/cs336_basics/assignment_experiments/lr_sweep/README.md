# Learning Rate Sweep Experiment

**Problem (learning_rate): Tune the learning rate (3 points)**

This experiment performs a hyperparameter sweep over learning rates to find the optimal value for training a 17M parameter transformer language model on TinyStories.

## Quick Start

### 1. Run the Sweep

```bash
# Full sweep on GPU (recommended)
uv run python -m cs336_basics.assignment_experiments.lr_sweep.learning_rate_sweep

# Low-resource mode (CPU/MPS)
uv run python -m cs336_basics.assignment_experiments.lr_sweep.learning_rate_sweep --low-resource

# Custom learning rates
uv run python -m cs336_basics.assignment_experiments.lr_sweep.learning_rate_sweep \
    --learning-rates 1e-4 3e-4 6e-4 1e-3
```

### 2. Monitor Progress

While the sweep is running, you can check progress:

```bash
# Diagnose current status and throughput
uv run python -m cs336_basics.assignment_experiments.lr_sweep.diagnose_lr_sweep
```

### 3. Analyze Results

After the sweep completes:

```bash
# Generate analysis and plots
uv run python -m cs336_basics.assignment_experiments.lr_sweep.analyze_lr_sweep
```

## Files

- **`learning_rate_sweep.py`** - Main sweep script that runs multiple experiments
- **`analyze_lr_sweep.py`** - Analysis script that generates comparison plots and reports
- **`diagnose_lr_sweep.py`** - Diagnostic tool to check GPU usage and training progress
- **`LEARNING_RATE_GUIDE.md`** - Detailed guide with tips and troubleshooting
- **`README.md`** - This file

## Target

- **Validation loss**: ≤ 1.45 on TinyStories (full mode) or ≤ 2.00 (low-resource mode)
- **Model size**: 17M parameters (non-embedding)

## Model Configuration

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

## Expected Runtime

- **GPU (H100)**: ~30-40 minutes per learning rate × 7 rates = ~4 hours
- **CPU**: ~1.5 hours per learning rate (low-resource mode)
- **MPS (Apple Silicon)**: ~36 minutes per learning rate (low-resource mode)

## Default Learning Rates

The sweep tests 7 learning rates by default:

1. **1e-4**: Conservative baseline
2. **3e-4**: Common starting point (GPT-2 style)
3. **6e-4**: Aggressive (GPT-3 style)
4. **1e-3**: Very aggressive
5. **3e-3**: Near stability edge
6. **6e-3**: Likely near/at divergence
7. **1e-2**: Expected to diverge

## Outputs

Results are saved to `cs336_basics/basics/runs/lr_sweep/`:

```
cs336_basics/basics/runs/lr_sweep/
├── lr_1e_04/              # Each learning rate
│   ├── config.json
│   ├── metrics.csv
│   ├── summary.json
│   ├── loss_curves.png
│   └── checkpoints/
├── lr_3e_04/
├── ...
├── lr_sweep_comparison.png    # Combined plots
├── lr_vs_final_loss.png
└── lr_sweep_analysis.md       # Analysis report
```

## Deliverables

### (a) Hyperparameter sweep over learning rates

- ✅ Learning curves for multiple learning rates: `lr_sweep_comparison.png`
- ✅ Final loss vs learning rate: `lr_vs_final_loss.png`
- ✅ Analysis report: `lr_sweep_analysis.md`
- ✅ Model achieving ≤ 1.45 validation loss: Best checkpoint from sweep

### (b) Edge of stability analysis

- ✅ Learning curves including divergent runs
- ✅ Analysis of how divergence point relates to best learning rate
- ✅ Discussion in analysis report

## Troubleshooting

### Slow Training

If training is taking much longer than expected:

1. **Check GPU usage**:
   ```bash
   uv run python -m cs336_basics.assignment_experiments.lr_sweep.diagnose_lr_sweep
   ```

2. **Verify CUDA is available**:
   ```bash
   uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
   ```

3. **Check expected throughput**:
   - GPU: 5,000-10,000 tokens/sec
   - CPU: 100-500 tokens/sec

### Common Issues

See `LEARNING_RATE_GUIDE.md` for detailed troubleshooting guide.

## Documentation

- **`LEARNING_RATE_GUIDE.md`** - Complete guide with:
  - Model configuration details
  - Training budget calculations
  - Expected outcomes
  - Optimization tips
  - Monitoring during training
  - Troubleshooting guide

## Integration with Infrastructure

This experiment uses the tracking infrastructure from `cs336_basics.basics`:

```python
from cs336_basics.basics import create_experiment_config
from cs336_basics.basics.run_experiment import run_experiment
```

All experiment results are tracked with:
- Gradient steps and wallclock time
- Loss curves (by steps and time)
- Learning rate schedules
- Checkpoints (best, periodic, final)
