# Assignment 1 Section 7 Experiments

This directory contains experiment scripts for CS336 Assignment 1, Section 7.

## Experiments

### Learning Rate Tuning (`learning_rate_sweep.py`)

**Problem (learning_rate): Tune the learning rate (3 points)**

Performs hyperparameter sweep over learning rates to find optimal value.

**Target**: Validation loss ≤ 1.45 on TinyStories (or ≤ 2.00 for low-resource mode)

**Model Configuration (17M parameters)**:
- vocab_size: 10000
- context_length: 256
- d_model: 512
- d_ff: 1344
- num_layers: 4
- num_heads: 16
- Total tokens: ~327,680,000 (or ~40,000,000 for low-resource)

#### Running the Sweep

```bash
# Full sweep on GPU (recommended)
uv run python -m cs336_basics.assignment_experiments.learning_rate_sweep

# Low-resource mode (CPU/MPS)
uv run python -m cs336_basics.assignment_experiments.learning_rate_sweep --low-resource

# Custom learning rates
uv run python -m cs336_basics.assignment_experiments.learning_rate_sweep \
    --learning-rates 1e-4 3e-4 6e-4 1e-3 3e-3

# Specify device
uv run python -m cs336_basics.assignment_experiments.learning_rate_sweep --device cuda
```

#### Analyzing Results

```bash
# Analyze sweep results
uv run python -m cs336_basics.assignment_experiments.analyze_lr_sweep \
    --sweep-dir cs336_basics/basics/runs/lr_sweep

# This generates:
# - lr_sweep_comparison.png (loss curves for all learning rates)
# - lr_vs_final_loss.png (final loss vs learning rate)
# - lr_sweep_analysis.md (detailed analysis report)
```

#### Expected Runtime

- **GPU (H100)**: ~30-40 minutes per learning rate × 7 rates = ~4 hours
- **CPU**: ~1.5 hours per learning rate (low-resource mode)
- **MPS (Apple Silicon)**: ~36 minutes per learning rate (low-resource mode)

#### Deliverables

**(a) Hyperparameter sweep over learning rates**
- Learning curves for multiple learning rates: `lr_sweep_comparison.png`
- Final loss vs learning rate: `lr_vs_final_loss.png`
- Analysis report: `lr_sweep_analysis.md`
- Model achieving ≤ 1.45 validation loss: Best checkpoint from sweep

**(b) Edge of stability analysis**
- Learning curves including divergent runs
- Analysis of how divergence point relates to best learning rate
- See analysis report for detailed findings

### Default Learning Rates Tested

The sweep tests 7 learning rates by default:
1. **1e-4**: Conservative baseline
2. **3e-4**: Common starting point (GPT-2 style)
3. **6e-4**: Aggressive (GPT-3 style)
4. **1e-3**: Very aggressive
5. **3e-3**: Near stability edge
6. **6e-3**: Likely near/at divergence
7. **1e-2**: Expected to diverge

## Structure

- **`cs336_basics/basics/`**: Experiment tracking infrastructure
  - Core tracking, logging, visualization
  - Analysis tools
  - Configuration templates

- **`cs336_basics/assignment_experiments/`**: Actual experiments (this directory)
  - `learning_rate_sweep.py` - LR tuning experiments
  - `analyze_lr_sweep.py` - LR sweep analysis
  - More experiments to be added...

## Experiment Outputs

Results are saved to `cs336_basics/basics/runs/<experiment_name>/`:

```
cs336_basics/basics/runs/
└── lr_sweep/
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

## Tips for Running Experiments

### 1. GPU Availability

Check GPU before running:
```bash
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. Low-Resource Mode

If running on CPU or Apple Silicon:
```bash
# Use low-resource flag (40M tokens instead of 327M)
uv run python -m cs336_basics.assignment_experiments.learning_rate_sweep --low-resource
```

Settings:
- Total tokens: 40,000,000 (vs 327,680,000)
- Batch size: 32 (vs 128)
- Target loss: 2.00 (vs 1.45)
- Expected time: ~1.5 hours per LR on CPU

### 3. Quick Testing

Test the pipeline with minimal iterations:
```bash
uv run python -m cs336_basics.assignment_experiments.learning_rate_sweep \
    --quick-test \
    --learning-rates 3e-4 1e-3
```

### 4. Monitoring Progress

During training, you'll see:
```
Iter    100/10000 | Loss: 4.2156 | LR: 1.20e-04 | Tok/s: 8543
Iter    200/10000 | Loss: 3.9821 | LR: 2.40e-04 | Tok/s: 8621

Validation | Loss: 3.8234 | Perplexity: 45.78
```

Check intermediate results:
```bash
# View progress of specific experiment
uv run python -m cs336_basics.basics.analyze_experiments summary \
    --experiments lr_3e_04

# List all completed experiments
uv run python -m cs336_basics.basics.analyze_experiments list
```

## Optimization Tips

### Compilation (Speed Up Training)

The sweep automatically handles compilation based on device:
- **CPU**: `torch.compile(model)`
- **MPS**: `torch.compile(model, backend="aot_eager")`
- **CUDA**: Default (optionally enable TF32)

### Memory Issues

If you encounter OOM errors:
1. Reduce batch size in the config
2. Use mixed precision (set `dtype: "float16"`)
3. Reduce context length

### Slow Training

If training is slower than expected:
1. Check dataloader isn't bottlenecking (should use memmap)
2. Verify GPU utilization: `nvidia-smi`
3. Ensure batch operations are used (no Python loops over batch)
4. Check validation isn't running too frequently

## Documentation

See `cs336_basics/basics/` for infrastructure documentation:
- **`README.md`**: Complete infrastructure guide
- **`QUICK_START.md`**: Get started in 5 minutes
- **`EXPERIMENT_LOG.md`**: Template for documenting results

## Adding New Experiments

Create new experiment scripts in this directory following this pattern:

```python
# cs336_basics/assignment_experiments/new_experiment.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cs336_basics.basics import create_experiment_config
from cs336_basics.basics.run_experiment import run_experiment

def run_new_experiment():
    config = create_experiment_config(
        experiment_name="new_experiment",
        description="What you're testing",
        # ... model and training params
    )

    run_experiment(
        experiment_name="new_experiment",
        config_dict=config.to_dict(),
        output_dir=Path("cs336_basics/basics/runs/new_experiment"),
    )

if __name__ == "__main__":
    run_new_experiment()
```

## Troubleshooting

### Import Errors
```bash
# Make sure you use uv run
uv run python -m cs336_basics.assignment_experiments.learning_rate_sweep
```

### Data Not Found
```bash
# Check tokenized datasets exist
ls cs336_basics/artifacts/datasets/*.npy

# If missing, tokenize datasets
uv run python -m cs336_basics.experiments.encode_datasets
```

### Slow First Run
First run compiles the model (torch.compile), which takes extra time.
Subsequent iterations will be faster.

## Next Steps

1. Run learning rate sweep
2. Analyze results with analysis script
3. Document findings in `cs336_basics/basics/EXPERIMENT_LOG.md`
4. Use best learning rate for subsequent experiments
