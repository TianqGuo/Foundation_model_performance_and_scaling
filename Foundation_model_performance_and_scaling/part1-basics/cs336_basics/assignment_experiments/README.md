# Assignment 1 Section 7 Experiments

This directory contains experiment scripts for CS336 Assignment 1, Section 7.

## Directory Structure

Each experiment is organized in its own subdirectory:

```
cs336_basics/assignment_experiments/
├── README.md                      # This file
├── __init__.py
│
└── lr_sweep/                      # Learning rate tuning experiment
    ├── __init__.py
    ├── README.md                  # Experiment-specific guide
    ├── learning_rate_sweep.py     # Main sweep runner
    ├── analyze_lr_sweep.py        # Analysis and visualization
    ├── diagnose_lr_sweep.py       # Diagnostic tool
    └── LEARNING_RATE_GUIDE.md     # Detailed reference guide
```

## Experiments

### 1. Learning Rate Tuning (`lr_sweep/`)

**Problem (learning_rate): Tune the learning rate (3 points)**

Performs hyperparameter sweep over learning rates to find optimal value.

**Quick Start**:
```bash
# Run sweep
uv run python -m cs336_basics.assignment_experiments.lr_sweep.learning_rate_sweep

# Monitor progress
uv run python -m cs336_basics.assignment_experiments.lr_sweep.diagnose_lr_sweep

# Analyze results
uv run python -m cs336_basics.assignment_experiments.lr_sweep.analyze_lr_sweep
```

**Target**: Validation loss ≤ 1.45 on TinyStories

**Documentation**: See `lr_sweep/README.md` and `lr_sweep/LEARNING_RATE_GUIDE.md`

---

### 2. Future Experiments

More experiments will be added here following the same pattern:

- `experiment_name/` - Dedicated directory for each experiment
  - `README.md` - Experiment overview and quick start
  - `*.py` - Experiment scripts
  - Supporting documentation

## Integration with Infrastructure

All experiments use the tracking infrastructure from `cs336_basics.basics`:

```python
from cs336_basics.basics import create_experiment_config
from cs336_basics.basics.run_experiment import run_experiment
from cs336_basics.basics import ExperimentTracker, ExperimentLogger
```

This provides:
- Automatic metric tracking (gradient steps + wallclock time)
- Loss curve visualization (by steps and time)
- Learning rate schedule plots
- Checkpoint management (best, periodic, final)
- Experiment comparison tools

## Experiment Outputs

All experiment results are saved to `cs336_basics/basics/runs/`:

```
cs336_basics/basics/runs/
├── lr_sweep/                      # Learning rate sweep results
│   ├── lr_1e_04/
│   ├── lr_3e_04/
│   ├── ...
│   ├── lr_sweep_comparison.png
│   ├── lr_vs_final_loss.png
│   └── lr_sweep_analysis.md
│
└── [future_experiment]/           # Future experiment results
```

Each experiment directory contains:
- `config.json` - Full configuration
- `metrics.csv` - All metrics with timestamps
- `summary.json` - Final statistics
- `loss_curves.png` - Training curves
- `lr_schedule.png` - Learning rate schedule
- `checkpoints/` - Model checkpoints

## Adding New Experiments

To add a new experiment:

### 1. Create Experiment Directory

```bash
mkdir cs336_basics/assignment_experiments/new_experiment
cd cs336_basics/assignment_experiments/new_experiment
touch __init__.py README.md experiment_script.py
```

### 2. Create Experiment Script

```python
# cs336_basics/assignment_experiments/new_experiment/experiment_script.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from cs336_basics.basics import create_experiment_config
from cs336_basics.basics.run_experiment import run_experiment

def run_experiment_fn():
    """Run the experiment."""
    config = create_experiment_config(
        experiment_name="new_experiment",
        description="What you're testing",
        # Model architecture
        vocab_size=10000,
        d_model=768,
        num_layers=12,
        # Training params
        batch_size=128,
        max_iterations=10000,
        learning_rate=3e-4,
        # ... other params
    )

    run_experiment(
        experiment_name="new_experiment",
        config_dict=config.to_dict(),
        output_dir=Path("cs336_basics/basics/runs/new_experiment"),
    )

if __name__ == "__main__":
    run_experiment_fn()
```

### 3. Document the Experiment

Create a README.md in the experiment directory with:
- Experiment description and goals
- Quick start commands
- Configuration details
- Expected outcomes
- Troubleshooting tips

### 4. Update Main README

Add a section to this README describing the new experiment.

## Common Commands

### GPU Check
```bash
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Monitor Training
```bash
# Check specific experiment progress
uv run python -m cs336_basics.basics.analyze_experiments summary \
    --experiments experiment_name

# List all experiments
uv run python -m cs336_basics.basics.analyze_experiments list
```

### Compare Experiments
```bash
uv run python -m cs336_basics.basics.analyze_experiments compare \
    --experiments exp1 exp2 exp3
```

## Tips for Running Experiments

### GPU vs CPU

**GPU Mode** (recommended):
- Expected throughput: 5,000-10,000 tokens/sec
- Expected time per experiment: 30-40 minutes (full training)

**CPU/MPS Mode** (low-resource):
- Expected throughput: 100-500 tokens/sec
- Expected time per experiment: 1-2 hours (reduced training budget)
- Use `--low-resource` flag when available

### Optimization Tips

1. **Use compilation**: Already enabled in infrastructure
   - CUDA: Default compilation
   - MPS: `aot_eager` backend
   - CPU: Standard compilation

2. **Memory optimization**:
   - Reduce batch size if OOM
   - Use mixed precision (float16/bfloat16) carefully
   - Reduce context length if needed

3. **Data loading**:
   - Infrastructure uses memory-mapped files (np.memmap)
   - No bottleneck from data loading

### Monitoring During Training

Look for these indicators in logs:
```
Iter    100/10000 | Loss: 4.21 | LR: 1.2e-04 | Tok/s: 8543
                                                        ^^^^
                    Should be 5000-10000 on GPU
```

## Troubleshooting

### Slow Training

If training is much slower than expected:

1. **Check device being used**:
   ```bash
   # Look for "Using device: cuda" in logs
   grep "device" cs336_basics/basics/runs/experiment_name/config.json
   ```

2. **Check GPU utilization** (if using GPU):
   ```bash
   nvidia-smi -l 1
   ```

3. **Use diagnostic tools** (experiment-specific):
   - Learning rate sweep: `diagnose_lr_sweep.py`

### Import Errors

Always use `uv run`:
```bash
uv run python -m cs336_basics.assignment_experiments.lr_sweep.learning_rate_sweep
```

### Data Not Found

Ensure tokenized datasets exist:
```bash
ls cs336_basics/artifacts/datasets/*.npy
```

If missing:
```bash
uv run python -m cs336_basics.experiments.encode_datasets
```

## Documentation

### Infrastructure Documentation
See `cs336_basics/basics/` for core infrastructure:
- **`README.md`**: Complete infrastructure guide
- **`QUICK_START.md`**: Get started in 5 minutes
- **`EXPERIMENT_LOG.md`**: Template for documenting results

### Experiment-Specific Documentation
Each experiment directory has its own:
- **`README.md`**: Quick start and overview
- Additional guides as needed (e.g., `LEARNING_RATE_GUIDE.md`)

## Next Steps

1. Review experiment-specific documentation in subdirectories
2. Run experiments with appropriate settings (GPU/CPU)
3. Monitor progress with diagnostic tools
4. Analyze results with provided analysis scripts
5. Document findings in `cs336_basics/basics/EXPERIMENT_LOG.md`
