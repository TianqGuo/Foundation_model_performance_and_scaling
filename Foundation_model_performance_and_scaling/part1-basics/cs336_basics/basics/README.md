# Assignment 1 Experiments

This directory contains the experiment tracking infrastructure and experiment runners for CS336 Assignment 1, Section 7: Experiments.

## Overview

The experiment infrastructure provides:

- **Comprehensive metric tracking**: Training/validation loss with gradient steps and wallclock time
- **Automatic visualization**: Loss curves, learning rate schedules, and experiment comparisons
- **Structured logging**: CSV and JSON exports for analysis
- **Experiment management**: Configuration templates, comparison tools, and experiment logs

## Quick Start

### 1. Run a Basic Experiment

```bash
# From assignment1-basics directory
uv run python -m cs336_basics/basics.run_experiment \
    --name "baseline_17m" \
    --config cs336_basics/basics/configs/tinystories_17m_baseline.json \
    --description "Baseline 17M parameter model on TinyStories"
```

### 2. View Results

After training, results are saved to `cs336_basics/basics/runs/<experiment_name>/`:

- `config.json` - Full experiment configuration
- `metrics.csv` - All logged metrics (loss, lr, throughput) by step and time
- `metrics.json` - Detailed metrics in JSON format
- `summary.json` - Experiment statistics and final results
- `loss_curves.png` - Training and validation loss plots
- `lr_schedule.png` - Learning rate schedule visualization
- `checkpoints/` - Model checkpoints

### 3. Compare Multiple Experiments

```python
from pathlib import Path
from cs336_basics/basics.experiment_logger import compare_experiments

experiment_dirs = [
    Path("cs336_basics/basics/runs/baseline_17m"),
    Path("cs336_basics/basics/runs/experiment_2"),
]

compare_experiments(
    experiment_dirs,
    output_path=Path("cs336_basics/basics/comparison.png"),
    experiment_names=["Baseline", "Experiment 2"]
)
```

## Directory Structure

```
cs336_basics/basics/
├── __init__.py                      # Package initialization
├── experiment_tracker.py            # Core tracking infrastructure
├── experiment_logger.py             # High-level logger with visualization
├── run_experiment.py                # Experiment runner script
├── README.md                        # This file
├── EXPERIMENT_LOG.md                # Log of all experiments (manually maintained)
├── configs/                         # Experiment configurations
│   └── tinystories_17m_baseline.json
└── runs/                            # Experiment outputs (generated)
    └── <experiment_name>/
        ├── config.json
        ├── metrics.csv
        ├── metrics.json
        ├── summary.json
        ├── loss_curves.png
        ├── lr_schedule.png
        └── checkpoints/
```

## Components

### ExperimentTracker

Low-level metric tracking with CSV/JSON export.

```python
from cs336_basics/basics.experiment_tracker import ExperimentTracker, ExperimentConfig

config = ExperimentConfig(
    experiment_name="test",
    experiment_id="abc123",
    vocab_size=10000,
    # ... other config
)

tracker = ExperimentTracker("test", config, Path("./output"))

# Log metrics
tracker.log_step(
    step=100,
    train_loss=2.5,
    val_loss=2.7,
    learning_rate=3e-4,
    tokens_per_sec=5000.0
)

# Finalize and save
tracker.finalize()
```

### ExperimentLogger

High-level logger with automatic plotting and Python logging integration.

```python
from cs336_basics/basics.experiment_logger import ExperimentLogger

logger = ExperimentLogger(
    experiment_name="test",
    config=config,
    output_dir=Path("./output")
)

# Log training step
logger.log_training_step(
    step=100,
    train_loss=2.5,
    learning_rate=3e-4,
    tokens_per_sec=5000.0
)

# Log validation step
logger.log_validation_step(
    step=100,
    val_loss=2.7,
    val_perplexity=14.88
)

# Finalize (auto-generates plots)
logger.finalize()
```

### run_experiment.py

Complete training runner with integrated tracking:

```bash
uv run python -m cs336_basics/basics.run_experiment \
    --name "my_experiment" \
    --config configs/my_config.json \
    --description "Testing new architecture" \
    --output-dir "./custom_output"
```

## Creating Experiment Configurations

Copy and modify the baseline configuration:

```json
{
  "description": "Brief description of what is being tested",
  "dataset": "TinyStories",
  "vocab_size": 10000,
  "context_length": 512,
  "d_model": 768,
  "num_layers": 12,
  "num_heads": 12,
  "d_ff": 3072,
  "rope_theta": 10000.0,
  "batch_size": 64,
  "max_iterations": 10000,
  "learning_rate": 0.0006,
  "min_learning_rate": 6e-05,
  "warmup_iters": 500,
  "weight_decay": 0.1,
  "beta1": 0.9,
  "beta2": 0.999,
  "grad_clip": 1.0,
  "train_data_path": "cs336_basics/artifacts/datasets/tinystories_train_tokens.npy",
  "val_data_path": "cs336_basics/artifacts/datasets/tinystories_tokens.npy",
  "device": null,
  "dtype": "float32",
  "seed": 42,
  "log_interval": 50,
  "eval_interval": 200,
  "checkpoint_interval": 1000
}
```

## Baseline Model Configuration (17M Parameters)

The baseline configuration is designed for ~17M parameters on TinyStories:

- **Architecture**: 12 layers, 768 hidden dim, 12 attention heads, 3072 FFN dim
- **Training**: 10K iterations with cosine LR schedule
- **Learning Rate**: Peak 6e-4, min 6e-5, 500-step warmup
- **Batch Size**: 64 sequences of 512 tokens
- **Regularization**: Weight decay 0.1, gradient clipping 1.0

## Best Practices

1. **Naming Conventions**: Use descriptive experiment names like `baseline_17m`, `no_dropout_17m`, `lr_sweep_1e3`

2. **Version Control**: Commit configuration files but not run outputs (runs/ is gitignored)

3. **Experiment Documentation**: Maintain `EXPERIMENT_LOG.md` with:
   - Date and experiment name
   - Hypothesis being tested
   - Configuration changes from baseline
   - Key observations and results
   - Links to plots and artifacts

4. **Reproducibility**: Always set seeds and save full configurations

5. **Comparison**: Use the comparison utility to plot multiple experiments together

## Tracking Deliverables

For CS336 Assignment 1, Section 7, the deliverables are:

1. **Logging Infrastructure**: The code in this directory (✓ Complete)

2. **Experiment Log**: `EXPERIMENT_LOG.md` documenting all experiments with:
   - What was tested (hypothesis/motivation)
   - Configuration parameters
   - Results summary
   - Loss curves with gradient steps and wallclock time
   - Key observations and conclusions

3. **Loss Curves**: Generated automatically for each experiment
   - Training loss vs gradient steps
   - Training loss vs wallclock time
   - Validation loss on both axes

## Tips

- Start with shorter runs (1000-2000 iterations) for rapid experimentation
- Monitor `tokens_per_sec` to ensure training efficiency
- Use validation loss (not training loss) to compare models
- Check loss curves for signs of overfitting or underfitting
- Save interesting checkpoints for later analysis
- Compare wallclock time across configurations to understand efficiency

## Example Workflow

```bash
# 1. Train baseline
uv run python -m cs336_basics/basics.run_experiment \
    --name "baseline" \
    --config cs336_basics/basics/configs/tinystories_17m_baseline.json

# 2. Train variant (e.g., different learning rate)
# First create configs/higher_lr.json with lr=0.001
uv run python -m cs336_basics/basics.run_experiment \
    --name "higher_lr" \
    --config cs336_basics/basics/configs/higher_lr.json

# 3. Compare experiments
python -c "
from pathlib import Path
from cs336_basics/basics.experiment_logger import compare_experiments
compare_experiments(
    [Path('cs336_basics/basics/runs/baseline'),
     Path('cs336_basics/basics/runs/higher_lr')],
    Path('cs336_basics/basics/lr_comparison.png')
)
"

# 4. Document findings in EXPERIMENT_LOG.md
```

## Notes

- All experiments use the TinyStories dataset tokenized with the 10K vocabulary
- Target model size is ~17M parameters for fast iteration
- Metrics are logged at regular intervals (configurable via log_interval)
- Validation happens every eval_interval steps
- Checkpoints saved every checkpoint_interval steps
- Best model (lowest validation loss) is always saved

## Integration with Existing Code

This infrastructure wraps the training code in `cs336_basics/train.py` and adds:
- Structured experiment configuration
- Automatic metric collection and export
- Visualization generation
- Experiment comparison utilities

The core training logic remains unchanged, ensuring compatibility with existing checkpoints and configurations.
