# Assignment 1 Experiments

This directory contains experiments for CS336 Assignment 1, Section 7.

## Structure

- **`cs336_basics/basics/`**: Experiment tracking infrastructure (tracking, logging, visualization)
- **`cs336_basics/assignment_experiments/`**: Actual experiments (this directory)

## Running Experiments

All experiments use the infrastructure from `cs336_basics/basics/`:

```bash
# Run an experiment
uv run python -m cs336_basics.basics.run_experiment \
    --name "experiment_name" \
    --config cs336_basics/basics/configs/config_file.json \
    --description "What you're testing"
```

## Experiment Outputs

Results are saved to `cs336_basics/basics/runs/<experiment_name>/`:
- `config.json` - Full configuration
- `metrics.csv` - All metrics
- `loss_curves.png` - Visualizations
- `checkpoints/` - Model checkpoints

## Documentation

See `cs336_basics/basics/` for complete documentation:
- `README.md` - Complete infrastructure guide
- `QUICK_START.md` - Get started in 5 minutes
- `EXPERIMENT_LOG.md` - Document your experiments

## Adding Experiments

Experiments for specific assignment sections will be added here as separate modules/scripts.
