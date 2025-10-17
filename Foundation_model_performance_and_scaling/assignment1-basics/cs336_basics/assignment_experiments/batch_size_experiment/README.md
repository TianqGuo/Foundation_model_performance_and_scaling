# Batch Size Sweep Experiment

**Problem (batch_size_experiment): Batch size variations (1 point)**

This experiment studies how different batch sizes affect throughput and model
quality for the 17M-parameter TinyStories transformer. It mirrors the
learning-rate sweep workflow so you can quickly launch runs, monitor
progress, and collect the learning curves required for the assignment
deliverables.

## Quick Start

```bash
# Default sweep (GPU, recommended)
uv run python -m cs336_basics.assignment_experiments.batch_size_experiment.batch_size_sweep

# Low-resource mode (CPU/MPS)
uv run python -m cs336_basics.assignment_experiments.batch_size_experiment.batch_size_sweep --low-resource

# Custom batch sizes and base lr
uv run python -m cs336_basics.assignment_experiments.batch_size_experiment.batch_size_sweep \
    --batch-sizes 8 32 128 512 --learning-rate 3e-4
```

## Outputs

Results live under `cs336_basics/basics/runs/batch_size_sweep/` with a
sub-directory per batch size containing the config template, metrics, and
checkpoints. The script also writes a `summary.json` capturing the final
validation loss and throughput for each trial.

## Targets and Tips

- Keep total tokens processed near **327,680,000** (auto-computed).
- Re-tune the learning rate if a configuration becomes unstable.
- Track Tok/s in the logs to discuss efficiency gains.
- Document how validation loss and convergence speed change as you scale the
  batch.

See `BATCH_SIZE_GUIDE.md` for assignment context, recommended hyperparameters,
and troubleshooting tips.
