# Quick Start Guide - Experiment Tracking

This guide will help you run your first experiment with full tracking in under 5 minutes.

## Prerequisites

1. Ensure you have the TinyStories dataset tokenized:
   ```bash
   # From assignment1-basics directory
   ls cs336_basics/artifacts/datasets/tinystories_*_tokens.npy
   ```

2. If not, run the tokenization:
   ```bash
   uv run python -m cs336_basics.tokenizer_training.bpe_train \
       --input-path data/TinyStoriesV2-GPT4-train.txt \
       --output-dir cs336_basics/artifacts/vocabularies \
       --vocab-size 10000 \
       --special-token "<|endoftext|>"

   uv run python -m cs336_basics.experiments.encode_datasets
   ```

## Step 1: Run Your First Experiment

The baseline configuration is already set up for a 17M parameter model:

```bash
cd assignment1-basics

uv run python -m cs336_basics/basics.run_experiment \
    --name "baseline_17m" \
    --config cs336_basics/basics/configs/tinystories_17m_baseline.json \
    --description "First baseline run on TinyStories"
```

This will:
- Train for 10,000 iterations (~30-60 minutes on GPU)
- Log metrics every 50 steps
- Evaluate every 200 steps
- Save checkpoints every 1000 steps
- Generate loss curves and statistics

## Step 2: Monitor Progress

During training, you'll see output like:

```
Iter    100/10000 | Loss: 4.2156 | LR: 1.20e-04 | Tok/s: 8543
Iter    200/10000 | Loss: 3.9821 | LR: 2.40e-04 | Tok/s: 8621

Validation | Loss: 3.8234 | Perplexity: 45.78
```

## Step 3: View Results

After training completes, results are in `cs336_basics/basics/runs/baseline_17m/`:

```bash
# View summary
uv run python -m cs336_basics/basics.analyze_experiments summary --experiments baseline_17m

# List all experiments
uv run python -m cs336_basics/basics.analyze_experiments list

# View loss curves
open cs336_basics/basics/runs/baseline_17m/loss_curves.png
```

## Step 4: Run Your Second Experiment

Let's try a different learning rate:

```bash
# First, create a new config (or copy and edit)
cp cs336_basics/basics/configs/tinystories_17m_baseline.json \
   cs336_basics/basics/configs/higher_lr.json

# Edit the learning_rate field to 0.001
# Then run:

uv run python -m cs336_basics/basics.run_experiment \
    --name "higher_lr" \
    --config cs336_basics/basics/configs/higher_lr.json \
    --description "Testing higher learning rate (1e-3)"
```

## Step 5: Compare Experiments

```bash
# Generate comparison table
uv run python -m cs336_basics/basics.analyze_experiments compare \
    --experiments baseline_17m higher_lr

# This creates:
# - Printed comparison table
# - cs336_basics/basics/runs/experiment_comparison.png
```

## Step 6: Document Your Findings

Open `cs336_basics/basics/EXPERIMENT_LOG.md` and fill in the results:

```markdown
## Experiment 1: Baseline Model (17M Parameters)

**Date**: 2025-01-15

**Results**:
- Training loss (final): 2.45
- Validation loss (final): 2.67
- Best validation loss: 2.65 at step 8000
- Training time: 0.75 hours

**Observations**:
- Training converged smoothly
- No signs of overfitting
- Validation loss plateaued around step 7000
```

## Common Commands

### Run an experiment
```bash
uv run python -m cs336_basics/basics.run_experiment \
    --name <name> \
    --config <config.json> \
    --description "<what you're testing>"
```

### Resume from checkpoint
```bash
uv run python -m cs336_basics/basics.run_experiment \
    --name <name> \
    --config <config.json> \
    --resume-from cs336_basics/basics/runs/<name>/checkpoints/checkpoint_5000.pt
```

### Analyze single experiment
```bash
uv run python -m cs336_basics/basics.analyze_experiments summary --experiments <name>
```

### Compare multiple experiments
```bash
uv run python -m cs336_basics/basics.analyze_experiments compare \
    --experiments exp1 exp2 exp3 \
    --output comparison_table.csv
```

### List all experiments
```bash
uv run python -m cs336_basics/basics.analyze_experiments list
```

## Quick Testing (Short Run)

For rapid iteration, you can create a quick test config:

```json
{
  "description": "Quick test run",
  "vocab_size": 10000,
  "context_length": 512,
  "d_model": 256,
  "num_layers": 4,
  "num_heads": 4,
  "d_ff": 1024,
  "batch_size": 32,
  "max_iterations": 500,
  "learning_rate": 0.0006,
  "min_learning_rate": 6e-05,
  "warmup_iters": 50,
  "train_data_path": "cs336_basics/artifacts/datasets/tinystories_train_tokens.npy",
  "val_data_path": "cs336_basics/artifacts/datasets/tinystories_tokens.npy",
  "log_interval": 25,
  "eval_interval": 100,
  "checkpoint_interval": 500
}
```

This runs in ~5 minutes and helps verify changes work before committing to longer runs.

## Troubleshooting

### "FileNotFoundError: ... tokens.npy"
You need to tokenize the dataset first. See Prerequisites above.

### "CUDA out of memory"
Reduce `batch_size` in your config file (try 32 or 16).

### "Import error: No module named 'cs336_basics/basics'"
Make sure you're running from the `assignment1-basics` directory.

### Training is very slow
- Check `tokens_per_sec` in the output
- Ensure you're using GPU: check "Using device: cuda" at start
- Try reducing `context_length` or `batch_size`

### Plots not generating
Make sure matplotlib backend is working:
```bash
python -c "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; print('OK')"
```

## What's Next?

1. **Read the full README**: `cs336_basics/basics/README.md`
2. **Review the experiment log**: `cs336_basics/basics/EXPERIMENT_LOG.md`
3. **Try ablations**: Modify architecture, learning rates, etc.
4. **Document everything**: Keep detailed notes of what you try and observe

## Tips for Success

1. **Start small**: Test with short runs before committing to full training
2. **Version configs**: Keep all experiment configs in `configs/` directory
3. **Name clearly**: Use descriptive names like `no_dropout_17m` or `lr_1e3`
4. **Compare frequently**: Use the comparison tools to understand differences
5. **Document immediately**: Write observations while they're fresh
6. **Save interesting checkpoints**: Note which ones to keep for later analysis

Happy experimenting! 🚀
