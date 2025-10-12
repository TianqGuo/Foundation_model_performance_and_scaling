# Transformer Language Model Training Guide

This guide explains how to use the training script (`cs336_basics/train.py`) that implements all CS336 Assignment 1 requirements.

## Features

The training script implements all requirements from section 5.3:

✅ **Configurable Hyperparameters**: Control all model and optimizer settings via CLI or config file
✅ **Memory-Efficient Data Loading**: Uses `np.memmap` for large datasets
✅ **Checkpoint Serialization**: Save/load checkpoints to user-provided paths
✅ **Periodic Logging**: Training and validation performance to console and file

## Quick Start

### 1. Prepare Your Data

First, tokenize your text data and save as `.npy` files:

```bash
# Using the dataset utilities
python cs336_basics/utils/dataset_utils.py prepare \
  --text_files data/train.txt \
  --tokenizer_vocab tokenizer_vocab.json \
  --tokenizer_merges tokenizer_merges.pkl \
  --output data/train_tokens.npy
```

Or create a dummy dataset for testing:

```bash
python cs336_basics/utils/dataset_utils.py dummy \
  --output data/train_data.npy \
  --size 100000 \
  --vocab_size 1000
```

### 2. Create Configuration File

Generate a sample configuration:

```bash
python cs336_basics/train.py --create_sample_config
```

This creates `sample_config.json`. Edit it with your settings:

```json
{
  "vocab_size": 10000,
  "context_length": 512,
  "d_model": 512,
  "num_layers": 6,
  "num_heads": 8,
  "d_ff": 2048,
  "batch_size": 32,
  "max_iterations": 10000,
  "learning_rate": 3e-4,
  "train_data_path": "data/train_data.npy",
  "val_data_path": "data/val_data.npy",
  "checkpoint_dir": "./checkpoints"
}
```

### 3. Start Training

Train with config file (recommended):

```bash
python cs336_basics/train.py --config my_config.json
```

Or with command-line arguments:

```bash
python cs336_basics/train.py \
  --train_data_path data/train_data.npy \
  --val_data_path data/val_data.npy \
  --vocab_size 10000 \
  --d_model 512 \
  --num_layers 6 \
  --num_heads 8 \
  --batch_size 32 \
  --max_iterations 10000 \
  --checkpoint_dir ./checkpoints
```

## Configuration Options

### Model Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `vocab_size` | Vocabulary size | **Required** |
| `context_length` | Maximum sequence length | 512 |
| `d_model` | Model dimension | 512 |
| `num_layers` | Number of transformer layers | 6 |
| `num_heads` | Number of attention heads | 8 |
| `d_ff` | Feed-forward dimension | 2048 |
| `rope_theta` | RoPE theta parameter | 10000.0 |

### Training Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `batch_size` | Training batch size | 32 |
| `max_iterations` | Maximum training iterations | 10000 |
| `learning_rate` | Peak learning rate | 3e-4 |
| `min_learning_rate` | Minimum learning rate | 3e-5 |
| `warmup_iters` | Warmup iterations | 100 |
| `weight_decay` | Weight decay (L2 regularization) | 0.1 |
| `beta1` | Adam beta1 | 0.9 |
| `beta2` | Adam beta2 | 0.999 |
| `grad_clip` | Gradient clipping threshold | 1.0 |

### Data Configuration

| Parameter | Description |
|-----------|-------------|
| `train_data_path` | Path to training data (.npy file) |
| `val_data_path` | Path to validation data (.npy file, optional) |

### Checkpointing

| Parameter | Description | Default |
|-----------|-------------|---------|
| `checkpoint_dir` | Directory for saving checkpoints | `./checkpoints` |
| `checkpoint_interval` | Save checkpoint every N iterations | 1000 |
| `resume_from` | Path to checkpoint to resume from | None |

### Logging

| Parameter | Description | Default |
|-----------|-------------|---------|
| `log_interval` | Log training metrics every N iterations | 10 |
| `eval_interval` | Evaluate on validation set every N iterations | 100 |
| `eval_iters` | Number of batches for validation | 20 |

### System

| Parameter | Description | Default |
|-----------|-------------|---------|
| `device` | Device (cpu, cuda, mps) | Auto-detect |
| `dtype` | Data type (float32, float16, bfloat16) | float32 |
| `compile_model` | Use torch.compile() | False |
| `seed` | Random seed | 42 |

## Advanced Usage

### Resume Training from Checkpoint

```bash
python cs336_basics/train.py \
  --config my_config.json \
  --resume_from checkpoints/checkpoint_5000.pt
```

### Mixed Precision Training

```bash
python cs336_basics/train.py \
  --config my_config.json \
  --dtype bfloat16
```

### Model Compilation (PyTorch 2.0+)

```bash
python cs336_basics/train.py \
  --config my_config.json \
  --compile_model
```

### Override Config Settings

Command-line arguments override config file settings:

```bash
python cs336_basics/train.py \
  --config my_config.json \
  --batch_size 64 \
  --learning_rate 1e-4
```

## Output Files

The training script creates the following in `checkpoint_dir`:

- `training.log` - Detailed training log
- `checkpoint_XXXX.pt` - Periodic checkpoints (every `checkpoint_interval` iterations)
- `best_model.pt` - Best model based on validation loss
- `final_checkpoint.pt` - Final checkpoint after training completes

## Example Training Session

```bash
# 1. Create dummy data for testing
python cs336_basics/utils/dataset_utils.py dummy \
  --output data/train_data.npy \
  --size 100000 \
  --vocab_size 1000

# 2. Split into train/val
python cs336_basics/utils/dataset_utils.py split \
  --input data/train_data.npy \
  --train_output data/train.npy \
  --val_output data/val.npy \
  --val_split 0.1

# 3. Create config
python cs336_basics/train.py --create_sample_config

# 4. Edit sample_config.json with your paths and settings
# ... (edit the file)

# 5. Start training
python cs336_basics/train.py --config sample_config.json
```

## Training Output

During training, you'll see logs like:

```
2025-10-11 12:00:00 - INFO - Starting Transformer Language Model Training
2025-10-11 12:00:00 - INFO - Using device: cuda
2025-10-11 12:00:00 - INFO - Loading training data from: data/train.npy
2025-10-11 12:00:00 - INFO - Training data loaded: 90,000 tokens
2025-10-11 12:00:00 - INFO - Model initialized with 25,165,824 trainable parameters
2025-10-11 12:00:00 - INFO - Starting training loop
2025-10-11 12:00:01 - INFO - Iter      0/10000 | Loss: 6.9087 | LR: 0.00e+00 | Time: 0.234s | Tok/s: 4381
2025-10-11 12:00:02 - INFO - Iter     10/10000 | Loss: 6.5432 | LR: 3.00e-05 | Time: 0.198s | Tok/s: 5172
...
```

## Memory-Efficient Data Loading

The script uses `np.memmap` to load data efficiently:

```python
# Automatically uses memory mapping for large datasets
train_data = load_data_memmap(train_data_path, dtype=np.uint16)

# Data is loaded on-demand, not fully into RAM
# Perfect for multi-GB datasets
```

## Checkpoint Format

Checkpoints contain:
- Model state (weights and buffers)
- Optimizer state (momentum, running averages)
- Training iteration
- Additional metadata (loss, learning rate, etc.)

Load checkpoints programmatically:

```python
from cs336_basics.transformer_training import load_checkpoint

model = TransformerLM(...)
optimizer = AdamW(...)

iteration = load_checkpoint("checkpoint.pt", model, optimizer)
# Continue training from iteration + 1
```

## Monitoring Training

### Console Logs
Real-time updates printed to console every `log_interval` iterations.

### File Logs
Detailed logs saved to `{checkpoint_dir}/training.log`.

### Validation Metrics
Evaluated every `eval_interval` iterations:
- Validation loss
- Perplexity
- Evaluation time

### Integration with W&B (Optional)

To add Weights & Biases logging, modify the training loop to call `wandb.log()`:

```python
import wandb

# In main():
wandb.init(project="transformer-lm", config=config)

# In training loop:
wandb.log({
    "train/loss": train_loss,
    "train/lr": current_lr,
    "val/loss": eval_metrics['loss'],
    "val/perplexity": eval_metrics['perplexity'],
})
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce `batch_size`
- Reduce `context_length`
- Use smaller model (`d_model`, `num_layers`)
- Use `dtype="float16"` or `dtype="bfloat16"`

### Slow Training
- Increase `batch_size` (if memory allows)
- Use `--compile_model` (PyTorch 2.0+)
- Use GPU if available (`--device cuda`)
- Reduce `eval_interval` and `eval_iters`

### NaN Loss
- Reduce `learning_rate`
- Increase `warmup_iters`
- Reduce `grad_clip` threshold
- Check data quality

## Implementation Details

### Learning Rate Schedule
Uses cosine annealing with warmup:
- Warmup: Linear increase from 0 to `learning_rate`
- Cosine: Smooth decay to `min_learning_rate`
- Post-training: Constant at `min_learning_rate`

### Gradient Clipping
Clips by global L2 norm:
```python
clip_gradients(model.parameters(), max_l2_norm=grad_clip)
```

### Optimizer
AdamW with decoupled weight decay:
```python
AdamW(params, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
```

## References

- CS336 Assignment 1: Language Modeling from Scratch
- Vaswani et al. (2017): Attention Is All You Need
- Loshchilov & Hutter (2019): Decoupled Weight Decay Regularization
- Touvron et al. (2023): LLaMA - Open and Efficient Foundation Language Models
