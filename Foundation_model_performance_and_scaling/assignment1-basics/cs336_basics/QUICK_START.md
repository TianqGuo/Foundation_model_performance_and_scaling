# Quick Start Guide - Training Your Transformer

## 🚀 30-Second Start

```bash
# 1. Create sample config
python cs336_basics/train.py --create_sample_config

# 2. Edit sample_config.json with your settings

# 3. Start training
python cs336_basics/train.py --config sample_config.json
```

## 📋 Prerequisites

You need:
1. Tokenized training data as a `.npy` file
2. (Optional) Tokenized validation data as a `.npy` file
3. Python environment with dependencies installed

## 🎯 Complete Workflow

### Step 1: Prepare Your Data

**Option A: Use real data with BPE tokenizer**
```bash
# Train BPE tokenizer
python cs336_basics/training/bpe_train.py \
  --input data/train.txt \
  --vocab_size 10000 \
  --output_dir tokenizer/

# Tokenize your data
python cs336_basics/utils/dataset_utils.py prepare \
  --text_files data/train.txt data/val.txt \
  --tokenizer_vocab tokenizer/vocab.json \
  --tokenizer_merges tokenizer/merges.pkl \
  --output data/tokens.npy
```

**Option B: Create dummy data for testing**
```bash
# Create dummy dataset
python cs336_basics/utils/dataset_utils.py dummy \
  --output data/train_data.npy \
  --size 100000 \
  --vocab_size 1000

# Split into train/val
python cs336_basics/utils/dataset_utils.py split \
  --input data/train_data.npy \
  --train_output data/train.npy \
  --val_output data/val.npy \
  --val_split 0.1
```

### Step 2: Configure Training

**Create config file:**
```bash
python cs336_basics/train.py --create_sample_config
```

**Edit `sample_config.json`:**
```json
{
  "vocab_size": 1000,
  "context_length": 128,
  "d_model": 256,
  "num_layers": 4,
  "num_heads": 4,
  "d_ff": 1024,
  "batch_size": 16,
  "max_iterations": 5000,
  "learning_rate": 3e-4,
  "train_data_path": "data/train.npy",
  "val_data_path": "data/val.npy",
  "checkpoint_dir": "./checkpoints"
}
```

### Step 3: Start Training

```bash
python cs336_basics/train.py --config sample_config.json
```

### Step 4: Monitor Progress

Watch the console output:
```
2025-10-11 12:00:00 - INFO - Starting Transformer Language Model Training
2025-10-11 12:00:00 - INFO - Using device: cuda
2025-10-11 12:00:00 - INFO - Model initialized with 25,165,824 trainable parameters
2025-10-11 12:00:01 - INFO - Iter      0/5000 | Loss: 6.9087 | LR: 0.00e+00 | Time: 0.234s
2025-10-11 12:00:02 - INFO - Iter     10/5000 | Loss: 6.5432 | LR: 3.00e-05 | Time: 0.198s
...
```

Check the log file:
```bash
tail -f checkpoints/training.log
```

## 🔧 Common Configurations

### Tiny Model (for testing)
```json
{
  "vocab_size": 1000,
  "context_length": 64,
  "d_model": 128,
  "num_layers": 2,
  "num_heads": 2,
  "d_ff": 512,
  "batch_size": 8,
  "max_iterations": 1000
}
```

### Small Model (~25M parameters)
```json
{
  "vocab_size": 10000,
  "context_length": 512,
  "d_model": 512,
  "num_layers": 6,
  "num_heads": 8,
  "d_ff": 2048,
  "batch_size": 32,
  "max_iterations": 50000
}
```

### Medium Model (~110M parameters)
```json
{
  "vocab_size": 32000,
  "context_length": 1024,
  "d_model": 768,
  "num_layers": 12,
  "num_heads": 12,
  "d_ff": 3072,
  "batch_size": 64,
  "max_iterations": 100000
}
```

## 📊 Monitoring Training

### Console Logs
Real-time training updates printed to console.

### File Logs
Detailed logs saved to `checkpoints/training.log`.

### Checkpoints
- `checkpoint_1000.pt`, `checkpoint_2000.pt`, ... - Periodic saves
- `best_model.pt` - Best model based on validation loss
- `final_checkpoint.pt` - Final model after training

## 🔄 Resume Training

If training is interrupted:
```bash
python cs336_basics/train.py \
  --config sample_config.json \
  --resume_from checkpoints/checkpoint_3000.pt
```

## 🎛️ CLI Override

Override config settings from command line:
```bash
python cs336_basics/train.py \
  --config sample_config.json \
  --batch_size 64 \
  --learning_rate 1e-4 \
  --max_iterations 10000
```

## 🧪 Test Your Setup

Run the test script to verify everything works:
```bash
python test_training.py
```

Expected output:
```
ALL TESTS PASSED!
The training script successfully:
  ✓ Loaded data with memory mapping
  ✓ Initialized model and optimizer
  ✓ Ran training loop with gradient updates
  ✓ Evaluated on validation data
  ✓ Saved checkpoints
  ✓ Logged training progress
```

## ⚡ Performance Tips

### Speed Up Training
- Use GPU: `--device cuda`
- Increase batch size: `--batch_size 64`
- Enable compilation: `--compile_model` (PyTorch 2.0+)
- Use mixed precision: `--dtype bfloat16`

### Reduce Memory Usage
- Decrease batch size: `--batch_size 8`
- Decrease context length: `--context_length 256`
- Use smaller model: reduce `d_model`, `num_layers`
- Use float16: `--dtype float16`

### Improve Training Stability
- Lower learning rate: `--learning_rate 1e-4`
- Increase warmup: `--warmup_iters 1000`
- Reduce gradient clipping: `--grad_clip 0.5`

## 🐛 Troubleshooting

### "ModuleNotFoundError"
Activate your virtual environment:
```bash
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### "Out of Memory"
Reduce batch size or model size:
```bash
python cs336_basics/train.py --config sample_config.json --batch_size 4
```

### "NaN Loss"
Reduce learning rate or increase warmup:
```bash
python cs336_basics/train.py --config sample_config.json --learning_rate 1e-4 --warmup_iters 500
```

### "No such file or directory: train_data.npy"
Create or specify correct path to data:
```bash
python cs336_basics/utils/dataset_utils.py dummy --output data/train_data.npy
```

## 📚 Further Reading

- **Full Documentation**: See `TRAINING_README.md`
- **Implementation Details**: See `IMPLEMENTATION_SUMMARY.md`
- **Help**: Run `python cs336_basics/train.py --help`

## 🎓 CS336 Assignment Compliance

This implementation satisfies all requirements from Assignment 1, Section 5.3:
- ✅ Configurable hyperparameters
- ✅ Memory-efficient data loading with np.memmap
- ✅ Checkpoint serialization
- ✅ Periodic logging

## 🚀 Ready to Train!

You're all set! Start with:
```bash
python test_training.py              # Test your setup
python cs336_basics/train.py --help  # See all options
```

Good luck with your training! 🎉
