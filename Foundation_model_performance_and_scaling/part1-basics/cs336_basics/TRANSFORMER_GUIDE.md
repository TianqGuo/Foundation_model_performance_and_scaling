# Transformer Training & Decoding Guide

## Overview
The transformer language model implementation in `cs336_basics` provides modular components for constructing, training, and decoding autoregressive models. This guide summarises how the pieces fit together and how to train or sample from the model.

## Code Architecture
- `transformer_training/model/`
  - Core building blocks: `embedding.py`, `attention.py`, `multihead_attention.py`, `transformer_block.py`, `linear.py`, `rmsnorm.py`, `rope.py`, `softmax.py`, `activations.py`, `swiglu.py`, `cross_entropy.py`.
  - `transformer_lm.py` — stitches blocks into the full `TransformerLM` class and exposes a `generate` helper.
- `transformer_training/optimizer/`
  - Custom optimizer (`adamw.py`), learning-rate scheduler (`lr_schedule.py`), and gradient clipping (`gradient_clipping.py`).
- `transformer_training/checkpointing.py`
  - Save/load utilities for model + optimizer state with metadata support.
- `transformer_decode/`
  - Shared decoding helpers (`SamplingConfig`, `decode`, `generate_text`) supporting temperature, top-k, and nucleus (top-p) sampling.
- `data/`
  - Data loading utilities (`data_loader.py`) for memory-mapped `.npy` datasets.
- `train.py`
  - End-to-end training script with argument parsing, JSON config support, logging, evaluation, and checkpointing.

## Data & Artifacts
- Tokenised datasets live in `artifacts/datasets/*.npy` (uint16 arrays produced by tokenizer scripts like `experiments/encode_datasets.py`).
- Example datasets: `tinystories_train_tokens.npy`, `tinystories_tokens.npy`, `owt_train_tokens.npy`, `sample_tokens.npy`.
- Checkpoints are stored in `checkpoints/` and contain model/optimizer state plus metadata (iteration, loss, perplexity).
- `sample_config.json` provides a template configuration with all hyperparameters; can be customized or overridden via CLI.

## Training Workflow
1. **Prepare data**: Generate tokenised `.npy` files using the BPE tokenizer guide.
2. **Configure**: Either edit `sample_config.json` or supply command-line flags:
   ```bash
   python -m cs336_basics.train \
       --config sample_config.json \
       --train-path artifacts/datasets/tinystories_train_tokens.npy \
       --val-path artifacts/datasets/tinystories_tokens.npy \
       --max-iters 1000 \
       --logdir experiments/runs/exp1
   ```
3. **Resume**: Pass `--load-checkpoint checkpoints/latest.pt` to resume from a saved state.
4. **Artifacts produced**: training logs (`logdir`), checkpoints, and optional evaluation metrics.

## Decoding & Text Generation
- Use the high-level helper:
  ```python
  from cs336_basics.transformer_decode import generate_text, SamplingConfig
  from cs336_basics.tokenizer import Tokenizer
  from cs336_basics.transformer_training.model.transformer_lm import TransformerLM
  import torch

  tokenizer = Tokenizer.from_files(
      "artifacts/vocabularies/tinystories_vocab.json",
      "artifacts/vocabularies/tinystories_merges.pkl",
      ["<|endoftext|>"]
  )
  model = TransformerLM(...)
  model.load_state_dict(torch.load("checkpoints/latest.pt")['model_state_dict'])
  model.eval()

  completion = generate_text(
      model,
      tokenizer,
      prompt="Once upon a time",
      max_new_tokens=100,
      temperature=0.8,
      top_p=0.95,
      eos_token="<|endoftext|>"
  )
  print(completion)
  ```
- Alternatively call `model.generate(...)` directly with token IDs when you already have tensors prepared.
- `SamplingConfig` captures the same parameters and can be reused across sampling runs.

## Experiments & Analysis
- `experiments/training_demo.py` — minimal training loop demonstration with logging examples.
- `experiments/learning_rate_tuning.py` — learning rate scheduling experiments and hyperparameter sweeps.
- `experiments/lr_schedule_demo.py` — visualizes learning rate schedules.
- `experiments/adamw_accounting.py` & `adamw_verification.py` — validate optimizer correctness and accounting.
- `experiments/realistic_training_time.py` & `experiments/simple_training_calc.py` — throughput estimates and training time calculations.
- `experiments/quick_experiments.py` — rapid prototyping and sanity checks.
- Decoding helpers from `transformer_decode` are used in experiments for qualitative text generation.

## Tips & Conventions
- Context length enforcement happens inside `TransformerLM.forward`; sequences longer than `context_length` raise a `ValueError`. During generation, the decode function automatically windows the input to the last `context_length` tokens.
- Optimizer parameters (`weight_decay`, scheduler warmup) are central to stable training—configure them carefully in the config file or via CLI.
- Always align tokenizer vocab/special tokens with the model checkpoints; mismatches typically manifest as nonsense outputs or index errors. The `vocab_size` parameter must match your trained tokenizer.
- Checkpoint files contain `model_state_dict`, `optimizer_state_dict`, `iteration`, and optional metadata like `val_loss` and `val_perplexity`.
- The training script uses `np.memmap` for memory-efficient data loading, allowing training on datasets larger than RAM.
