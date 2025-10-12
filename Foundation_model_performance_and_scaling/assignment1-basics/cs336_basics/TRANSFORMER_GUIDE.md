# Transformer Training & Decoding Guide

## Overview
The transformer language model implementation in `cs336_basics` provides modular components for constructing, training, and decoding autoregressive models. This guide summarises how the pieces fit together and how to train or sample from the model.

## Code Architecture
- `transformer_training/model/`
  - Core building blocks (`embedding.py`, `attention.py`, `transformer_block.py`, etc.).
  - `transformer_lm.py` stitches blocks into the full `TransformerLM` class and exposes a `generate` helper.
- `transformer_training/optimizer/`
  - Custom optimizer (`adamw.py`), learning-rate scheduler (`lr_schedule.py`), and gradient clipping.
- `transformer_training/checkpointing.py`
  - Save/load utilities for model + optimizer state.
- `transformer_decode/`
  - Shared decoding helpers (`SamplingConfig`, `decode`, `generate_text`) supporting temperature and nucleus sampling.
- `training/`
  - Backward-compatible shims exporting the new training/decoding helpers for legacy imports/tests.
- `train.py`
  - End-to-end training script (argument-parsed configuration, logging, evaluation, checkpointing).

## Data & Artifacts
- Tokenised datasets live in `artifacts/datasets/*.npy` (uint16 arrays produced by tokenizer scripts).
- Checkpoints are stored in `checkpoints/` and contain model/optimizer state plus metadata.
- Additional configs or prompts can be added to `sample_config.json` or passed via CLI.

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
- `experiments/training_demo.py` — minimal training loop demonstration and logging.
- `experiments/learning_rate_tuning.py` — sweeps scheduling hyperparameters.
- `experiments/adamw_accounting.py` & `adamw_verification.py` — validate optimizer behaviour.
- `experiments/realistic_training_time.py` — back-of-the-envelope throughput estimates.
- `transformer_decode` helpers are also accessible in experiments for qualitative sampling.

## Tips & Conventions
- Context length enforcement happens inside `TransformerLM.forward`; sequences longer than `context_length` are truncated automatically during decoding.
- Optimizer parameters (`weight_decay`, scheduler warmup) are central to stable training—use the provided helpers to reproduce reference behaviour.
- Always align tokenizer vocab/special tokens with the model checkpoints; mismatches typically manifest as nonsense outputs or index errors.
- Checkpoint metadata (iteration, model class) can be inspected with `verify_checkpoint` for quick sanity checks.
