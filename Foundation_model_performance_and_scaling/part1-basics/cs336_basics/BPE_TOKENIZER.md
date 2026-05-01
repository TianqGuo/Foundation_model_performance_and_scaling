# BPE Tokenizer Guide

## Overview
The Byte Pair Encoding (BPE) tokenizer implementation in `cs336_basics` is used to turn raw text into token ID sequences that can be consumed by the transformer language model. The tokenizer code supports training custom vocabularies, serialising the trained merges/vocabulary to disk, and running a production-style encoder/decoder at inference time.

## Code Layout
- `tokenizer/`
  - `bpe_tokenizer.py` — runtime encoder/decoder (`Tokenizer`) used by experiments and model evaluation.
- `tokenizer_training/`
  - `bpe_train.py` — training pipeline for learning merges and vocab from text corpora; includes streaming, parallel pre-tokenisation, and compatibility helpers.
- `transformer_decode/` — decoding utilities share vocabulary handling with the tokenizer (for generating text later).

## Training a Tokenizer
1. **Prepare data**: place raw text files under `data/` (e.g. `TinyStoriesV2-GPT4-train.txt`).
2. **Run training** using the provided module, e.g.:
   ```bash
   python -m cs336_basics.tokenizer_training.bpe_train \
       --input-path data/TinyStoriesV2-GPT4-train.txt \
       --output-dir artifacts/vocabularies/tinystories \
       --vocab-size 10000 \
       --special-token "<|endoftext|>"
   ```
   The script will produce `*_vocab.json` and `*_merges.pkl` (and `.txt` mirrors) in the output directory.
3. **Artifacts**: resulting files live under `artifacts/vocabularies/` and are consumed by experiments/tests. They are versionable and can be shared between training sessions.

## Using the Tokenizer
```python
from cs336_basics.tokenizer import Tokenizer

# Load from saved files
tokenizer = Tokenizer.from_files(
    "artifacts/vocabularies/tinystories_vocab.json",
    "artifacts/vocabularies/tinystories_merges.pkl",
    special_tokens=["<|endoftext|>"]
)

text = "Once upon a time<|endoftext|>"
token_ids = tokenizer.encode(text)
recovered = tokenizer.decode(token_ids)
```

## Experiments & Utilities
- `experiments/tokenizer_experiments.py` — runs compression ratio analysis, cross-domain tokenization tests, throughput estimation, and dataset encoding demonstrations.
- `experiments/encode_datasets.py` — full dataset encoding script for producing tokenized `.npy` files.
- `experiments/quick_experiments.py` — lightweight tokenizer sanity checks.
- `experiments/pretokenization_example.py` — demonstrates pre-tokenization patterns.
- `artifacts/datasets/*.npy` — memmapped token arrays (uint16) generated from prepared corpora, used for transformer training.
- `artifacts/vocabularies/` — trained vocabulary files (`*_vocab.json`, `*_merges.pkl`) for TinyStories and OpenWebText.

## Tips
- Special tokens must be supplied consistently during training and inference.
- The tokenizer training script automatically handles large files via streaming and parallel processing; adjust `--min-size-for-parallel` if needed.
- Vocab IDs are stored as Python `dict[int, bytes]` to keep compatibility with the reference tokenizer. `Tokenizer.byte_to_id` is exposed for advanced workflows (e.g., detecting `<|endoftext|>` when decoding).
