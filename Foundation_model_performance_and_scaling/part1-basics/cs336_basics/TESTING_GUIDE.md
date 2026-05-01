# Test Execution Guide

## Environment Setup
1. This project uses `uv` for environment management. Install `uv` following the instructions in `assignment1-basics/README.md`.
2. All tests assume the repo root is the working directory (`assignment1-basics/`).
3. Run tests using `uv run pytest` to automatically manage the environment.

## Running the Full Suite
Execute the complete set of assignment tests:
```bash
uv run pytest
```
Pytest automatically discovers test modules under `tests/`. The test suite covers:
- `test_tokenizer.py` — BPE tokenizer encoding/decoding
- `test_train_bpe.py` — BPE training pipeline
- `test_model.py` — transformer model components
- `test_optimizer.py` — AdamW optimizer and learning rate schedules
- `test_serialization.py` — checkpoint saving/loading
- `test_data.py` — data loading and batching
- `test_nn_utils.py` — neural network utilities
- `test_training.py` — end-to-end training integration

## Targeted Test Runs
Common patterns when iterating on features:
- Run a specific file:
  ```bash
  uv run pytest tests/test_tokenizer.py
  ```
- Filter by keyword (matches test names and class names):
  ```bash
  uv run pytest -k "encode"
  ```
- Re-run only failed tests from last run:
  ```bash
  uv run pytest --lf
  ```
- Run with verbose output:
  ```bash
  uv run pytest -vv
  ```

## Performance Notes
- Some tests (e.g., `test_training.py`) can take longer because they exercise full training loops. Use `-k` to filter specific tests while iterating.
- To parallelize tests, install `pytest-xdist` and run: `uv run pytest -n auto`

## Debugging Tips
- Use `-vv` for verbose output with detailed test information
- Stop after first failure: `uv run pytest -x`
- Stop after N failures: `uv run pytest --maxfail=2`
- Drop into debugger on failure: `uv run pytest --pdb`
- Show local variables on failure: `uv run pytest -l`

## Regenerating Artifacts
Tests expect tokenizer vocabularies and tokenised datasets under `cs336_basics/artifacts/`. To regenerate:

1. **Train tokenizers** (creates vocabulary files):
   ```bash
   uv run python -m cs336_basics.tokenizer_training.bpe_train \
       --input-path data/TinyStoriesV2-GPT4-train.txt \
       --output-dir cs336_basics/artifacts/vocabularies \
       --vocab-size 10000 \
       --special-token "<|endoftext|>"
   ```

2. **Encode datasets** (creates `.npy` token files):
   ```bash
   uv run python -m cs336_basics.experiments.encode_datasets
   ```

## Test Adapter Configuration
The tests use adapter functions defined in `tests/adapters.py` to connect your implementation to the test suite. Ensure these adapters correctly import and expose your module functions.
