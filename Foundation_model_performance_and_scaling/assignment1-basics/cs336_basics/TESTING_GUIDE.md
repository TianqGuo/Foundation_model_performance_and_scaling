# Test Execution Guide

## Environment Setup
1. Activate the venv, please refer to assignment1-basics/README.md for installation
   ```bash
   source .venv/bin/activate
    ```
2. All tests assume the repo root is the working directory (`assignment1-basics/`).

## Running the Full Suite
Execute the complete set of assignment tests:
```bash
pytest
```
Pytest automatically discovers modules under `tests/`. The suite covers tokeniser, model, optimiser, serialization, and integration behaviour.

## Targeted Test Runs
Common patterns when iterating on features:
- Run a specific file:
  ```bash
  pytest tests/test_tokenizer.py
  ```
- Filter by keyword (matches test names and class names):
  ```bash
  pytest -k "generate and not slow"
  ```
- Re-run failures quickly:
  ```bash
  pytest --lf
  ```

## Performance Notes
- Some tests (e.g., `test_training.py`) can take longer because they exercise training loops. Use `-k` or run files individually while you iterate.
- When working inside WSL/containers, consider `PYTEST_ADDOPTS="-n auto"` with `pytest-xdist` if installed to parallelise the suite.

## Debugging Tips
- Use `-vv` for verbose output, including individual test names.
- `pytest --maxfail=1` stops after the first failure.
- Combine `-x` and `--pdb` to drop into the debugger on failure.

## Regenerating Artifacts
Tests expect tokenizer vocabularies and tokenised datasets under `cs336_basics/artifacts/`. If they are missing, re-run the scripts described in `BPE_TOKENIZER.md` before running the suite.

## Keeping Imports Green
When refactoring package structure, ensure shim modules (e.g., `cs336_basics/training/`) continue to re-export moved utilities. Run at least the following smoke tests after layout changes:
```bash
pytest tests/test_tokenizer.py tests/test_model.py tests/test_optimizer.py tests/test_serialization.py
```
