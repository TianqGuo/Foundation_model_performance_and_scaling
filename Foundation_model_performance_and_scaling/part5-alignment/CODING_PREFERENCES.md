# Coding Preferences and Project Organization

This document captures preferred coding style and project organization for Part 5 (Alignment).

## Directory Structure

### Core Principle: Keep Root Clean
- **Root directory**: Only essential files (READMEs, config files, top-level scripts)
- **DO NOT clutter root** with temporary files, logs, or module-specific documentation
- Each section lives in its own subfolder inside `cs336_alignment/`

### Module Organization
```
cs336_alignment/
├── prompts/                 # Prompt templates (.prompt files)
├── section3_zero_shot/      # Section 3: Zero-shot prompting baseline
│   ├── *.py
│   └── part_5_3.sh
├── section4_sft/            # Section 4: Supervised Fine-Tuning
│   ├── *.py
│   └── part_5_4.sh
├── section5_expert_iter/    # Section 5: Expert Iteration (STaR)
│   ├── *.py
│   └── part_5_5.sh
├── section6_grpo/           # Section 6: GRPO with verified rewards
│   ├── *.py
│   └── part_5_6.sh
└── [supp_section*/]         # Supplement sections added as needed
```

Sections are added incrementally — do not create a section folder until that section is being implemented.

### Data Files
```
data/                        # Raw input data (gitignored)
├── math/                    # MATH competition dataset
├── gsm8k/
├── mmlu/
├── alpaca_eval/
└── simple_safety_tests/
```

### Results and Output
```
results/                     # All output files
├── section3/                # Section 3 outputs
├── section4/                # Section 4 outputs
├── section5/                # Section 5 outputs
├── section6/                # Section 6 outputs
└── supplement/              # Supplement section outputs
```

**Key principles:**
- All results go to `results/` directory
- **IMPORTANT**: Each section gets its own subdirectory in `results/`
  - ✅ `results/section3/zero_shot_accuracy.txt`
  - ❌ `results/zero_shot_accuracy.txt`
- Use descriptive naming: `results/{section}/{task}.txt`

## File Naming Conventions

### Python Files
- One script per task/purpose — clear descriptive names
- Keep names concise but clear

### Shell Scripts
- Task-based naming: `part_5_{section}.sh` (e.g., `part_5_3.sh`, `part_5_4.sh`)
- Supplement sections: `supp_5_{section}.sh` (e.g., `supp_5_3.sh`, `supp_5_4.sh`)
- Always `set -e`, always `cd "$(dirname "$0")"`, always echo progress
- Compute ROOT as an absolute path — never use relative paths after `cd` changes directory:
  ```bash
  ROOT="$(cd ../.. && pwd)"   # from cs336_alignment/<section>/ → part5-alignment/
  ```

### Output Files
- Pattern: `results/{section}/{description}.txt` or `.csv`
- Examples:
  - `results/section3/zero_shot_accuracy.txt`
  - `results/section4/sft_eval.csv`
- **NOT**: Files in root `results/` without a subdirectory
- **NOT**: Generic names like `output.txt`, `results.csv`

## Documentation Style

### Minimal Documentation Files
- **ONE README per module** — not per section
- Put explanations **inside scripts as comments**, not separate files
- Only create docs when absolutely necessary

### Script Documentation
- **Include all usage info in the script header**
- Format:
  ```bash
  #!/bin/bash
  # ==============================================================================
  # Title and Purpose
  # ==============================================================================
  #
  # USAGE:
  #   cd path/to/script
  #   ./script.sh
  #
  # WHAT IT DOES:
  #   Description of steps
  #
  # OUTPUT:
  #   Where files are saved
  #
  # NOTES:
  #   Important information
  #
  # ==============================================================================
  ```

## Code Organization Preferences

### Python Scripts
1. **Clear separation of concerns**: One script per task/purpose
2. **Reusable functions**: Import from shared modules within the package
3. **Command-line arguments**: Use `argparse` with sensible defaults
4. **Output paths**: Default to `results/{section}/` with descriptive names
5. **Auto-create directories**: `output_path.parent.mkdir(parents=True, exist_ok=True)`
6. **Confirm output**: `print(f"Results saved to {output}")`

### Shell Scripts
1. **One script per section/task**
2. **Self-documenting**: All instructions in header comments
3. **Error handling**: Use `set -e` to exit on errors
4. **User feedback**: Echo what's happening at each step
5. **Absolute paths via ROOT**: Compute ROOT once, use `${ROOT}/...` throughout

## Platform Considerations

### Development Workflow: Local Testing → Remote Training

**Two Environments**:
1. **Local laptop** (WSL2/Windows with RTX 4090, 16 GB VRAM)
   - Unit tests, code development, small-scale smoke tests
   - Can run inference on small models for sanity checks
   - Full SFT/GRPO training is too slow

2. **Remote instance** (H100, Ubuntu, cluster)
   - Full SFT, Expert Iteration, GRPO training runs
   - Cluster may already have models/datasets at `/data/` — always soft-link before downloading

**Code Compatibility Requirements**:
- ✅ All scripts must work on **both** environments
- ✅ Use absolute paths via ROOT in shell scripts
- ✅ Check for cluster data directories before downloading
- ✅ No hardcoded environment-specific paths

### Writing Portable Scripts

```bash
# ✅ Good: Absolute ROOT, check cluster first
ROOT="$(cd ../.. && pwd)"
if [ -d "/data/MATH" ]; then
    ln -s "/data/MATH" "${ROOT}/data/math"
else
    # download dataset
fi
```

```python
# ✅ Good: Auto-create output dir, print path
output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)
print(f"Results saved to {output_path}")

# ❌ Bad: Absolute paths, no mkdir
output_path = "/home/user/results/output.txt"
```

### What Runs Where

| Task | Local (RTX 4090) | Remote (H100) |
|------|-----------------|---------------|
| `pytest tests/` | ✅ YES | ✅ YES |
| Zero-shot inference (small model) | ✅ YES | ✅ YES |
| SFT / Expert Iteration / GRPO training | ❌ too slow | ✅ YES |
| DPO training (supplement) | ❌ too slow | ✅ YES |

### Line Endings and Permissions

**Always use Unix (LF) line endings** — CRLF breaks bash on both WSL2 and Linux:
```bash
sed -i 's/\r$//' script.sh
chmod +x script.sh
```

## Dependencies and Imports

- Specify all dependencies in `pyproject.toml`
- Use absolute package imports: `from cs336_alignment.section4_sft.train import ...`
- Test adapters in `tests/adapters.py` use lazy inline imports:
  ```python
  def run_sft(text: str):
      from cs336_alignment.section4_sft.train import train_step
      return train_step(text)
  ```

## Git and Version Control

### What to Commit
- ✅ Source code (`.py`, `.sh`)
- ✅ Configuration files (`pyproject.toml`)
- ✅ One README per module
- ✅ Small text summaries and written answers

### What to Ignore
- ❌ Raw datasets — too large, regenerable
- ❌ Model checkpoints (`.pt`, `.pth`, `.bin`, `.safetensors`)
- ❌ Large result files — regenerable
- ❌ Cache directories (`__pycache__/`, `.pytest_cache/`)

### Important Git Rules
- ❌ **NEVER commit** without explicit approval
- ❌ **NEVER run** `git commit` or `git push` without being asked
- ✅ **ALWAYS show** `git status` and ask before staging anything

## Example: Good vs Bad Organization

### ❌ Bad
```
part5-alignment/
├── zero_shot_results.txt     # Cluttered root
├── train_sft.py              # Floating script with no module home
├── cs336_alignment/
│   ├── __init__.py
│   └── sft.py                # Not in a section subfolder
```

### ✅ Good
```
part5-alignment/
├── CODING_PREFERENCES.md     # Essential reference doc
├── data/math/                # Raw data (gitignored)
├── results/
│   └── section4/             # Section-specific outputs
│       └── sft_eval.txt
└── cs336_alignment/
    └── section4_sft/         # Section 4 implementation
        ├── train.py
        ├── evaluate.py
        └── part_5_4.sh
```

---

**For Future Sessions**: Import this file at the start to understand organizational preferences and coding style.