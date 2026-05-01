# Coding Preferences and Project Organization

This document captures preferred coding style and project organization for this assignment.

## Directory Structure

### Core Principle: Keep Root Clean
- **Root directory**: Only essential files (READMEs, config files, top-level scripts)
- **DO NOT clutter root** with temporary files, logs, or module-specific documentation
- Each assignment part lives in its own subfolder inside `cs336_data/`

### Module Organization
```
cs336_data/
├── filtering_cc/           # Part 2: Filtering Common Crawl
│   ├── *.py                # Python implementation files
│   ├── *.sh                # Shell scripts for running tasks
│   └── README.md           # Single README with usage info
│
└── [future_part]/          # Each new assignment part gets its own folder
```

### Data Files
```
data/                       # Raw input data (gitignored)
├── CC/                     # Common Crawl WARC/WET sample files
│   ├── *.warc.gz
│   └── *.warc.wet.gz
└── [other_datasets]/
```

### Results and Output
```
results/                    # All output files (gitignored except summaries)
├── filtering_cc/           # Part 2 outputs
│   └── look_at_cc_observations.txt
└── [part_name]/            # Each part gets its own subdirectory
```

**Key principles:**
- All results go to `results/` directory
- **IMPORTANT**: Each part gets its own subdirectory in `results/`
  - ✅ `results/filtering_cc/look_at_cc_observations.txt`
  - ❌ `results/look_at_cc_observations.txt`
- Use descriptive naming: `results/{part}/{task}.txt`

## File Naming Conventions

### Python Files
- One script per task/purpose — clear descriptive names
- Implementation modules: `extract.py`, `explore_cc.py`
- Keep names concise but clear

### Shell Scripts
- Task-based naming: `part_{section}.sh` (e.g., `part_2_1.sh`, `part_2_2.sh`)
- Always `set -e`, always `cd "$(dirname "$0")"`, always echo progress

### Output Files
- Pattern: `results/{part}/{description}.txt` or `.csv`
- Examples:
  - `results/filtering_cc/look_at_cc_observations.txt`
- **NOT**: Files in root `results/` without a subdirectory
- **NOT**: Generic names like `output.txt`, `results.csv`

## Documentation Style

### Minimal Documentation Files
- **ONE README per module subfolder** — not multiple MD files
- Put explanations **inside scripts as comments**, not separate files
- Only create docs when absolutely necessary

### README Contents
- Brief overview
- Quick start commands
- File listing with descriptions
- **NO**: Excessive explanations or tutorials (put in code comments)

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
4. **Output paths**: Default to `results/{part}/` with descriptive names
5. **Auto-create directories**: `output_path.parent.mkdir(parents=True, exist_ok=True)`
6. **Confirm output**: `print(f"Results saved to {output}")`

### Shell Scripts
1. **One script per assignment section/task**
2. **Self-documenting**: All instructions in header comments
3. **Error handling**: Use `set -e` to exit on errors
4. **User feedback**: Echo what's happening at each step
5. **Portable paths**: Use `cd "$(dirname "$0")"` then relative paths

## Platform Considerations

### Development Workflow: Local Testing → Remote Deployment

**Two Environments**:
1. **Local laptop** (WSL2/Windows with RTX 4090, 16 GB VRAM)
   - For unit tests, code development, small-scale smoke tests
   - No GPU needed for most data pipeline work (filtering, deduplication)
   - Large CC file downloads are feasible but slow

2. **Remote instance** (H100, Ubuntu, Together cluster or vast.ai/lambda.ai)
   - For LM training runs and leaderboard submissions
   - Cluster may already have CC files at `/data/CC/` — always soft-link before downloading

**Code Compatibility Requirements**:
- ✅ All scripts must work on **both** environments
- ✅ Use relative paths (not absolute)
- ✅ Check for cluster data directories before downloading
- ✅ No hardcoded environment-specific paths

### Writing Portable Scripts

```python
# ✅ Good: Relative paths
output_path = Path("../../results/filtering_cc/output.txt")
output_path.parent.mkdir(parents=True, exist_ok=True)

# ❌ Bad: Absolute paths
output_path = "/home/user/results/output.txt"
```

```bash
# ✅ Good: Check cluster path first, then download
if [ -f "/data/CC/${FILENAME}" ]; then
    ln -s "/data/CC/${FILENAME}" "${DATA_DIR}/${FILENAME}"
else
    wget "${URL}" -O "${DATA_DIR}/${FILENAME}"
fi
```

### What Runs Where

| Task | Local (RTX 4090) | Remote (H100) |
|------|-----------------|---------------|
| `pytest tests/` | ✅ YES | ✅ YES |
| Data filtering pipeline | ✅ YES | ✅ YES |
| CC file download | ✅ slow | ✅ fast (or symlink) |
| LM training (leaderboard) | ❌ too slow | ✅ YES |

### Line Endings and Permissions

**Always use Unix (LF) line endings** — CRLF breaks bash on both WSL2 and Linux:
```bash
sed -i 's/\r$//' script.sh
chmod +x script.sh
```

## Dependencies and Imports

- Specify all dependencies in `pyproject.toml`
- Use absolute package imports: `from cs336_data.filtering_cc.extract import ...`
- Adapters in `tests/adapters.py` are the interface between tests and your implementation

## Git and Version Control

### What to Commit
- ✅ Source code (`.py`, `.sh`)
- ✅ Configuration files (`pyproject.toml`)
- ✅ One README per module
- ✅ Text observation summaries (small `.txt` files)

### What to Ignore
- ❌ Raw data files (`.warc.gz`, `.wet.gz`) — too large
- ❌ Model files (`.bin`, `.pt`, `.pth`)
- ❌ Result CSVs — regenerable
- ❌ Cache directories (`__pycache__/`, `.pytest_cache/`)

### Important Git Rules
- ❌ **NEVER commit** without explicit approval
- ❌ **NEVER run** `git commit` or `git push` without being asked
- ✅ **ALWAYS show** `git status` and ask before staging anything

## Example: Good vs Bad Organization

### ❌ Bad
```
assignment4-data/
├── observations.txt          # Cluttered root
├── explore.py                # Floating script with no module home
├── cs336_data/
│   ├── __init__.py
│   └── extract.py            # Not in a subfolder
```

### ✅ Good
```
assignment4-data/
├── CODING_PREFERENCES.md     # Essential reference doc
├── Requirements.md           # Assignment requirements
├── data/CC/                  # Raw data (gitignored)
├── results/
│   └── filtering_cc/         # Part-specific outputs
│       └── look_at_cc_observations.txt
└── cs336_data/
    └── filtering_cc/         # Part 2 implementation
        ├── extract.py
        ├── explore_cc.py
        ├── part_2_1.sh
        └── README.md
```

---

**For Future Sessions**: Import this file at the start to understand organizational preferences and coding style.