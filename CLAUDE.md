# Global Coding Preferences

## Module / Folder Structure

Each logical unit of work (a "part" or "feature") lives in its own **module folder**. Within that module, each **section or sub-task** gets its own **subfolder** containing the implementation file(s) and a runner shell script.

```
<project>/
└── <module>/               # one folder per part/feature
    ├── <section_a>/        # one subfolder per section/task
    │   ├── implementation.py
    │   └── part_N_M.sh
    └── <section_b>/
        ├── implementation.py
        └── part_N_M.sh
```

Never place section-level files flat in the module root if they belong in a section subfolder.

### Data Files

```
data/                       # Raw input data (gitignored)
└── <dataset>/
```

### Results

All outputs go under `results/<module>/<file>`, never flat in `results/`.

```
results/
├── <module>/
│   └── task_output.txt        ✅
└── task_output.txt            ❌
```

## Shell Scripts

- One per section, named consistently (e.g. `part_2_3.sh`)
- Header block: USAGE, WHAT IT DOES, OUTPUT, NOTES
- Always: `set -e` and `cd "$(dirname "$0")"`
- Compute root with `ROOT="$(cd <relative-path> && pwd)"` and use absolute `${ROOT}/...` paths throughout — never relative paths that break when `cd` changes directory
- Echo progress at each step
- Check for cluster/remote data paths before downloading:
  ```bash
  if [ -f "/data/<dataset>/${FILENAME}" ]; then
      ln -s "/data/<dataset>/${FILENAME}" "${DATA_DIR}/${FILENAME}"
  else
      wget "${URL}" -O "${DATA_DIR}/${FILENAME}"
  fi
  ```

## Python Scripts

- Use `argparse` with sensible defaults
- Auto-create output directories: `output_path.parent.mkdir(parents=True, exist_ok=True)`
- Print the path of any file saved: `print(f"Results saved to {output}")`
- One script per task/purpose — never one giant script with multiple modes

### Import Style

- Use absolute package imports: `from <package>.<module>.<section> import ...`
- Test adapters import inline inside the function body (lazy imports):
  ```python
  def run_something(text: str):
      from mypackage.module.section import my_function
      return my_function(text)
  ```

### Error Handling

Handle resource errors (OOM, timeout) gracefully rather than crashing:
```python
try:
    result = operation()
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("WARNING: Out of memory, skipping")
        # clean up and return a safe fallback
        return None
```

## README

- One README per module level — not per section
- Section-level notes go in shell script headers or inline code comments
- Do not create extra documentation files without asking

## Assets / Models

- Downloaded model/asset files go in a dedicated `assets/` folder under the module
- Provide a `get_assets.sh` (or equivalent) to download/symlink them
- Always check for cluster copies before downloading
- Never commit large binary files

## Platform: Local vs Remote

**Two environments:**

1. **Local (WSL2/Windows, consumer GPU)**
   - Unit tests, code development, small-scale smoke tests
   - May have limited GPU memory or missing profiling tools (e.g. `nsys` limited under WSL2)

2. **Remote (Linux server/cluster, high-end GPU)**
   - Full training runs, multi-GPU benchmarks, profiling
   - May have data pre-cached at `/data/` — always symlink before downloading

**Code compatibility requirements:**
- All scripts must work on both environments
- Use relative paths (not hardcoded absolute paths)
- Auto-detect capabilities rather than hardcoding hardware-specific values
- Check for cluster data directories before downloading

**Line endings:** Always use Unix (LF) — CRLF breaks bash on both WSL2 and Linux:
```bash
sed -i 's/\r$//' script.sh
chmod +x script.sh
```

## Git and Version Control

### What to Commit
- ✅ Source code (`.py`, `.sh`)
- ✅ Configuration files (e.g. `pyproject.toml`)
- ✅ One README per module
- ✅ Small text summaries and written answers

### What to Ignore
- ❌ Large raw data files — too large, regenerable from source
- ❌ Model checkpoints and binary assets (`.pt`, `.pth`, `.bin`)
- ❌ Large or binary result files (profiling outputs, CSVs) — regenerable
- ❌ Cache directories (`__pycache__/`, `.pytest_cache/`, etc.)

### Rules
- **NEVER commit** without explicit approval
- **NEVER run** `git commit` or `git push` without being asked
- **ALWAYS show** `git status` and ask before staging anything

## Before Creating Any File

Ask:
1. Which module/section subfolder does it belong in?
2. Is it temporary — should it be gitignored?
3. Is documentation needed, or can it go in code comments?

Never place new files in the repo root or module root without asking.
