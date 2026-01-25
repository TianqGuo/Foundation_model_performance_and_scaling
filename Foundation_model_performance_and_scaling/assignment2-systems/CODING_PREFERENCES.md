# Coding Preferences and Project Organization

This document captures my preferred coding style and project organization for future reference.

## Directory Structure

### Core Principle: Keep Root Clean
- **Root directory**: Only essential files (READMEs, config files, main scripts)
- **DO NOT clutter root** with temporary files, logs, or module-specific documentation
- Each module should be self-contained in its own subdirectory

### Module Organization
```
cs336_systems/
├── profiling_benchmarking/     # Benchmarking module
│   ├── *.py                     # Python implementation files
│   ├── *.sh                     # Shell scripts for running tasks
│   └── README.md                # Single README with usage info
│
├── nsight_systems_profiler/     # Profiling module
│   ├── *.py                     # Python implementation files
│   ├── *.sh                     # Shell scripts for each task
│   └── README.md                # Single README with usage info
│
└── [future_module]/             # Each new task gets its own folder
```

### Results and Output
```
results/                         # All output files
├── profiling_benchmarking/      # Organized by module
│   ├── separate.csv
│   └── warmup_comparison.csv
├── nsight_profiles/             # Organized by module
│   ├── part_a/
│   ├── part_b_c/
│   └── ANALYSIS_SUMMARY.txt
├── mixed_precision/             # Organized by module
│   └── mixed_precision_benchmark.csv
└── [module_name]/               # Each module gets its own subdirectory
```

**Key principles:**
- All results go to `results/` directory
- **IMPORTANT**: Each module gets its own subdirectory in `results/`
  - ✅ `results/mixed_precision/mixed_precision_benchmark.csv`
  - ❌ `results/mixed_precision_benchmark.csv`
- Use descriptive naming: `results/{module}/{script}.csv`
- Organized by module and task for scalability

## File Naming Conventions

### Python Files
- Module files: `{function}_module.py` (e.g., `profile_model.py`, `benchmark_separate.py`)
- Implementation files: descriptive names without prefixes
- Keep names concise but clear

### Shell Scripts
- Task-based naming: `part_{letter}.sh` for assignment questions
- Example: `part_b.sh`, `part_c.sh`
- Alternative: `profile_part_a.sh` if module name needed for clarity

### Output Files
- Pattern: `results/{module}/{description}.csv`
- Examples:
  - `results/profiling_benchmarking/separate.csv`
  - `results/profiling_benchmarking/warmup_comparison.csv`
  - `results/mixed_precision/mixed_precision_benchmark.csv`
- **NOT**: All files in root `results/` directory
- **NOT**: Generic names like `results.csv`, `results_main.csv`

## Documentation Style

### Minimal Documentation Files
- **ONE README per module** - not multiple MD files
- Put explanations **inside scripts as comments**, not separate files
- Only create docs when absolutely necessary

### README Contents
- Brief overview
- Quick start commands
- File listing with descriptions
- Usage examples
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
2. **Reusable functions**: Import from shared modules
3. **Command-line arguments**: Use argparse with sensible defaults
4. **Output paths**: Default to `results/` with descriptive names

### Shell Scripts
1. **One script per assignment question/task**
2. **Self-documenting**: All instructions in comments
3. **Error handling**: Use `set -e` to exit on errors
4. **User feedback**: Echo what's happening, show progress

### Example Pattern
```python
# Good: Separate script for each task
benchmark_separate.py       # Measures forward/backward separately
warmup_comparison.py        # Tests warmup effect

# Bad: One giant script with modes
benchmark.py --mode separate --mode warmup  # Too many responsibilities
```

## Platform Considerations

### Development Workflow: Local Testing → H100 Deployment

**Two Environments**:
1. **Local laptop** (WSL2/Windows with RTX 4090)
   - For testing code logic and small-scale runs
   - Limited GPU memory (16 GB)
   - WSL2 has GPU driver limitations (e.g., nsys profiling)

2. **Lightning AI H100 instance** (Ubuntu with H100)
   - For production runs and full profiling
   - High GPU memory (80 GB)
   - Native Linux - full GPU access

**Code Compatibility Requirements**:
- ✅ All scripts must work on **both** environments
- ✅ Use relative paths (not absolute)
- ✅ Auto-detect capabilities (e.g., GPU memory)
- ✅ Graceful degradation (e.g., skip large models on small GPU)
- ✅ No hardcoded environment-specific paths

### Writing Portable Scripts

**Python Scripts**:
```python
# ✅ Good: Auto-detect GPU memory
import torch
gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

if gpu_memory_gb < 20:
    print("WARNING: Limited GPU memory, skipping xl/2.7B models")
    model_sizes = ["small", "medium", "large"]
else:
    model_sizes = ["small", "medium", "large", "xl", "2.7B"]

# ✅ Good: Relative paths
output_path = Path("../../results/output.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

# ❌ Bad: Absolute paths
output_path = "/home/user/results/output.csv"  # Won't work on different machines
```

**Shell Scripts**:
```bash
# ✅ Good: Check capabilities before running
if ! command -v nsys &> /dev/null; then
    echo "WARNING: nsys not found, skipping profiling"
    exit 0
fi

# ✅ Good: Auto-detect WSL2
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "Running in WSL2 - some features may be limited"
fi

# ✅ Good: Relative paths from script location
cd "$(dirname "$0")"
OUTPUT_DIR="../../results"
```

### What Runs Where: Local vs Remote

**Local Laptop (Single RTX 4090)**:
- ✅ **Unit tests** - Correctness verification (e.g., `pytest tests/`)
- ✅ **Code development** - Writing and debugging implementations
- ✅ **Small-scale testing** - Verify scripts run without errors
- ❌ **Multi-GPU benchmarks** - Requires 2+ GPUs (DDP, distributed training)
- ❌ **Large model benchmarks** - XL/2.7B models may OOM on 16GB GPU
- ❌ **Performance profiling** - WSL2 has limited nsys support

**Remote Multi-GPU Instance (H100s on vast.ai/lambda.ai)**:
- ✅ **Performance benchmarks** - Actual timing measurements for assignments
- ✅ **Multi-GPU training** - DDP, FSDP, distributed implementations
- ✅ **Large models** - XL/2.7B models with 80GB GPU memory
- ✅ **Profiling** - Full nsys, nvprof access on native Linux

**Quick Reference Table**:

| Task Type | Local (RTX 4090) | Remote (H100s) |
|-----------|------------------|----------------|
| `pytest tests/` | ✅ YES | ✅ YES |
| Single-GPU benchmarks (small/medium models) | ✅ YES | ✅ YES |
| Multi-GPU benchmarks (2+ GPUs) | ❌ NO | ✅ YES |
| Large model benchmarks (XL/2.7B) | ❌ NO (OOM) | ✅ YES |
| Performance measurements for reports | ❌ NO | ✅ YES |
| Nsys profiling | ❌ NO (WSL2) | ✅ YES |

**Examples**:
```bash
# ✅ Local: Run unit tests (correctness only)
uv run pytest tests/test_ddp.py -v

# ❌ Local: Don't benchmark multi-GPU performance
# This requires 2+ GPUs - run on remote instance instead!
# uv run python benchmark_bucketed.py --world-size 2

# ✅ Remote (H100 instance): Benchmark with 2 GPUs
uv run python benchmark_bucketed.py --model-size xl --world-size 2 --bucket-sizes 1 10 100 1000
```

### Testing Workflow

**1. Test locally (WSL2/RTX 4090)**:
```bash
# Test correctness with unit tests
uv run pytest tests/test_ddp.py -v

# Verify script runs without errors (small models only)
python benchmark.py --model-size small --context-length 512
```

**2. Deploy to H100**:
```bash
# Copy entire module folder
scp -r cs336_systems/module_name/ h100:~/assignment2-systems/cs336_systems/

# Run full benchmark suite on H100
ssh h100
cd ~/assignment2-systems/cs336_systems/module_name
./run_benchmark.sh  # Runs with multiple GPUs and large models
```

**3. Download results**:
```bash
# Download only text summaries (not large binary files)
scp h100:~/assignment2-systems/results/**/*.txt ./results/
scp h100:~/assignment2-systems/results/**/*.csv ./results/

# Don't download: *.nsys-rep, *.sqlite (too large)
```

### Line Endings and Permissions

**Always use Unix (LF) line endings**:
```bash
# Fix line endings when creating files
sed -i 's/\r$//' script.sh  # Remove Windows \r

# Make executable
chmod +x script.sh
```

**Why?**: Both WSL2 and native Linux require LF endings. CRLF causes:
```
bash: ./script.sh: /bin/bash^M: bad interpreter
```

## Dependencies and Imports

### Module Dependencies
- Clearly specify in `pyproject.toml`
- Use relative imports within modules
- Import from installed packages (e.g., `cs336_basics`)

### Example Import Pattern
```python
# Within module
from .benchmark import MODEL_CONFIGS, create_model

# From other assignment modules
from cs336_basics.transformer_training.model import TransformerLM
```

## Output and Results Management

### Default Behavior
- **Always provide sensible defaults** for output paths
- **Create directories automatically** if they don't exist
- **Don't fail silently**: Print where files are saved

### Example Pattern
```python
parser.add_argument(
    "--output",
    type=str,
    default="results/module_name/output.csv",  # Good default - organized by module
    help="Output CSV file path (default: results/module_name/output.csv)",
)

# In code:
output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)  # Auto-create subdirectory
df.to_csv(args.output, index=False)
print(f"Results saved to {args.output}")  # Confirm to user
```

**Key principle**: Always organize results by module subdirectory for scalability:
- ✅ `results/mixed_precision/benchmark.csv` - Easy to find module-specific results
- ❌ `results/mixed_precision_benchmark.csv` - Root directory becomes cluttered

## What to Ask Before Creating Files

### Always Clarify
1. **Where to put it?** - Existing module or new folder?
2. **Is this temporary?** - Should it be gitignored?
3. **Is documentation needed?** - Or can it go in code comments?

### General Rules
- ❌ **Don't create** extra documentation files without asking
- ❌ **Don't create** files in root without asking
- ✅ **Do create** well-organized module folders
- ✅ **Do include** comprehensive inline documentation

## Error Handling

### Shell Scripts
```bash
set -e  # Exit on any error

# Handle OOM gracefully
if run_command; then
    echo "✓ Success"
else
    echo "✗ Failed (likely OOM)"
fi
```

### Python Scripts
```python
try:
    # Main logic
    result = operation()
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("WARNING: Out of memory")
        error_type = "OOM"

    # Clean up gracefully
    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        pass  # Ignore if cleanup fails

    return error_result
```

## Git and Version Control

### What to Commit
- ✅ Source code (`.py`, `.sh`)
- ✅ Configuration files
- ✅ Essential READMEs (one per module)
- ✅ Requirements/dependencies
- ✅ Analysis summaries (`.txt` files with extracted metrics)

### What to Ignore
- ❌ Results files (`.csv`, large data)
- ❌ Binary profiling files (`.nsys-rep`, `.sqlite`)
- ❌ Temporary files
- ❌ Cache directories

### Why Not Commit Large Binary/Result Files?

**Problem**: Large files cause multiple issues:
- ❌ Slow git operations (push/pull/clone)
- ❌ Bloat repository history permanently
- ❌ Binary files may trigger GitHub secret detection (false positives)
- ❌ Waste storage for data that can be regenerated

**Examples**:
- Profiling files: `.nsys-rep`, `.sqlite` (10-50 MB each)
- Model checkpoints: `.pt`, `.pth` (100+ MB)
- Large datasets: `.csv`, `.parquet` (varies)
- Compiled binaries: `.so`, `.dylib`

**Better Alternative**: Extract summaries
- ✅ Commit text summaries (~20 KB) instead of binary files
- ✅ Contains key metrics needed for analysis
- ✅ Human-readable and git-friendly
- ✅ Can regenerate full results anytime by re-running scripts

### .gitignore Pattern
```gitignore
# Results directory - regenerable outputs
results/

# Large binary files
*.nsys-rep
*.sqlite
*.pt
*.pth
*.ckpt

# CSV results (can be large)
*.csv

# BUT keep small text summaries
!results/**/ANALYSIS_SUMMARY.txt
!results/**/*_report.txt
!results/**/*.md

# Caches
__pycache__/
.pytest_cache/
```

### Important Git Rules
- ❌ **NEVER commit** when you're unsure what's staged
- ❌ **NEVER let me run** `git commit` or `git push` directly
- ✅ **ALWAYS ask** before committing files
- ✅ **ALWAYS review** what's staged with `git status` first

## Summary of Key Preferences

1. **Keep root directory clean** - put everything in module folders
2. **One README per module** - not multiple documentation files
3. **Explanations in code comments** - not separate files
4. **Descriptive output names** - `module_task.csv`, not `results.csv`
5. **Results organized by module** - `results/{module}/file.csv`, not `results/file.csv`
6. **Shell script per task** - `part_b.sh`, `part_c.sh`
7. **Self-documenting code** - comprehensive headers and comments
8. **Handle errors gracefully** - especially OOM errors
9. **Fix line endings** - always use Unix LF format
10. **Ask before creating files** - especially in root or documentation

## Example: Good vs Bad Organization

### ❌ Bad
```
assignment2-systems/
├── results_main.csv              # Cluttered root
├── results_warmup.csv
├── QUICK_START.md               # Too many docs
├── DETAILED_GUIDE.md
├── TROUBLESHOOTING.md
├── cs336_systems/
│   └── profiling_benchmarking/
│       └── benchmark.py
```

### ✅ Good
```
assignment2-systems/
├── CODING_PREFERENCES.md         # Essential reference doc
├── results/                      # All outputs here, organized by module
│   ├── profiling_benchmarking/  # Module subdirectory
│   │   ├── separate.csv
│   │   └── warmup_comparison.csv
│   ├── nsight_profiles/         # Module subdirectory
│   │   └── ANALYSIS_SUMMARY.txt
│   └── mixed_precision/         # Module subdirectory
│       └── mixed_precision_benchmark.csv
└── cs336_systems/
    ├── profiling_benchmarking/
    │   ├── benchmark.py
    │   ├── benchmark_separate.py
    │   ├── warmup_comparison.py
    │   ├── part_b.sh            # Scripts with inline docs
    │   ├── part_c.sh
    │   └── README.md            # Single README
    ├── nsight_systems_profiler/
    │   ├── profile_model.py
    │   ├── profile_part_a.sh
    │   └── README.md
    └── mixed_precision/
        ├── accumulation_comparison.py
        ├── run_accumulation.sh
        └── README.md
```

---

**For Future Sessions**: Import this file at the start so I understand your organizational preferences and coding style.
