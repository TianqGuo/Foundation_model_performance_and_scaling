# Results Directory

This directory contains benchmark and profiling results from the cs336_systems module.

## File Naming Convention

All CSV files in this directory follow the naming pattern:
```
results/<module_name>_<script_name>.csv
```

### Profiling and Benchmarking Results

- `profiling_benchmarking_direct.csv` - Results from `benchmark_direct.py` (direct benchmarking sweep)
- `profiling_benchmarking_run_benchmarks.csv` - Results from `run_benchmarks.py` (subprocess-based sweep)
- `profiling_benchmarking_warmup_comparison.csv` - Results from `warmup_comparison.py` (warmup effect analysis)
- `profiling_benchmarking_ctx*.csv` - Context length sweep results
- `profiling_benchmarking_main.csv` - Legacy/manual benchmark results

### Custom Output Files

You can specify custom output paths using the `--output` flag:
```bash
uv run python -m cs336_systems.profiling_benchmarking.benchmark_direct \
    --output results/my_custom_benchmark.csv
```

## Default Behavior

All benchmarking scripts now default to saving results in this directory with descriptive filenames. If you don't specify an `--output` flag, results will be saved here automatically.

## Git Ignore

This directory is included in `.gitignore` to prevent committing large result files. Add specific result files to version control manually if needed.
