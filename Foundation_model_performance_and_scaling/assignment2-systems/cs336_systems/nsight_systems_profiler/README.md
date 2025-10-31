## Nsight Systems Profiling Scripts

This directory contains scripts for profiling Transformer models with NVIDIA Nsight Systems (Assignment 1.1.4).

### Quick Start

```bash
cd cs336_systems/nsight_systems_profiler

# Run specific parts (each profiles ALL 5 model sizes)
./profile_part_a.sh    # Part (a): Forward pass timing - 5 models
./profile_part_b_c.sh  # Parts (b) & (c): Kernel analysis - 5 models × 2 profiles
./profile_part_d.sh    # Part (d): Training step - 5 models
./profile_part_e.sh    # Part (e): Attention components - 5 models

# Extract text summaries (commit these, not binary files)
./extract_all_analyses.sh    # NVTX timing → ANALYSIS_SUMMARY.txt
./export_stats_reports.sh    # Kernel statistics → stats_reports/

# Or profile everything (models × contexts × types)
./profile_all.sh
```

### Model Coverage

**All scripts now profile all 5 model sizes**: small, medium, large, xl, 2.7B

- `profile_part_a.sh` - 5 models × forward pass = **5 profiles**
- `profile_part_b_c.sh` - 5 models × (forward + forward+backward) = **10 profiles**
- `profile_part_d.sh` - 5 models × training = **5 profiles**
- `profile_part_e.sh` - 5 models × attention = **5 profiles**
- Total: **25 profiles** covering all assignment requirements

### Files

**Profiling Scripts**:
- `profile_model.py` - Main profiling script with NVTX annotations
- `annotated_attention.py` - NVTX-annotated attention for detailed profiling
- `profile_part_a.sh` - Forward pass profiling (question a)
- `profile_part_b_c.sh` - Kernel analysis (questions b & c)
- `profile_part_d.sh` - Training step profiling (question d)
- `profile_part_e.sh` - Attention component analysis (question e)
- `profile_all.sh` - Profile all models, contexts, and types

**Analysis Scripts**:
- `analyze_wsl_profiles.py` - Extract NVTX timing from .sqlite files
- `extract_all_analyses.sh` - Batch process all profiles → ANALYSIS_SUMMARY.txt
- `export_stats_reports.sh` - Extract kernel statistics → stats_reports/

### Assignment Questions

**Part (a):** Compare nsys forward pass timing with Python timing
**Part (b):** Find most expensive CUDA kernel and invocation count
**Part (c):** Identify non-matmul kernels taking significant time
**Part (d):** Compare kernel distribution: inference vs training
**Part (e):** Compare softmax vs matmul runtime/FLOPs in attention

### Output Structure

```
results/nsight_profiles/
├── ANALYSIS_SUMMARY.txt          # NVTX timing (commit this)
├── stats_reports/                # Kernel statistics (commit this)
│   ├── part_a/
│   ├── part_b_c/
│   ├── part_d/
│   └── part_e/
├── part_a/
│   ├── small_forward_ctx512.nsys-rep     # Binary (don't commit)
│   ├── medium_forward_ctx512.nsys-rep
│   ├── large_forward_ctx512.nsys-rep
│   ├── xl_forward_ctx512.nsys-rep
│   └── 2.7B_forward_ctx512.nsys-rep
├── part_b_c/
│   ├── small_forward_annotated.nsys-rep
│   ├── small_forward_backward_annotated.nsys-rep
│   └── ... (10 profiles total)
├── part_d/
│   └── ... (5 training profiles)
└── part_e/
    └── ... (5 attention profiles)
```

**Total output**: 25 binary profiles + text summaries

### Working with Results

**On H100 (after profiling)**:
```bash
# Extract text summaries
./extract_all_analyses.sh    # Creates ANALYSIS_SUMMARY.txt
./export_stats_reports.sh    # Creates stats_reports/

# Download ONLY text files to local (not 125MB+ binary files)
# From local machine:
scp h100:~/path/results/nsight_profiles/ANALYSIS_SUMMARY.txt ./results/nsight_profiles/
scp -r h100:~/path/results/nsight_profiles/stats_reports/ ./results/nsight_profiles/
```

**Viewing Binary Profiles**:
1. Download `.nsys-rep` files to local Windows machine (if needed for GUI)
2. Open with Nsight Systems GUI
3. Use NVTX ranges to filter (exclude warmup)
4. Check "CUDA GPU Kernel Summary" in Stats System View

### Manual Usage

```bash
# Basic profiling
uv run nsys profile -o output.nsys-rep \
  python -m cs336_systems.nsight_systems_profiler.profile_model \
  --model-size small \
  --context-length 512 \
  --profile-type forward

# With annotated attention
uv run nsys profile -o output.nsys-rep \
  python -m cs336_systems.nsight_systems_profiler.profile_model \
  --model-size small \
  --use-annotated-attention

# Full options
python -m cs336_systems.nsight_systems_profiler.profile_model --help
```

### Options

- `--model-size`: small, medium, large, xl, 2.7B
- `--context-length`: 128, 256, 512, 1024
- `--batch-size`: Default 4
- `--profile-type`: forward, forward_backward, training
- `--use-annotated-attention`: Add detailed NVTX ranges to attention
- `--warmup-steps`: Steps before profiling (default 5)
- `--measure-steps`: Steps to profile (default 10)

### NVTX Ranges

The profiler adds NVTX ranges for:
- `warmup` - Warmup steps (filter these out)
- `forward` / `backward` / `optimizer_step` - Major phases
- `step_N` - Individual measurement steps
- `scaled_dot_product_attention` - Attention layers (with --use-annotated-attention)
  - `computing_attention_scores` - Q @ K^T
  - `computing_softmax` - Softmax
  - `final_matmul` - Attention @ V

### Requirements

- NVIDIA Nsight Systems installed (`nsys` command)
- CUDA-capable GPU
- H100 80GB recommended for all models (xl/2.7B require high memory)

### Environment Compatibility

**All scripts work on both**:
- ✅ Local WSL2 (testing with small models)
- ✅ H100 Lightning AI instance (production with all models)

**Local WSL2 Limitations**:
- NVTX timing works correctly ✓
- GPU kernel data may not be captured (virtualized driver)
- Use for testing script logic, then run on H100 for complete data

**H100 (Recommended)**:
- Complete kernel data capture ✓
- Sufficient memory for all 5 model sizes ✓
- Native Linux profiling capabilities ✓

**Workflow**: Test locally → Deploy to H100 → Download text summaries

### Runtime Estimates

On H100 instance:
- `profile_part_a.sh`: ~10-15 min (5 models)
- `profile_part_b_c.sh`: ~30-40 min (10 profiles)
- `profile_part_d.sh`: ~15-20 min (5 models)
- `profile_part_e.sh`: ~15-20 min (5 models)
- **Total**: ~70-95 minutes for all parts

Note: xl and 2.7B models take longer than smaller models.
