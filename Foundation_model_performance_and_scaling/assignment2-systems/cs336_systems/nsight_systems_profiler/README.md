## Nsight Systems Profiling Scripts

This directory contains scripts for profiling Transformer models with NVIDIA Nsight Systems (Assignment 1.1.4).

### Quick Start

```bash
cd cs336_systems/nsight_systems_profiler

# Run specific parts
./profile_part_a.sh    # Part (a): Forward pass timing
./profile_part_b_c.sh  # Parts (b) & (c): Kernel analysis
./profile_part_d.sh    # Part (d): Training step
./profile_part_e.sh    # Part (e): Attention components

# Or profile everything (1-3 hours)
./profile_all.sh
```

### Files

- `profile_model.py` - Main profiling script with NVTX annotations
- `annotated_attention.py` - NVTX-annotated attention for detailed profiling
- `profile_part_a.sh` - Forward pass profiling (question a)
- `profile_part_b_c.sh` - Kernel analysis (questions b & c)
- `profile_part_d.sh` - Training step profiling (question d)
- `profile_part_e.sh` - Attention component analysis (question e)
- `profile_all.sh` - Profile all models and contexts

### Assignment Questions

**Part (a):** Compare nsys forward pass timing with Python timing
**Part (b):** Find most expensive CUDA kernel and invocation count
**Part (c):** Identify non-matmul kernels taking significant time
**Part (d):** Compare kernel distribution: inference vs training
**Part (e):** Compare softmax vs matmul runtime/FLOPs in attention

### Output

Profiles saved to: `../../results/nsight_profiles/`

### Viewing Profiles

1. Download `.nsys-rep` files to your local machine
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
- A100 40GB recommended for xl/2.7B models
