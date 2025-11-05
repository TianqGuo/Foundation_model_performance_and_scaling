# Memory Profiling Module

This module provides tools to profile GPU memory usage of Transformer models using PyTorch's built-in memory profiler. The output can be visualized at [pytorch.org/memory_viz](https://pytorch.org/memory_viz).

## Overview

This module implements section 1.1.6 (Memory Profiling) of Assignment 2. It profiles memory usage during:
- Forward pass only
- Full training step (forward + backward + optimizer)
- Mixed precision (BF16/FP16) vs FP32 comparison

## Quick Start

Run all profiling tasks for the writeup:

```bash
cd cs336_systems/memory_profiling
./run_all.sh
```

This will generate memory snapshots and analysis reports for all parts (a-d) of section 1.1.6.

## Files

### Python Scripts
- `profile_memory.py` - Main memory profiling script with CLI interface
- `analyze_activation_size.py` - Calculate theoretical activation tensor sizes (part d)

### Shell Scripts
- `run_all.sh` - Master script to run all profiling tasks
- `profile_part_a.sh` - Part (a): Forward and training step profiling
- `profile_part_b.sh` - Part (b): Peak memory across context lengths
- `profile_part_c.sh` - Part (c): Mixed precision memory comparison

## Usage

### Individual Profiling Runs

**Profile forward pass only:**
```bash
python -m cs336_systems.memory_profiling.profile_memory \
    --model-size 2.7B \
    --context-length 512 \
    --profile-type forward
```

**Profile full training step:**
```bash
python -m cs336_systems.memory_profiling.profile_memory \
    --model-size 2.7B \
    --context-length 512 \
    --profile-type training
```

**Profile with mixed precision:**
```bash
python -m cs336_systems.memory_profiling.profile_memory \
    --model-size 2.7B \
    --context-length 512 \
    --profile-type training \
    --use-mixed-precision
```

### Command-Line Arguments

**Required:**
- `--model-size`: Model size (small, medium, large, xl, 2.7B)

**Optional:**
- `--context-length`: Sequence length (default: 512)
- `--batch-size`: Batch size (default: 4)
- `--warmup-steps`: Warmup steps before profiling (default: 5)
- `--measure-steps`: Number of steps to profile (default: 10)
- `--profile-type`: What to profile - 'forward' or 'training' (default: forward)
- `--use-mixed-precision`: Enable mixed precision
- `--dtype`: Mixed precision dtype - 'bf16' or 'fp16' (default: bf16)
- `--output`: Output pickle file path
- `--device`: Device to use (default: cuda)

## Running by Part

### Part (a): Forward and Training Step Visualization

Generate memory snapshots for visualization:

```bash
./profile_part_a.sh
```

**Outputs:**
- `results/memory_profiling/2.7B_ctx512_forward_snapshot.pickle`
- `results/memory_profiling/2.7B_ctx512_training_snapshot.pickle`

**For writeup:**
1. Open [pytorch.org/memory_viz](https://pytorch.org/memory_viz)
2. Drag and drop the pickle files
3. Take screenshots of the "Active memory timeline"
4. Describe how timeline peaks align with execution stages (2-3 sentences)

### Part (b): Peak Memory Across Context Lengths

Profile peak memory for context lengths 128, 256, 512:

```bash
./profile_part_b.sh
```

**Output:**
- `results/memory_profiling/peak_memory_summary.txt` - Table of peak memory usage

**For writeup:**
- Include the table with forward pass and full training step memory for each context length

### Part (c): Mixed Precision Memory Impact

Compare FP32 vs BF16 memory usage:

```bash
./profile_part_c.sh
```

**Output:**
- `results/memory_profiling/mixed_precision_memory_summary.txt` - Comparison table

**For writeup:**
- Discuss whether mixed precision significantly affects memory usage (2-3 sentences)
- Compare forward-only vs full training step memory savings
- Explain why optimizer states limit memory savings

### Part (d): Activation Tensor Size Calculation

Calculate theoretical activation size:

```bash
python -m cs336_systems.memory_profiling.analyze_activation_size
```

**Output:**
- Prints detailed derivation and final answer

**For writeup:**
- Single-precision size (MB) of one residual stream activation tensor
- Include 1-2 sentence derivation

### Part (e): Largest Allocations Analysis

**Manual steps:**
1. Open a memory snapshot at [pytorch.org/memory_viz](https://pytorch.org/memory_viz)
2. Set "Detail" slider to 10% to show only largest allocations
3. Click on large allocations to view stack traces
4. Note the sizes and originating code paths

**For writeup:**
- Size of largest allocations
- Where they originate (from stack trace) - 1-2 sentences

## Output Structure

All outputs are saved to `results/memory_profiling/`:

```
results/memory_profiling/
├── 2.7B_ctx128_forward_snapshot.pickle
├── 2.7B_ctx128_training_snapshot.pickle
├── 2.7B_ctx256_forward_snapshot.pickle
├── 2.7B_ctx256_training_snapshot.pickle
├── 2.7B_ctx512_forward_snapshot.pickle
├── 2.7B_ctx512_training_snapshot.pickle
├── 2.7B_ctx512_forward_fp32_snapshot.pickle
├── 2.7B_ctx512_forward_bf16_snapshot.pickle
├── 2.7B_ctx512_training_fp32_snapshot.pickle
├── 2.7B_ctx512_training_bf16_snapshot.pickle
├── peak_memory_summary.txt
├── mixed_precision_memory_summary.txt
└── activation_size_analysis.txt
```

## Visualization

All `.pickle` files can be visualized at [pytorch.org/memory_viz](https://pytorch.org/memory_viz):

1. Open the website in a browser
2. Drag and drop a `.pickle` file onto the page
3. Explore the "Active memory timeline" view
4. Use the "Detail" slider to filter small allocations
5. Click on allocations to see stack traces

**Key features:**
- **Active memory timeline**: Shows memory usage over time
- **Detail slider**: Filters allocations by size (10% = top 10% largest)
- **Stack traces**: Click allocations to see where they originate
- **Zoom/pan**: Interact with timeline to examine specific regions

## Requirements

- PyTorch with CUDA support
- GPU with sufficient memory (~40GB for 2.7B model)
- BF16 support (A100, H100) for optimal mixed precision (will fall back to FP16)

## Platform Notes

**Local testing (RTX 4090 / small GPU):**
```bash
# Test with small model first
python -m cs336_systems.memory_profiling.profile_memory \
    --model-size small --context-length 512 --profile-type forward
```

**Production runs (H100 / large GPU):**
```bash
# Run all tasks
./run_all.sh
```

## Troubleshooting

**Out of memory errors:**
- Use smaller `--context-length`
- Add `--use-mixed-precision` flag
- Test with smaller models first

**BF16 not supported:**
- Script automatically falls back to FP16
- Check GPU capabilities: `torch.cuda.is_bf16_supported()`

**Missing memory snapshots:**
- Ensure profiling completes without OOM errors
- Check that `results/memory_profiling/` directory is created
- Verify pickle files are generated (should be 5-50 MB each)

## Implementation Details

### Memory Profiling Process

1. **Warmup phase** (not profiled):
   - Run several iterations to stabilize GPU state
   - Cache compilation, load libraries, etc.

2. **Start memory recording**:
   - `torch.cuda.memory._record_memory_history(max_entries=1000000)`
   - Tracks all GPU memory allocations

3. **Measurement phase** (profiled):
   - Run forward pass or training step
   - Record peak memory usage

4. **Save snapshot**:
   - `torch.cuda.memory._dump_snapshot(path)`
   - Creates pickle file with allocation history

5. **Stop recording**:
   - `torch.cuda.memory._record_memory_history(enabled=None)`

### Mixed Precision Implementation

Uses `torch.autocast` context manager:
```python
if use_mixed_precision:
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
else:
    autocast_ctx = nullcontext()

with autocast_ctx:
    output = model(input_ids)
```

This automatically casts operations to lower precision while keeping accumulations in FP32.

## References

- [PyTorch Memory Profiler Documentation](https://pytorch.org/docs/stable/torch_cuda_memory.html)
- [PyTorch Memory Visualizer](https://pytorch.org/memory_viz)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
