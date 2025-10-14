# Learning Rate Sweep - Performance Issue Diagnosis

## Problem Summary

The learning rate sweep is taking **12+ hours** instead of the expected **~4 hours** on GPU.

## Root Cause

Training throughput is **10-20× slower than expected**:

### Current Status (from diagnostic)

```
GPU: NVIDIA GeForce RTX 4090 Laptop GPU (16 GB)
CUDA: Available ✓

Experiment: lr_1e_04
  Progress: 2.5% (251/10000 steps)
  Time: 4.6 hours
  Throughput: 501 tokens/sec  ❌ WAY TOO SLOW

Experiment: lr_3e_04
  Progress: 2.5% (251/10000 steps)
  Time: 5.2 hours
  Throughput: 438 tokens/sec  ❌ WAY TOO SLOW
```

### Expected Performance

- **RTX 4090**: Should get 5,000-10,000 tokens/sec
- **Current**: Getting only 400-500 tokens/sec
- **Slowdown**: 10-20× slower than expected

### Time Impact

At current speed:
- **Per experiment**: ~50-60 hours (instead of 30-40 minutes)
- **Full sweep (7 LRs)**: ~350-400 hours (instead of 4 hours)

This is **completely unacceptable**.

## Potential Causes

### 1. &#x1F50D; Torch Compilation Not Working

The code uses `torch.compile()` but it may not be working effectively:

```python
# In run_experiment.py
if device.type == "cuda":
    model = torch.compile(model)  # May not be working!
```

**Check**: Look for "Compiling model..." messages in logs. If compilation fails silently, this could cause 5-10× slowdown.

### 2. &#x1F50D; Data Loading Bottleneck

Even though using `memmap`, there could be issues:
- Data not properly memory-mapped
- Copying overhead from numpy to torch
- Device transfers happening inefficiently

**Check**: Profile a single training step to see where time is spent.

### 3. &#x1F50D; Inefficient Data Transfer to GPU

The `get_batch` function converts numpy to torch then moves to device:

```python
inputs = torch.tensor(batch_inputs, dtype=torch.long, device=device)
targets = torch.tensor(batch_targets, dtype=torch.long, device=device)
```

If this happens on every batch, it could be slow.

### 4. &#x1F50D; Laptop GPU Thermal Throttling

RTX 4090 **Laptop** GPU may be thermal throttling:
- Clock speeds reduced due to heat
- Performance significantly degraded
- More common in laptops than desktops

**Check**: Monitor GPU clocks and temperature during training:
```bash
nvidia-smi dmon -s pucvmet
```

### 5. &#x1F50D; Mixed Precision Not Used

Training in full float32 without mixed precision:
- Slower than float16/bfloat16
- Uses more memory (less efficient)
- No TF32 acceleration

**Check**: Config shows `"dtype": "float32"` - should use at least TF32 on Ampere+ GPUs.

## Solutions

### Immediate Actions

#### 1. Stop Current Sweep

The current sweep will take 350+ hours. Stop it:

```bash
# Find the process
ps aux | grep learning_rate_sweep

# Kill it (replace PID)
kill <PID>
```

#### 2. Enable Performance Optimizations

Create a new configuration with optimizations:

**Option A: Enable TF32 (Recommended for RTX 4090)**

```python
# Add to run_experiment.py before training loop
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
```

**Option B: Use Mixed Precision (More aggressive)**

Change config:
```json
{
  "dtype": "bfloat16"  // or "float16"
}
```

#### 3. Verify Compilation

Add debug output to check if compilation is working:

```python
# In run_experiment.py
print(f"Compiling model on {device}...")
model = torch.compile(model)
print("Model compiled successfully!")
```

#### 4. Profile a Single Step

Run a quick test to profile performance:

```bash
# Create a test script that runs just 10 steps
uv run python -m cs336_basics.assignment_experiments.lr_sweep.learning_rate_sweep \
    --quick-test \
    --learning-rates 3e-4
```

Check tokens/sec in output - should be >5000.

#### 5. Check Thermal Throttling

Monitor during training:

```bash
# Terminal 1: Run training
uv run python -m cs336_basics.assignment_experiments.lr_sweep.learning_rate_sweep ...

# Terminal 2: Monitor GPU
watch -n 1 nvidia-smi
```

Look for:
- GPU utilization should be 95-100%
- Temperature should be <85°C
- Power draw should be near max TDP
- Clock speeds should be stable

If temperature >85°C and clocks are dropping: **thermal throttling**

**Solutions for throttling**:
- Improve laptop cooling (cooling pad, elevate laptop)
- Reduce power limit slightly for better sustained performance
- Run in a cooler environment

### Long-term Solutions

#### 1. Use Desktop GPU or Cloud GPU

Laptop GPUs are not ideal for long training runs:
- Thermal limitations
- Power constraints
- Slower memory bandwidth

Consider:
- Desktop RTX 4090 (better cooling)
- Cloud GPU (H100, A100, even RTX 4090)
- University/lab GPU cluster

#### 2. Optimize Model Implementation

Profile and optimize:
- Use fused operations
- Optimize attention implementation
- Use FlashAttention if possible

#### 3. Reduce Training Budget (Low-Resource Mode)

Use `--low-resource` flag:
- 40M tokens instead of 327M (8× less)
- 5,000 steps instead of 10,000 (2× less)
- Still achieve meaningful results

At 400 tok/s, this would take ~7 hours per LR instead of 50 hours.

## Recommended Next Steps

1. **Stop current sweep** (it won't finish in reasonable time)

2. **Run diagnostic profile**:
   ```bash
   # Test one LR with just 100 steps
   # Monitor GPU usage and throughput
   ```

3. **Enable TF32** (easy, 2-3× speedup):
   ```python
   torch.backends.cuda.matmul.allow_tf32 = True
   ```

4. **Check for thermal throttling**:
   ```bash
   nvidia-smi dmon -s pucvmet
   ```

5. **If still slow after optimizations**:
   - Use `--low-resource` mode (7-10 hours total vs 350 hours)
   - Or move to better GPU setup (desktop or cloud)

## Expected Results After Fixes

With optimizations enabled:

| Configuration | Tokens/sec | Time per LR | Total Sweep |
|--------------|------------|-------------|-------------|
| Current (broken) | 400-500 | 50-60h | 350-400h ❌ |
| TF32 enabled | 1,500-2,000 | 15-20h | 100-140h ⚠️ |
| TF32 + compilation fixed | 3,000-4,000 | 7-10h | 50-70h ⚠️ |
| Optimal (desktop GPU) | 5,000-10,000 | 3-5h | 20-35h ✓ |
| Low-resource mode | 400-500 | 7h | 50h ⚠️ |

Even with fixes, laptop GPU may struggle. **Strongly recommend**:
- Use desktop GPU if available
- Use cloud GPU (even colab free tier may be faster)
- Use low-resource mode as fallback

## Files to Modify

1. **`cs336_basics/basics/run_experiment.py`** - Add TF32 enable, debug compilation
2. **`cs336_basics/assignment_experiments/lr_sweep/learning_rate_sweep.py`** - Add performance flags
3. Create performance test script to verify optimizations work

## Contact/Questions

If performance doesn't improve after these steps:
1. Share output of diagnostic script
2. Share output of `nvidia-smi dmon` during training
3. May need to investigate model implementation or data loading code
