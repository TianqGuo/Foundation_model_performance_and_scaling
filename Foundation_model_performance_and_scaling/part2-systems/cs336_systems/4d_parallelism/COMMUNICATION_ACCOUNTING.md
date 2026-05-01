# Section 2.4: 4D Parallelism - Communication Accounting

Analysis for the **XXL model configuration** with 4D parallelism considerations.

---

## Model Configuration

**XXL Model Specifications:**
- `d_model = 16,384`
- `d_ff = 53,248`
- `num_blocks = 126`

**Simplifying Assumptions:**
1. Omit attention, input embeddings, and output linear layers
2. Each FFN block contains two linear layers:
   - First layer: `d_model → d_ff` (16,384 → 53,248)
   - Second layer: `d_ff → d_model` (53,248 → 16,384)
3. Model consists of 126 blocks of these two linear layers
4. No activation checkpointing
5. **Activations and gradient communications**: BF16 (2 bytes per parameter)
6. **Accumulated gradients, master weights, optimizer state**: FP32 (4 bytes per parameter)

---

## Parameter Count Calculation

### Parameters per Block:
```
First linear layer:  d_model × d_ff = 16,384 × 53,248 = 872,415,232 parameters
Second linear layer: d_ff × d_model = 53,248 × 16,384 = 872,415,232 parameters

Total per block: 872,415,232 + 872,415,232 = 1,744,830,464 parameters
```

### Total Model Parameters:
```
Total parameters = num_blocks × params_per_block
                = 126 × 1,744,830,464
                = 219,848,638,464 parameters
                ≈ 220 billion parameters
```

---

## Part (a): Memory for Single Device (3 pts)

### Question:
How much memory would it take to store the master model weights, accumulated gradients and optimizer states in FP32 on a single device? How much memory is saved for backward (these will be in BF16)? How many H100 80GB GPUs worth of memory is this?

### Calculation:

**Memory Components (FP32):**

1. **Master Weights (FP32)**:
   ```
   Memory = 220B params × 4 bytes = 880 × 10^9 bytes = 880 GB
   ```

2. **Accumulated Gradients (FP32)**:
   ```
   Memory = 220B params × 4 bytes = 880 × 10^9 bytes = 880 GB
   ```

3. **Optimizer States (AdamW with 2 states: momentum + variance)**:
   ```
   Memory = 2 × 220B params × 4 bytes = 1,760 × 10^9 bytes = 1,760 GB
   ```

**Total Memory (FP32)**:
```
Total = Weights + Gradients + Optimizer States
      = 880 + 880 + 1,760
      = 3,520 GB
```

**Memory Saved Using BF16 for Backward Gradients:**

During backward pass, gradients are computed in BF16 before accumulation to FP32:
```
Gradients in FP32: 880 GB
Gradients in BF16: 220B × 2 bytes = 440 GB
Memory saved: 880 - 440 = 440 GB
```

**Number of H100 GPUs Required:**
```
Number of GPUs = 3,520 GB / 80 GB per GPU = 44 GPUs
```

### Answer:
**Storing master weights, accumulated gradients, and optimizer states in FP32 requires 3,520 GB of memory (equivalent to 44 H100 80GB GPUs); using BF16 for backward pass gradients saves 440 GB compared to FP32 gradients.**

---

## Part (b): Sharding Across FSDP Devices (2 pts)

### Question:
Now assume your master weights, optimizer state, gradients and half of your activations (in practice every second layer) are sharded across `N_FSDP` devices. Write an expression for how much memory this would take per device. What value does `N_FSDP` need to be for the total memory cost to be less than 1 v5p TPU (95GB per device)?

### Memory Expression Per Device:

With FSDP, the following are sharded across `N_FSDP` devices:
- Master weights (FP32): 880 GB
- Gradients (FP32): 880 GB
- Optimizer states (FP32): 1,760 GB
- Half of activations (BF16): sharded

Let `A` = total activation memory for all layers (in GB, BF16).

**Sharded components** (divided across N_FSDP):
- Weights: 880 GB / N_FSDP
- Gradients: 880 GB / N_FSDP
- Optimizer states: 1,760 GB / N_FSDP
- Half of activations: (A/2) / N_FSDP

**Replicated components** (stored on each device):
- Other half of activations: A/2

**Memory per device:**
```
Memory_per_device = (880 + 880 + 1,760 + A/2) / N_FSDP + A/2
                  = (3,520 + A/2) / N_FSDP + A/2
```

### Activation Memory Estimation:

For a batch of size `b` and sequence length `seq_len`:

**Activations per FFN block (BF16):**
- Input to first linear: `b × seq_len × d_model × 2 bytes`
- Output of first linear: `b × seq_len × d_ff × 2 bytes`

**Total activations for 126 blocks:**
```
A = 126 × b × seq_len × (d_model + d_ff) × 2 bytes
  = 126 × b × seq_len × (16,384 + 53,248) × 2
  = 126 × b × seq_len × 69,632 × 2
  = 17,547,264 × b × seq_len bytes
```

For typical training (e.g., `b=4`, `seq_len=2048`):
```
A = 17,547,264 × 4 × 2,048 / 10^9 = 143.7 GB
```

**Memory per device with typical batch:**
```
Memory_per_device = (3,520 + 143.7/2) / N_FSDP + 143.7/2
                  = (3,520 + 71.85) / N_FSDP + 71.85
                  = 3,591.85 / N_FSDP + 71.85
```

### Finding N_FSDP for <95 GB per device:

```
3,591.85 / N_FSDP + 71.85 < 95
3,591.85 / N_FSDP < 23.15
N_FSDP > 155.15

Therefore: N_FSDP ≥ 156 devices
```

**For minimal activation memory** (small batch):
If we consider only the parameter-related memory (worst case with negligible activations):
```
3,520 / N_FSDP < 95
N_FSDP > 37.05

Therefore: N_FSDP ≥ 38 devices (minimum)
```

### Answer:
**Memory per device = `(3,520 + A/2) / N_FSDP + A/2` GB, where A is total activation memory in GB; for typical training with batch size 4 and sequence length 2048 (A ≈ 144 GB), we need N_FSDP ≥ 156 devices to stay under 95 GB per device, though with minimal activations (e.g., batch size 1, short sequences), N_FSDP ≥ 38 devices suffices.**

---

## Part (c): Compute-Bound Batch Size Analysis (3 pts)

### Question:
Consider only the forward pass. Use the communication bandwidth of `W_ici = 2 × 9 × 10^10` bytes/s and FLOPS/s of `C = 4.6 × 10^14` for TPU v5p as given in the TPU Scaling Book. Following the notation of the Scaling Book, use `M_X = 2`, `M_Y = 1` (a 3D mesh), with:
- `X = 16` being your FSDP dimension
- `Y = 4` being your TP dimension

At what per-device batch size is this model compute bound? What is the overall batch size in this setting?

### Given Parameters:
- Communication bandwidth: `W_ici = 2 × 9 × 10^10 = 1.8 × 10^11` bytes/s
- Compute throughput: `C = 4.6 × 10^14` FLOPS/s
- Mesh dimensions: `M_X = 2`, `M_Y = 1`
- FSDP dimension: `X = 16`
- TP dimension: `Y = 4`
- Total devices: `X × Y = 64` devices

### Computation Time (Forward Pass):

**FLOPs for forward pass:**

For one FFN block with batch size `b` and sequence length `s`:
```
First linear:  2 × b × s × d_model × d_ff = 2 × b × s × 16,384 × 53,248
Second linear: 2 × b × s × d_ff × d_model = 2 × b × s × 53,248 × 16,384

Per block: 4 × b × s × 16,384 × 53,248 = 3.488 × 10^12 × b × s FLOPs
Total:     126 × 3.488 × 10^12 × b × s = 4.395 × 10^14 × b × s FLOPs
```

**Per-device computation:**
With data parallelism across X=16 devices (FSDP dimension):
```
FLOPs_per_device = 4.395 × 10^14 × b_device × s FLOPs
```

**Computation time per device:**
```
T_compute = FLOPs_per_device / C
          = (4.395 × 10^14 × b_device × s) / (4.6 × 10^14)
          = 0.955 × b_device × s seconds
```

### Communication Time (Forward Pass):

**FSDP Communication (All-Gather Weights):**

With FSDP, each device holds 1/X of the model weights. For forward pass, need to all-gather weights.

Model size in BF16 (for forward pass):
```
Model_size_BF16 = 220B × 2 bytes = 4.4 × 10^11 bytes
```

Communication volume for all-gather:
```
Volume_FSDP = (X - 1) / X × Model_size_BF16
            = (16 - 1) / 16 × 4.4 × 10^11
            = 0.9375 × 4.4 × 10^11
            = 4.125 × 10^11 bytes
```

Effective bandwidth for FSDP (X dimension):
```
Bandwidth_FSDP = W_ici × M_X = 1.8 × 10^11 × 2 = 3.6 × 10^11 bytes/s
```

FSDP communication time:
```
T_FSDP = Volume_FSDP / Bandwidth_FSDP
       = 4.125 × 10^11 / 3.6 × 10^11
       = 1.146 seconds
```

**TP Communication (All-Reduce Activations):**

With tensor parallelism across Y=4 devices, activations are sharded. Need to all-reduce.

Activation size per layer (largest intermediate, in BF16):
```
Activation_per_layer = b_device × s × d_ff × 2 bytes
                     = b_device × s × 53,248 × 2
                     = 106,496 × b_device × s bytes
```

For 126 blocks, assuming all-reduce after each block:
```
Volume_TP_per_block = 2 × (Y - 1) / Y × 106,496 × b_device × s
                    = 2 × 3/4 × 106,496 × b_device × s
                    = 159,744 × b_device × s bytes

Total_Volume_TP = 126 × 159,744 × b_device × s
                = 2.013 × 10^7 × b_device × s bytes
```

Effective bandwidth for TP (Y dimension):
```
Bandwidth_TP = W_ici × M_Y = 1.8 × 10^11 × 1 = 1.8 × 10^11 bytes/s
```

TP communication time:
```
T_TP = Total_Volume_TP / Bandwidth_TP
     = (2.013 × 10^7 × b_device × s) / (1.8 × 10^11)
     = 1.118 × 10^-4 × b_device × s seconds
```

**Total Communication Time:**
```
T_comm = T_FSDP + T_TP
       = 1.146 + 1.118 × 10^-4 × b_device × s seconds
```

### Compute-Bound Condition:

Model is compute-bound when:
```
T_compute > T_comm
0.955 × b_device × s > 1.146 + 1.118 × 10^-4 × b_device × s
(0.955 - 1.118 × 10^-4) × b_device × s > 1.146
0.9549 × b_device × s > 1.146
b_device × s > 1.200
```

**Assuming sequence length s = 2048:**
```
b_device > 1.200 / 2048 = 0.000586
b_device ≥ 1 (minimum per-device batch size)
```

**At b_device = 1, s = 2048:**
```
T_compute = 0.955 × 1 × 2048 = 1,955.8 seconds
T_comm = 1.146 + 1.118 × 10^-4 × 1 × 2048 = 1.146 + 0.229 = 1.375 seconds
```

The model is compute-bound with b_device = 1.

**Overall batch size:**
```
Total_batch = b_device × X (FSDP/data parallel dimension)
            = 1 × 16
            = 16
```

### Answer:
**With sequence length 2048, the model becomes compute-bound at per-device batch size of 1 (satisfying 0.955 × b × s > 1.146 seconds); the overall batch size in this setting is 16 (per-device batch of 1 × 16 FSDP devices), where computation time (1,956 sec) significantly exceeds communication time (1.38 sec).**

---

## Part (d): Techniques to Reduce Batch Size While Retaining Throughput (2 pts)

### Question:
In practice, we want the overall batch size to be as small as possible, and we also always use our compute effectively (in other words we want to never be communication bound). What other tricks can we employ to reduce the batch size of our model but retain high throughput?

### Answer:

Several advanced techniques allow reducing batch size while maintaining high compute utilization and avoiding communication bottlenecks:

**1. Gradient Accumulation**: Split the desired effective batch size across multiple micro-batches, accumulating gradients locally before communication. This reduces per-step batch size while maintaining the same optimization trajectory. For example, with gradient accumulation factor of 4, we can achieve effective batch size 16 with per-device micro-batch size 1/4, reducing activation memory by 4× while keeping the same number of optimizer steps. The trade-off is 4× more forward/backward passes, but since we're compute-bound, this utilizes otherwise idle compute cycles during communication.

**2. Activation Checkpointing (Recomputation)**: Rather than storing all intermediate activations for backward pass, checkpoint only selected layers and recompute others during backward. This reduces activation memory quadratically with checkpoint frequency (e.g., checkpointing every k layers reduces memory by factor of k) at the cost of ~33% additional computation for recomputation (Griewank & Walther, 2000; Chen et al., 2016). When compute-bound, this extra computation overlaps with communication, making it nearly "free" while enabling smaller batch sizes to fit in memory.

**3. Selective Precision for Activations**: Use mixed precision more aggressively by storing activations in lower precision (INT8 or FP8) during forward pass and upcasting only during backward. Recent work (Micikevicius et al., 2022) shows FP8 can reduce activation memory by 2× compared to BF16 with minimal accuracy impact. Combined with activation checkpointing, this enables 4-8× reduction in activation memory.

**4. Pipeline Parallelism with Micro-batching**: Partition model layers across pipeline stages and process multiple micro-batches concurrently (Huang et al., 2019; Narayanan et al., 2021). This reduces per-device batch size to micro-batch size while pipeline bubbles fill compute gaps. For example, with 4 pipeline stages and 8 micro-batches, we achieve 8× smaller per-device batch with only ~12.5% pipeline bubble overhead, and the bubble overhead becomes negligible when number of micro-batches >> number of stages.

**5. Sequence Parallelism**: For transformer models, shard the sequence dimension across devices in addition to tensor/data parallelism (Korthikanti et al., 2023). This distributes activation memory across devices without increasing communication volume significantly, since activations in non-attention layers are already sequence-independent. Sequence parallelism can reduce per-device activation memory by the sequence-parallel degree with minimal overhead.

**References:**
- Griewank & Walther (2000): "Algorithm 799: Revolve"
- Chen et al. (2016): "Training Deep Nets with Sublinear Memory Cost"
- Huang et al. (2019): "GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism"
- Narayanan et al. (2021): "Efficient Large-Scale Language Model Training with Pipeline Parallelism"
- Micikevicius et al. (2022): "FP8 Formats for Deep Learning"
- Korthikanti et al. (2023): "Reducing Activation Recomputation in Large Transformer Models"

---

## Summary

| Part | Key Result |
|------|------------|
| (a) | 3,520 GB total memory (44 H100s); 440 GB saved with BF16 backward |
| (b) | Memory = `(3,520 + A/2) / N_FSDP + A/2` GB; need N_FSDP ≥ 156 for typical batch |
| (c) | Compute-bound at b_device = 1 (s=2048); overall batch = 16 |
| (d) | Gradient accumulation, activation checkpointing, mixed precision, pipeline parallelism, sequence parallelism |

