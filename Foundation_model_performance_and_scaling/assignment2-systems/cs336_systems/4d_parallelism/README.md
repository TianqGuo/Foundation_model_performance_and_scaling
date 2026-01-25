# Section 2.4: 4D Parallelism

Analysis and calculations for multi-dimensional parallelism strategies in large-scale model training.

## Overview

This section explores the memory and communication requirements for training extremely large models (220B parameters) using combinations of:
- **Data Parallelism (DP)**: Splitting batches across devices
- **Fully-Sharded Data Parallelism (FSDP)**: Sharding weights, gradients, and optimizer states
- **Tensor Parallelism (TP)**: Sharding activations across operations
- **Pipeline Parallelism (PP)**: Splitting model layers across devices

We focus on dense models and analyze the **XXL configuration** (220B parameters) to understand scaling requirements.

## Files

- **`COMMUNICATION_ACCOUNTING.md`**: Complete analysis with detailed calculations for all four parts
  - Part (a): Single-device memory requirements
  - Part (b): FSDP sharding analysis
  - Part (c): Compute-bound batch size calculations
  - Part (d): Techniques to reduce batch size

## Key Results

### XXL Model Configuration
- `d_model = 16,384`
- `d_ff = 53,248`
- `num_blocks = 126`
- **Total parameters**: ~220 billion

### Part (a): Memory Requirements
- **Total memory (FP32)**: 3,520 GB
  - Master weights: 880 GB
  - Gradients: 880 GB
  - Optimizer states (AdamW): 1,760 GB
- **Memory saved with BF16 backward**: 440 GB
- **H100 GPUs needed**: 44 × 80GB GPUs

### Part (b): FSDP Sharding
- **Memory per device**: `(3,520 + A/2) / N_FSDP + A/2` GB
  - Where A = activation memory (depends on batch size)
- **For typical training** (batch=4, seq_len=2048, A≈144 GB):
  - Need **N_FSDP ≥ 156 devices** to stay under 95 GB per device
- **Minimal activations** (small batch):
  - Need **N_FSDP ≥ 38 devices**

### Part (c): Compute-Bound Batch Size
Using TPU v5p specifications:
- Communication bandwidth: `W_ici = 1.8 × 10^11` bytes/s
- Compute: `C = 4.6 × 10^14` FLOPS/s
- Mesh: 16 FSDP × 4 TP = 64 devices

**Results** (sequence length 2048):
- **Per-device batch size**: 1 (compute-bound)
- **Overall batch size**: 16
- Computation time: 1,956 sec >> Communication time: 1.38 sec

### Part (d): Optimization Techniques
Five key techniques to reduce batch size while maintaining throughput:
1. **Gradient Accumulation**: 4× memory reduction with micro-batching
2. **Activation Checkpointing**: k× memory reduction with ~33% computation overhead
3. **Selective Precision**: 2× memory reduction with FP8/INT8 activations
4. **Pipeline Parallelism**: Enables micro-batch processing across stages
5. **Sequence Parallelism**: Reduces activation memory by sharding sequence dimension

## References

### TPU Scaling Book (Austin et al., 2025)
- Part 5: Communication and memory cost derivations
- Mesh parallelism strategies
- Bandwidth and compute formulas

### Ultra-Scale Playbook (Nouamane Tazi, 2025)
- Appendix: Pipeline parallelism details
- Advanced optimization techniques

### Key Papers
- Griewank & Walther (2000): "Algorithm 799: Revolve" - Activation checkpointing theory
- Chen et al. (2016): "Training Deep Nets with Sublinear Memory Cost" - Checkpointing implementation
- Huang et al. (2019): "GPipe" - Pipeline parallelism with micro-batching
- Narayanan et al. (2021): "Efficient Large-Scale Language Model Training" - Advanced pipelining
- Micikevicius et al. (2022): "FP8 Formats for Deep Learning" - Low precision training
- Korthikanti et al. (2023): "Reducing Activation Recomputation" - Sequence parallelism

## Usage

This is a theoretical analysis section with no code implementation required. The calculations demonstrate:
- Memory scaling challenges for large models
- Trade-offs between different parallelism strategies
- Techniques to optimize memory and communication

## Next Steps

Future sections may implement:
- FSDP (Fully-Sharded Data Parallelism)
- Tensor Parallelism for large FFN layers
- Pipeline Parallelism with micro-batching
- Combined 3D/4D parallelism strategies

---

**Note**: This analysis assumes simplified FFN-only architecture without attention mechanisms. Real transformers with attention would have additional memory and communication costs.