# CS336 Assignment 2 (systems): Systems and Parallelism Requirements
- **Version**: 1.0.4
- **Term**: Spring 2025

## 1 Assignment Overview
- Implement and validate:
  1. Benchmarking and profiling harness.
  2. Flash Attention 2 Triton kernel.
  3. Distributed data parallel training.
  4. Optimizer state sharding.
- Clone `github.com/stanford-cs336/assignment2-systems` to obtain starter code and writeup; pull updates as needed.
- Key repository areas:
  - `cs336-basics/`: staff solution from Assignment 1; adjust `pyproject.toml` to use custom model if desired.
  - `/`: empty `cs336_systems` module to extend freely.
  - `tests/*.py`: official tests invoking hooks in `tests/adapters.py`; ensure compatibility while optionally adding private tests.
  - `README.md`: directory layout and environment setup guidance.
- Submission package:
  - `writeup.pdf` with written answers (typeset).
  - `code.zip` from `test_and_make_submission.sh` containing implemented code.
- Assignment narrative: profile Transformer performance on a single GPU, optimize self-attention kernels, and extend training across multiple GPUs.

### 1.1 Profiling and Benchmarking
#### 1.1.1 Setup - Importing your Basics Transformer Model
- Use `uv run` to install local `cs336-basics` and `cs336-systems` packages; verify imports (e.g., `import cs336_basics.model`).
- Optionally re-point `pyproject.toml` to a custom Assignment 1 implementation.

#### 1.1.2 Model Sizing
- Benchmark configurations (vocab size 10,000; batch size 4; varying context length):

| Size | d_model | d_ff | num_layers | num_heads |
| --- | --- | --- | --- | --- |
| small | 768 | 3072 | 12 | 12 |
| medium | 1024 | 4096 | 24 | 16 |
| large | 1280 | 5120 | 36 | 20 |
| xl | 1600 | 6400 | 48 | 25 |
| 2.7B | 2560 | 10240 | 32 | 32 |

- Automate table creation for the writeup (e.g., `pandas.DataFrame.to_markdown()` or custom scripts).

#### 1.1.3 End-to-End Benchmarking (benchmarking_script, 4 pts)
- Build a script that:
  - Instantiates a basics Transformer using supplied hyperparameters.
  - Synthesizes random data batches.
  - Executes `w` warm-up iterations before timing `n` measured steps (forward-only or forward+backward selectable via CLI arguments).
  - Calls `torch.cuda.synchronize()` after each step and uses `timeit.default_timer()` or equivalent high-resolution timer.
- Deliverable (a): submit the benchmarking script as described above.
- Deliverable (b): record average and standard deviation timings (5 warm-up steps, 10 measured steps) for forward and backward passes across all model sizes in §1.1.2; summarize findings in 1–2 sentences.
- Deliverable (c): repeat measurements with 0, 1, and 2 warm-up steps; report impacts and explain causes in 2–3 sentences.

#### 1.1.4 Nsight Systems Profiler (nsys_profile, 5 pts)
- Profile forward, backward, and optimizer steps with `nsys profile` for each model size (Table 1) and context lengths {128, 256, 512, 1024}; document any OOM scenarios.
- Use NVTX ranges (manual annotations and/or `--pytorch`) to isolate warm-ups, attention sub-steps, and pass types.
- Deliverable (a): confirm total forward-pass runtime and compare with Python benchmarking (1–2 sentences).
- Deliverable (b): identify the CUDA kernel with highest cumulative GPU time during forward pass, count invocations per pass, and compare against full training (forward+backward) kernel dominance (1–2 sentences).
- Deliverable (c): note non-matmul CUDA kernels that contribute meaningful runtime (1–2 sentences).
- Deliverable (d): profile full training step with AdamW; contrast matrix-multiplication share of runtime versus inference-only, and note shifts for other kernels (1–2 sentences).
- Deliverable (e): compare softmax versus matrix multiplication runtimes within self-attention and relate differences to their FLOP counts (1–2 sentences).

#### 1.1.5 Mixed Precision (mixed_precision_accumulation, benchmarking_mixed_precision; 3 pts)
- Context: FP16/BF16 Tensor Cores accelerate matmuls versus FP32 but require careful handling to avoid underflow/overflow; use `torch.autocast(device="cuda", dtype=...)` to apply mixed precision selectively while retaining FP32 accumulations where needed.
- Autocast usage: enclose forward computations in the autocast context; prefer keeping accumulations in FP32 even when operands are downcast.

- Deliverable (mixed_precision_accumulation, 1 pt):
  - Run the provided accumulation loops comparing FP32 and FP16 accumulation strategies.
  - Explain observed accuracy differences across the four accumulation variants in 2–3 sentences.

- Deliverable (benchmarking_mixed_precision, 2 pts):
  - (a) For the `ToyModel` defined in the prompt, assume FP32 parameters and FP16 autocast; report data types observed for: model parameters inside autocast, `fc1` output, `ln` output, predicted logits, loss, and gradients.
  - (b) Describe (2–3 sentences) which layer normalization components are sensitive to mixed precision, and whether BF16 requires special handling compared to FP16, with justification.
  - (c) Extend the benchmarking script with an optional BF16 mixed-precision mode (consider `contextlib.nullcontext` for no-op contexts). Benchmark forward/backward passes with and without mixed precision for each model size in §1.1.2, then summarize timings and trends over 2–3 sentences.

#### 1.1.6 Memory Profiling (memory_profiling; 4 pts)
- Extend the profiling script with a flag that wraps runs in `torch.cuda.memory._record_memory_history(max_entries=1_000_000)`, dumps to `memory_snapshot.pickle` via `_dump_snapshot`, and stops recording with `_record_memory_history(enabled=None)` after either a forward-only run or a complete training step (forward+backward+optimizer). Reuse existing options for model selection, context length, and mixed precision so the 2.7B model at context lengths 128/256/512 can be profiled consistently. Analyze snapshots using https://pytorch.org/memory_viz.
- Deliverable (a): capture two “Active memory timeline” screenshots for the 2.7B model—one forward-only, one full training step—and add a 2–3 sentence explanation of how timeline peaks align with the execution stages.
- Deliverable (b): tabulate the peak GPU memory for each context length {128, 256, 512}, reporting separate values for forward-only and full training runs.
- Deliverable (c): repeat peak-memory measurements with mixed precision enabled for the 2.7B model (forward and full training); provide a 2–3 sentence discussion of the observed memory deltas.
- Deliverable (d): compute the single-precision size (in MB, using 1024² bytes) of one Transformer residual-stream activation tensor under the reference hyperparameters, and show the 1–2 sentence derivation and result.
- Deliverable (e): inspect the forward-pass snapshot at reduced “Detail” levels to identify the largest allocations, report their sizes, and name the originating code path per the stack trace in 1–2 sentences.

### 1.2 Optimizing Attention with FlashAttention-2

#### 1.2.1 Benchmarking PyTorch Attention (pytorch_attention; 2 pts)
- Implement a standalone benchmarking script for a single-head attention module (batch size fixed to 8, no explicit head dimension). For each combination of head dimension `d_model ∈ {16, 32, 64, 128}` and sequence length `seq_len ∈ {256, 1024, 4096, 8192, 16384}`, allocate random `Q/K/V` tensors, run warm-ups, then measure 100 forward passes and 100 backward passes, calling `torch.cuda.synchronize()` after each iteration. Record timing per configuration and capture peak memory in use right before backward begins (e.g., via `torch.cuda.max_memory_allocated()` or equivalent). Handle out-of-memory cases and note them explicitly.
- For the smallest configuration that OOMs, compute the theoretical memory footprint of naïve attention (scores + softmax intermediates + gradients) using the Assignment 1 formulas, and explain how much memory is reclaimed during backward when sequence length changes; discuss mitigation strategies to remove this `seq_len²` storage cost (e.g., tiled/streaming attention such as FlashAttention-2).
- Deliverable: a table covering every `(d_model, seq_len)` pair with forward/backward timings or “OOM”, the detailed memory accounting for the selected failing case, and a 1–2 paragraph write-up answering the OOM threshold, backward-memory scaling with `seq_len`, and the proposed approach to eliminate the memory overhead.

### 1.3 Benchmarking JIT-Compiled Attention

Since PyTorch 2.0, the framework includes a JIT compiler (`torch.compile`) that automatically optimizes PyTorch functions by analyzing computation graphs and generating fused Triton kernels. Usage is simple: `compiled_layer = torch.compile(layer)`. The compiled layer behaves identically to the original (forward/backward) but with potential performance improvements.

#### 1.3.1 torch.compile Experiments (torch_compile; 2 pts)
- Augment the attention benchmarking script from §1.2.1 with an option to wrap the attention module in `torch.compile(...)`. For every `(d_model, seq_len)` configuration in that grid, rerun the 100 forward and 100 backward pass measurements for both the vanilla and compiled versions (same warm-ups and `torch.cuda.synchronize()` calls). Record timing pairs side-by-side to show the compiler’s impact.
- Integrate `torch.compile` into the end-to-end Transformer benchmarking script (the one used for §1.1.3) so entire models can run either vanilla or compiled. Measure average runtimes for forward-only and full training steps (forward+backward+optimizer) on the same model/context configurations you benchmarked previously, keeping warm-up counts identical. Compare compiled versus uncompiled timings in a concise table.
- Deliverables: (a) table comparing forward/backward timings for compiled vs. baseline attention across the §1.2.1 configuration grid; (b) table comparing forward-only and full training runtimes of the vanilla versus compiled Transformer; accompany tables with brief text noting observed performance changes.

Despite torch.compile optimizations, current implementations suffer from poor memory access patterns at long sequence lengths due to the quadratic attention matrix. This motivates implementing FlashAttention-2 in Triton for explicit control over memory access patterns and computation scheduling.

#### 1.3.2 FlashAttention-2 Forward Pass (flash_forward; 15 pts)

Implement FlashAttention-2 [Dao, 2023] in Triton to replace PyTorch attention with efficient tiled computation that avoids materializing the full attention matrix in global memory. See `FLASHATTENTION2_BACKGROUND.md` for algorithm details, mathematical background, and implementation guidance.

**Deliverables**:

(a) **(5 pts)** Pure PyTorch (no Triton) `autograd.Function` implementing FlashAttention-2 forward pass. Takes Q, K, V and `is_causal` flag; produces O and logsumexp L. Ignore `is_causal` for now. Interface: `def forward(ctx, Q, K, V, is_causal=False)`. Save L, Q, K, V, O for backward; return O. Implement `backward` to raise `NotImplementedError`. Use tile sizes ≥16×16; assume dimensions are clean powers of 2 and ≥16 (no bounds checking needed). Compare against Equations 4-6 and 12.
   - Implement `adapters.get_flashattention_autograd_function_pytorch`
   - Test: `uv run pytest -k test_flash_forward_pass_pytorch`

(b) **(8 pts)** Triton kernel for FlashAttention-2 forward pass following Algorithm 1, wrapped in `torch.autograd.Function`.
   - Launch grid: `(T_q, batch_size)`—each program instance handles one batch index and one query tile
   - Single loop iterating key tiles 1≤j≤T_k; advance block pointers at loop end
   - Function declaration:
     ```python
     @triton.jit
     def flash_fwd_kernel(
         Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
         stride_qb, stride_qq, stride_qd,
         stride_kb, stride_kk, stride_kd,
         stride_vb, stride_vk, stride_vd,
         stride_ob, stride_oq, stride_od,
         stride_lb, stride_lq,
         N_QUERIES, N_KEYS, scale,
         D: tl.constexpr,
         Q_TILE_SIZE: tl.constexpr,
         K_TILE_SIZE: tl.constexpr,
     ):
         query_tile_index = tl.program_id(0)
         batch_index = tl.program_id(1)
         Q_block_ptr = tl.make_block_ptr(
             Q_ptr + batch_index * stride_qb,
             shape=(N_QUERIES, D),
             strides=(stride_qq, stride_qd),
             offsets=(query_tile_index * Q_TILE_SIZE, 0),
             block_shape=(Q_TILE_SIZE, D),
             order=(1, 0),
         )
         ...
     ```
     where `scale = 1/√d`, `Q_TILE_SIZE = B_q`, `K_TILE_SIZE = B_k` (tunable).
   - Precision guidelines:
     - On-chip buffers (O_i, l, m): dtype `tl.float32`; accumulate with `acc` argument: `tl.dot(..., acc=acc)`
     - Cast `P̃_i^(j)` to dtype of `V^(j)` before multiply; cast O_i before writing to global memory
     - Use `tensor.to`, `tensor.dtype`, `*_block_ptr.type.element_ty`
   - Debug: compare each Triton operation result against part (a) PyTorch tiled implementation
   - Implement `adapters.get_flash_autograd_function_triton`
   - Test: `uv run pytest -k test_flash_forward_pass_triton`

(c) **(2 pts)** Add `is_causal` boolean flag (last argument) for causal masking. In Triton kernel, add parameter `is_causal: tl.constexpr`. Construct query/key index vectors and compare to form B_q×B_k mask; add `-1e6` to masked elements of `S_i^(j)`. Save via `ctx.is_causal = is_causal`. Default `False` to preserve previous tests.

**Implementing the backward pass with recomputation**: Unlike the standard backward pass in Equations 7-11, recomputation can avoid the softmax operation in the backward pass shown in Equations 13-19. This means the backward pass can be computed using a trivial kernel without online tricks. For this part, implement backward by calling `torch.compile` on a regular PyTorch function (not Triton).

(d) **(5 pts) flash_backward** - Implement the backward pass for FlashAttention-2 `autograd.Function` using PyTorch (not Triton) and `torch.compile`.
   - Implementation should take Q, K, V, O, dO, and L tensors as input and return dQ, dK, and dV
   - Remember to compute and use the D vector
   - Follow computations in Equations 13-19
   - Implement `adapters.get_flash_autograd_function_triton` (if not already done in part b)
   - Test: `uv run pytest -k test_flash_backward`

(e) **(5 pts) flash_benchmarking** - Compare performance of (partially) Triton FlashAttention-2 implementation with regular PyTorch attention.
   - Write benchmarking script using `triton.testing.do_bench`
   - Compare forward, backward, and end-to-end forward-backward passes
   - Report table with latencies for both Triton and PyTorch implementations
   - Benchmark settings:
     - Single H100 GPU
     - Batch size: 1
     - Causal masking: enabled
     - Sequence lengths: powers of 2 from 128 to 65536
     - Embedding dimensions: powers of 2 from 16 to 128
     - Precisions: `torch.bfloat16` and `torch.float32`
     - Randomly generate inputs before benchmarking
     - Adjust tile sizes depending on input sizes as needed
   - Deliverable: Table of results comparing FlashAttention-2 vs PyTorch attention, reporting forward, backward, and end-to-end latencies for all parameter combinations

#### 1.3.3 FlashAttention-2 Leaderboard (Optional Challenge)

Assignment 2's leaderboard tests the speed of FlashAttention-2 implementation (forward and backward passes). Challenge: further improve performance using any optimization tricks while maintaining correctness.

**Restrictions**:
- Cannot change input/outputs of the function
- Must use Triton (no CUDA)
- Must pass same tests as regular implementation
- Implementation must be your own (no pre-existing implementations)

**Test Configuration**:
- Timing measured on H100
- Batch size: 1
- Sequence length: 16,384
- d_model: 1024 (16 heads × 64 d_head)
- Precision: BF16 with causal masking

**Test Code**:
```python
def test_timing_flash_forward_backward():
    n_heads = 16
    d_head = 64
    sequence_length = 16384
    q, k, v = torch.randn(
        3, n_heads, sequence_length, d_head,
        device='cuda', dtype=torch.bfloat16, requires_grad=True
    )

    flash = torch.compile(FlashAttention2.apply)

    def flash_forward_backward():
        o = flash(q, k, v, True)
        loss = o.sum()
        loss.backward()

    results = triton.testing.do_bench(flash_forward_backward, rep=10000, warmup=1000)
    print(results)
```

**Optimization Ideas**:
- Tune tile sizes using Triton autotune
- Tune additional Triton config parameters
- Implement backward pass in Triton (see §1.3.4)
- Use two passes for backward: one for dQ, another for dK/dV (avoids atomics/synchronization)
- Stop program instances early with causal masking (skip all-zero tiles)
- Separate non-masked tiles from diagonal tiles (compute first without index comparisons)
- Use TMA (Tensor Memory Accelerator) on H100 following [Triton TMA tutorial](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html)

**Submission**: Submit best times to [github.com/stanford-cs336/assignment2-systems-leaderboard](https://github.com/stanford-cs336/assignment2-systems-leaderboard)

**Verification**: Top 5-10 submissions verified for correctness and performance

#### 1.3.4 OPTIONAL: Triton Backward Pass (flash_backward_triton)

Implement tiled FlashAttention-2 backward pass in Triton for better performance and leaderboard submission.

**Key Optimization**: Compute P twice (once for dQ, once for dK/dV) to skip synchronization across thread blocks.

**Algorithm 2: Tiled FlashAttention-2 Backward Pass**

```
Input: Q, O, dO ∈ R^(N_q×d), K, V ∈ R^(N_k×d), L ∈ R^(N_q), tile sizes B_q, B_k

Compute D = rowsum(dO ◦ O) ∈ R^(N_q)
Split Q, O, dO into T_q = ⌈N_q/B_q⌉ tiles Q_1,...,Q_Tq, O_1,...,O_Tq, dO_1,...,dO_Tq (size B_q×d)
Split K, V into T_k = ⌈N_k/B_k⌉ tiles K^(1),...,K^(Tk), V^(1),...,V^(Tk) (size B_k×d)
Split L, D into T_q tiles L_1,...,L_Tq, D_1,...,D_Tq (size B_q)

for j = 1,...,T_k do                          ← Outer loop: key tiles
  Load K^(j), V^(j) from global memory
  Initialize dK^(j) = dV^(j) = 0 ∈ R^(B_k×d)

  for i = 1,...,T_q do                        ← Inner loop: query tiles
    Load Q_i, O_i, dO_i, dQ_i from global memory

    Compute S_i^(j) = Q_i(K^(j))⊤/√d ∈ R^(B_q×B_k)
    Compute P_i^(j) = exp(S_i^(j) - L_i) ∈ R^(B_q×B_k)

    Compute dV^(j) += (P_i^(j))⊤ @ dO_i ∈ R^(B_k×d)
    Compute dP_i^(j) = dO_i @ V_j⊤ ∈ R^(B_q×B_k)
    Compute dS_i^(j) = P_i^(j) ◦ (dP_i^(j) - D_i) / √d ∈ R^(B_q×B_k)

    Load dQ_i from global memory
    Update dQ_i += dS_i^(j) @ K^(j) ∈ R^(B_q×d)
    Write dQ_i to global memory (MUST BE ATOMIC for correctness!)

    Compute dK^(j) += (dS_i^(j))⊤ @ Q_i ∈ R^(B_k×d)
  end for

  Write dK^(j) and dV^(j) to global memory as j-th tiles of dK and dV
end for

Return dQ, dK, dV
```

**Implementation Notes**:
- Launch grid: `(T_k, batch_size)` - each program handles one key tile for one batch
- Outer loop over key tiles, inner loop over query tiles
- Atomic operations required for dQ updates (multiple key tiles write to same query tile)
- dK and dV can be accumulated locally without atomics
- Use `tl.atomic_add` for dQ updates
- Pre-compute D before kernel launch (can be separate kernel or CPU)

**Deliverable**: Implement Triton backward kernel and integrate into `FlashAttentionTritonFunc.backward()`

## 2 Distributed Data Parallel Training

Focus on using multiple GPUs to train language models through data parallelism. Begin with PyTorch distributed communication primitives, then study and improve distributed data parallel training implementations for better communication efficiency.

### 2.1 Single-Node Distributed Communication in PyTorch

#### 2.1.1 PyTorch Distributed Basics

**Core Concepts**:
- **Process groups**: Multiple worker processes coordinating via shared master (rank 0)
- **Collective operations**: All-reduce, all-gather, etc. operating on all processes in group
- **Backends**:
  - `gloo`: CPU-only machines, local development
  - `nccl`: GPU training, better performance for CUDA tensors
- **Process spawning**: Use `mp.spawn(fn, args, nprocs)` where `fn(rank, *args)` receives rank as first argument

**Key setup patterns**:
```python
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # For multi-GPU: torch.cuda.set_device(rank)
    # Or use device = f"cuda:{rank}" explicitly
```

**Terminology**:
- **node**: A machine on the network
- **worker/process**: Instance of program participating in distributed training (used interchangeably for single-process workers)
- **world size**: Total number of workers in process group
- **global rank**: Integer ID (0 to world_size-1) uniquely identifying a worker
- **local world size**: Number of workers on a given node (equals world size for single-node)
- **local rank**: Integer ID (0 to local_world_size-1) for worker on local machine (equals global rank for single-node)

#### 2.1.2 Best Practices for Benchmarking Distributed Applications

- Run benchmarks on same machine for controlled comparisons
- Perform warm-up steps before timing (5 iterations generally sufficient, especially for NCCL)
- Call `torch.cuda.synchronize()` when benchmarking GPUs (necessary even with `async_op=False`, which only queues operations)
- Aggregate measurements across ranks (use `dist.all_gather_object` for collecting results)
- Debug locally with Gloo on CPU, benchmark with NCCL on GPU (switch via `init_process_group` and device casts)

#### 2.1.3 Benchmarking All-Reduce (distributed_communication_single_node; 5 pts)

Write a script to benchmark all-reduce operation runtime in single-node multi-process setup.

**Experiment variations**:
- **Backend + device**: Gloo + CPU, NCCL + GPU
- **Data size**: float32 tensors of 1MB, 10MB, 100MB, 1GB
- **Number of processes**: 2, 4, or 6 processes

**Resource requirements**:
- Up to 6 GPUs
- Each benchmarking run <5 minutes

**Deliverable**:
- Plot(s) and/or table(s) comparing the various settings
- 2-3 sentences of commentary about results and how the various factors (backend, data size, process count) interact

### 2.2 A Naïve Implementation of Distributed Data Parallel Training

Build a minimal implementation of vanilla distributed data parallel (DDP) training using collective communication operations.

#### 2.2.1 Data Parallelism Overview

**Concept**: Split batches across multiple devices (e.g., GPUs) to enable training on large batch sizes that don't fit on a single device.

**Example**: 4 devices × 32 max batch size per device → 128 effective batch size

**Vanilla DDP Algorithm**:

1. **Initialization**:
   - Each device constructs a randomly-initialized model
   - Use broadcast collective to send model parameters from rank 0 to all other ranks
   - Each device holds identical copy of model parameters and optimizer states

2. **Training iteration**:
   - **Step 1**: Given batch with `n` examples, shard across `d` devices → each device gets `n/d` disjoint examples
     - Note: `n` should divide `d` (training bottlenecked by slowest process)

   - **Step 2**: Each device runs:
     - Forward pass on its `n/d` examples using local model copy
     - Backward pass to calculate gradients
     - Result: Each device holds gradients from its `n/d` examples

   - **Step 3**: Use all-reduce collective to average gradients across devices
     - Result: Each device holds gradients averaged across all `n` examples

   - **Step 4**: Each device runs optimizer step to update local parameters
     - Parameters and optimizer states stay synchronized because:
       - All start from same initialization
       - All use same averaged gradients each iteration

   - **Repeat**: Iteration complete, continue to next batch

**Key properties of vanilla DDP**:
- Full model replication: Each device holds complete copy of parameters
- Full optimizer replication: Each device holds complete optimizer states
- Only gradients communicated: All-reduce after backward pass
- Memory inefficient: No sharding of parameters or optimizer states

#### 2.2.2 Naïve DDP Implementation (naive_ddp; 5 pts)

Write a script to perform distributed data parallel training by all-reducing individual parameter gradients after the backward pass.

**Implementation requirements**:
- Naïve approach: All-reduce each parameter's gradient individually
- Use collective communication operations (all-reduce)
- Synchronize parameters across all devices after gradient averaging

**Testing for correctness**:
- Train a small toy model on randomly-generated data
- Compare with single-process training
- Verify that DDP weights match single-process weights
- See `tests/test_ddp_individual_parameters.py` for reference

**Deliverable**:
- Script implementing naïve DDP training
- Correctness verification showing DDP matches single-process training

#### 2.2.3 Naïve DDP Benchmarking (naive_ddp_benchmarking; 3 pts)

Benchmark the naïve DDP implementation to understand overhead of data parallel training.

**Benchmarking setup**:
- Model: XL model size from §1.1.2 (d_model=1600, d_ff=6400, num_layers=48, num_heads=25)
- Configuration: Single-node, 2 GPUs (1 node × 2 GPUs)
- Naive implementation: Individual parameter gradients all-reduced after backward pass

**Measurements required**:
1. Total time per training step
2. Proportion of time spent on communicating gradients
3. Breakdown showing communication overhead

**Analysis**:
- Identify communication bottleneck in naïve approach
- Understand why individual parameter all-reduces are inefficient
- Motivate need for optimized DDP implementations

**Deliverable**:
- Description of benchmarking setup
- Measured time per training iteration for each setting
- Measured time spent communicating gradients
- Analysis of communication overhead (1-2 paragraphs)

### 2.3 Improving Upon the Minimal DDP Implementation

The minimal DDP implementation from §2.2 has two key limitations:

1. **Multiple communication calls**: Conducts a separate all-reduce operation for every parameter tensor. Each communication call incurs overhead, so batching communication calls may minimize this overhead.

2. **No overlap of computation and communication**: Waits for the backward pass to finish before communicating gradients. However, the backward pass is incrementally computed—when a parameter gradient is ready, it can immediately be communicated without waiting for other parameters. This allows overlapping communication of gradients with computation of the backward pass, reducing distributed training overhead.

This section addresses each limitation and measures the impact on training speed.

#### 2.3.1 Reducing the Number of Communication Calls (minimal_ddp_flat_benchmarking; 2 pts)

Rather than issuing a communication call for each parameter tensor, improve performance by batching the all-reduce.

**Implementation approach**:
- Take all gradients to all-reduce
- Concatenate them into a single tensor
- All-reduce the combined gradients across all ranks
- Helper utilities: `torch._utils._flatten_dense_tensors` and `torch._utils._unflatten_dense_tensors`

**Benchmarking setup**:
- Model: XL model size from §1.1.2
- Configuration: Single-node, 2 GPUs (1 node × 2 GPUs)
- Compare: Batched all-reduce vs. individual parameter all-reduces

**Deliverable**:
- Measured time per training iteration with single batched all-reduce call
- Measured time spent communicating gradients
- 1-2 sentences comparing results when batching vs. individually communicating gradients

#### 2.3.2 Overlapping Computation with Communication of Individual Parameter Gradients

Overlap backward pass computation with gradient communication by all-reducing parameter gradients as soon as they're ready, rather than waiting for the entire backward pass to complete.

**Implementation approach**:

1. **Backward hooks**: Use `register_post_accumulate_grad_hook()` to automatically call a function on a parameter after its gradient has been accumulated in the backward pass
   - See: `pytorch.org/docs/stable/generated/torch.Tensor.register_post_accumulate_grad_hook.html`

2. **Asynchronous communication**: PyTorch collective operations support both synchronous and asynchronous execution:
   - `async_op=False` (default): Blocks until operation is queued on GPU
   - `async_op=True`: Returns immediately with a distributed request handle
   - Call `handle.wait()` on returned handle to wait for operation to be queued on GPU

**Example synchronous vs. asynchronous all-reduce**:
```python
tensors = [torch.rand(5) for _ in range(10)]

# Synchronous - block until operation is queued on GPU
for tensor in tensors:
    dist.all_reduce(tensor, async_op=False)

# Asynchronous - return immediately after each call
handles = []
for tensor in tensors:
    handle = dist.all_reduce(tensor, async_op=True)
    handles.append(handle)

# ... possibly execute other commands that don't rely on all_reduce results ...

# Ensure all-reduce calls were queued
for handle in handles:
    handle.wait()
handles.clear()
```

**Problem (ddp_overlap_individual_parameters; 5 pts)**:

Implement a Python class to handle distributed data parallel training that overlaps backward computation with gradient communication.

**Required interface**:
```python
class DDP:
    def __init__(self, module: torch.nn.Module):
        """Construct DDP container to handle gradient synchronization across ranks.

        Args:
            module: PyTorch nn.Module to be parallelized
        """
        pass

    def forward(self, *inputs, **kwargs):
        """Call wrapped module's forward() with provided arguments."""
        pass

    def finish_gradient_synchronization(self):
        """Wait for asynchronous communication calls to be queued on GPU."""
        pass
```

**Usage pattern**:
```python
model = ToyModel().to(device)
ddp_model = DDP(model)

for _ in range(train_steps):
    x, y = get_batch()
    logits = ddp_model(x)
    loss = loss_fn(logits, y)
    loss.backward()
    ddp_model.finish_gradient_synchronization()  # Wait before optimizer.step()
    optimizer.step()
```

**Deliverable**:
- DDP container class implementation overlapping gradient communication with backward computation
- Implement adapters:
  - `adapters.get_ddp_individual_parameters`
  - `adapters.ddp_individual_parameters_on_after_backward` (optional, may not be needed)
- Test: `uv run pytest tests/test_ddp_individual_parameters.py`
- Run tests multiple times (e.g., 5) to ensure reliable passing

**Problem (ddp_overlap_individual_parameters_benchmarking; 1 pt)**:

(a) **Performance benchmarking**: Benchmark the DDP implementation overlapping backward pass with communication of individual parameter gradients.

**Comparison settings**:
- Setup: 1 node, 2 GPUs, XL model size (§1.1.2)
- Compare against:
  - Minimal DDP with individual parameter all-reduces
  - Minimal DDP with single batched all-reduce

**Deliverable**:
- Measured time per training iteration when overlapping backward with communication
- 1-2 sentences comparing results

(b) **Profiler visualization**: Use Nsight profiler to visually demonstrate compute/communication overlap.

**Setup**:
- Configuration: 1 node, 2 GPUs, XL model size
- Compare:
  - Initial DDP implementation (no overlap)
  - This DDP implementation (with overlap)

**Deliverable**:
- 2 screenshots from profiler:
  1. Initial DDP implementation showing no overlap
  2. This DDP implementation showing compute/communication overlap
- Visual demonstration that one implementation overlaps compute with communication while the other doesn't

#### 2.3.3 Overlapping Computation with Communication of Bucketed Parameter Gradients

Combine the benefits of batching (§2.3.1) and overlapping (§2.3.2) by organizing parameters into buckets and all-reducing buckets when each of their constituent tensors are ready.

**Motivation**:
- Batching communication reduces overhead of many small all-reduces
- But batching all parameters requires waiting for entire backward pass
- **Solution**: Bucket parameters and all-reduce buckets as they become ready

**Problem (ddp_overlap_bucketed; 8 pts)**:

Implement a Python class to handle distributed data parallel training using gradient bucketing to improve communication efficiency.

**Required interface**:
```python
class DDP:
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        """Construct DDP container with gradient bucketing.

        Args:
            module: PyTorch nn.Module to be parallelized
            bucket_size_mb: Maximum size (MB) of parameters per bucket
        """
        pass

    def forward(self, *inputs, **kwargs):
        """Call wrapped module's forward() with provided arguments."""
        pass

    def finish_gradient_synchronization(self):
        """Wait for asynchronous communication calls to be queued on GPU."""
        pass
```

**Implementation guidance**:
- Allocate parameters to buckets using **reverse order** of `model.parameters()`
  - Gradients become ready in approximately reverse order during backward pass
- Each bucket holds at most `bucket_size_mb` MB of parameters
- All-reduce each bucket when all its constituent parameter gradients are ready

**Deliverable**:
- DDP container class with gradient bucketing that overlaps communication with backward computation
- Implement adapters:
  - `adapters.get_ddp_bucketed`
  - `adapters.ddp_bucketed_on_after_backward` (optional)
  - `adapters.ddp_bucketed_on_train_batch_start` (optional)
- Test: `uv run pytest tests/test_ddp.py`
- Run tests multiple times (e.g., 5) to ensure reliable passing

**Problem (ddp_bucketed_benchmarking; 3 pts)**:

(a) **Bucket size impact**: Benchmark bucketed DDP implementation with varying bucket sizes.

**Benchmarking setup**:
- Configuration: 1 node, 2 GPUs, XL model size
- Bucket sizes: 1, 10, 100, 1000 MB
- Compare results to previous experiments without bucketing

**Analysis questions**:
- Do results align with expectations?
- If not aligned, why not?
- What changes in experimental setup would yield results aligned with expectations?
- May need PyTorch profiler to understand communication call ordering/execution

**Deliverable**:
- Measured time per training iteration for various bucket sizes
- 3-4 sentence commentary about:
  - Results
  - Expectations
  - Potential reasons for any mismatch

(b) **Mathematical modeling**: Model DDP communication overhead mathematically.

**Assumption**: Time to compute gradients for a bucket equals time to communicate gradient buckets.

**Task 1**: Write equation modeling communication overhead of DDP as function of:
- `s`: Total size (bytes) of model parameters
- `w`: All-reduce algorithm bandwidth (bytes per second, computed as data size / time to finish all-reduce)
- `o`: Overhead (seconds) associated with each communication call
- `nb`: Number of buckets

**Task 2**: From the overhead equation, derive equation for optimal bucket size that minimizes DDP overhead.

**Deliverable**:
- Equation modeling DDP overhead
- Equation for optimal bucket size
