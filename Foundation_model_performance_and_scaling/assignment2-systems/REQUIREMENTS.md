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
