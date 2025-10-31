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
