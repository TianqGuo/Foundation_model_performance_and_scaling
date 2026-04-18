# Foundation Model Systems — Performance, Profiling & Scaling

End-to-end implementation of a language model training stack, from tokenizer and Transformer architecture through GPU kernel optimization, multi-GPU distributed training, empirical scaling law experiments, and web-scale data filtering.

---

## Tech Stack

**Languages & Frameworks:** Python, PyTorch, Triton

**GPU & Profiling:** CUDA, Nsight Systems (`nsys`), NVTX annotations, PyTorch `memory_snapshot`, HBM-aware kernel design, FLOP/throughput benchmarking, `torch.cuda.synchronize()`

**Kernel Optimization:** FlashAttention-2, online softmax, tiling, activation recomputation, `torch.autograd.Function`, BF16 mixed precision, `torch.compile` / JIT, kernel fusion

**Distributed Training:** DDP (Distributed Data Parallel), gradient bucketing, comm/compute overlap, ZeRO-style optimizer sharding, FSDP (analysis), Tensor Parallelism (analysis), NCCL, Gloo, all-reduce, all-gather, broadcast collectives

**Model Components:** BPE tokenizer, Transformer (RoPE, RMSNorm, SwiGLU, causal masking, weight tying), AdamW, cosine LR schedule with warmup, gradient clipping, temperature scaling, nucleus (top-p) sampling

**Scaling Laws:** IsoFLOPs, Chinchilla methodology, power-law fitting (`scipy.optimize`), compute-optimal model selection

**Data Pipeline:** Common Crawl / WARC/WET processing, Resiliparse (HTML extraction), FastWARC, fastText (language ID, quality classifier, NSFW/toxic detection via Dolma), Gopher quality filters, MinHash + LSH fuzzy deduplication, exact line deduplication, PII masking, OpenWebText, Paloma/C4 benchmark

---

## Overview

| Part | Topic | Key Deliverables |
|---|---|---|
| [Part 1](#part-1-tokenizer--transformer) | Tokenizer + Transformer | Byte-level BPE tokenizer, full Transformer LM, AdamW, training loop, ablations |
| [Part 2](#part-2-systems--gpu-optimization) | Systems & GPU Optimization | FlashAttention-2 (Triton), online softmax, tiling, BF16, `torch.compile`, DDP, ZeRO sharding, Nsight profiling |
| [Part 3](#part-3-scaling-laws) | Scaling Laws | IsoFLOPs fitting, Chinchilla methodology, compute-optimal prediction at 10¹⁹ FLOPs |
| [Part 4](#part-4-data-filtering--pipeline) | Data Filtering & Pipeline | Common Crawl pipeline, fastText classifiers, MinHash+LSH dedup, Gopher filters, Paloma evaluation |

---

## Key Findings

**GPU optimization (Part 2)**
- Combined optimizations (FlashAttention-2 + BF16 + `torch.compile` + DDP) achieved **2× average / 6.35× peak** throughput improvement over the unoptimized FP32 baseline
- BF16 provides 16× higher theoretical throughput than FP32 on H100 (312 vs 19.5 TFLOP/s); profiling confirmed attention and matmul as dominant kernels in the forward pass
- FlashAttention-2 eliminates O(seq_len²) HBM reads by keeping Q/K/V tiles in SRAM — memory usage no longer grows quadratically with sequence length

**Architectural ablations (Part 1, models 17M–124M parameters)**
- Pre-norm converged more stably than post-norm — post-norm showed vanishing gradient symptoms in early training
- RoPE consistently outperformed NoPE (no positional embeddings) on language modeling loss
- SwiGLU outperformed SiLU feed-forward networks, consistent with findings in LLaMA/Qwen literature
- Removing RMSNorm caused training instability / loss divergence

**Scaling laws (Part 3)**
- IsoFLOPs power-law fit on synthetic training data (9 compute budgets, 6×10¹⁸ to 3×10²¹ FLOPs):
  - **N_opt = 1.163 × C^0.469** — model size scales slightly sub-linearly with compute (exponent ~0.5 matches Chinchilla)
  - **D_opt = 0.143 × C^0.531** — tokens scale slightly super-linearly, meaning data-efficiency improves at larger budgets
  - Extrapolated to 10²³ FLOPs: **N_opt ≈ 70B params, D_opt ≈ 238B tokens**
  - Extrapolated to 10²⁴ FLOPs: **N_opt ≈ 206B params, D_opt ≈ 809B tokens**
- Part 3b (live API experiments at 10¹⁹ FLOP target): full experiment design and fitting infrastructure implemented; not executed — requires Stanford cluster API access

**Data filtering (Part 4)**
- Pipeline architecture complete (WARC extraction, language ID, PII masking, quality filtering, MinHash+LSH deduplication); implementation and testing in progress

---

## Part 1: Tokenizer + Transformer

**Goal:** Build a complete language model training stack from scratch using only core PyTorch primitives (no `torch.nn` layers, no `torch.optim` implementations — only `nn.Parameter`, container classes, and the `Optimizer` base class).

### Byte-Level BPE Tokenizer
- Implemented byte-pair encoding (BPE) over UTF-8 bytes, supporting arbitrary Unicode input without out-of-vocabulary tokens
- Trained on **TinyStories** (10K vocab) and **OpenWebText** (32K vocab, 11.1 GB corpus; ~2.5 hrs training time)
- Parallel pre-tokenization with GPT-2 regex pre-tokenizer; vocabulary and merge rules serialized to disk

### Transformer Language Model
- Pre-norm Transformer block: multi-head self-attention with **RoPE** positional embeddings, **RMSNorm**, **SwiGLU** feed-forward, **causal masking**, **weight tying** between embedding and output projection
- Custom `torch.autograd.Function` implementations throughout
- Text generation with **temperature scaling** and **nucleus (top-p) sampling**

### Training Infrastructure
- **AdamW optimizer** with decoupled weight decay, implemented from scratch
- **Cosine annealing LR schedule** with linear warmup, gradient clipping, periodic checkpointing
- Datasets tokenized to `.npy` for fast dataloader access; perplexity evaluated on held-out sets

### Architectural Ablations (models 17M–124M parameters)
| Ablation | Comparison |
|---|---|
| Pre-norm vs Post-norm | Effect on training stability and convergence |
| RoPE vs NoPE | Effect of positional embeddings on language modeling loss |
| SwiGLU vs SiLU | Effect of gating in the feed-forward network |
| RMSNorm vs no normalization | Effect of layer normalization on gradient flow |

---

## Part 2: Systems & GPU Optimization

**Goal:** Profile the training stack to find bottlenecks, then optimize single-GPU throughput and scale to multiple GPUs.

### Profiling

Three complementary profiling paths:

- **End-to-end benchmarking** — automated FLOP counting and throughput measurement (tokens/sec) for forward + backward passes; `torch.cuda.synchronize()` for accurate GPU timing; sweeping model sizes and context lengths via Slurm/`submitit`
- **Nsight Systems** — GPU kernel-level traces (`nsys profile`) with **NVTX annotations** capturing CPU/GPU timelines; traces stored as `.nsys-rep`/`.sqlite` in `results/nsight_profiles/`
- **Memory profiling** — PyTorch `memory_snapshot` capturing peak and allocated memory across 5 model sizes (small → 2.7B) and context lengths 128–512, in both FP32 and BF16; snapshots in `results/memory_profiling/`

Model configurations benchmarked:

| Size | d_model | d_ff | Layers | Heads |
|------|---------|------|--------|-------|
| small | 768 | 3072 | 12 | 12 |
| medium | 1024 | 4096 | 24 | 16 |
| large | 1280 | 5120 | 36 | 20 |
| xl | 1600 | 6400 | 48 | 25 |
| 2.7B | 2560 | 10240 | 32 | 32 |

### FlashAttention-2 (Triton Kernel)

Implemented following Dao 2023, avoiding reads/writes of the O(seq_len²) attention matrix to **HBM**:

- **Online softmax** — compute softmax in tiles without materializing the full attention matrix
- **Tiling** — process Q/K/V in SRAM-resident tiles; block pointer arithmetic in Triton
- **Recomputation (activation checkpointing)** — discard intermediate attention weights; recompute from saved {Q, K, logsumexp} during backward, trading compute for memory
- **Causal masking** — skip all-zero tiles above the diagonal for autoregressive models
- Forward pass: pure Triton kernel; backward pass: `torch.compile` over PyTorch autograd

### Optimizations Implemented

| Technique | Module |
|---|---|
| **FlashAttention-2** — fused attention Triton kernel | `cs336_systems/flash_attention/` |
| **BF16 mixed precision** (stable vs FP16 loss scaling) | `cs336_systems/mixed_precision/` |
| **`torch.compile` / JIT** — auto-fused Triton kernels | `cs336_systems/torch_compile_benchmarking/` |
| **Naive DDP** (all-reduce after full backward) | `cs336_systems/naive_ddp/` |
| **DDP with comm/compute overlap** (per-parameter all-reduce) | `cs336_systems/ddp_overlap_individual/` |
| **DDP with gradient bucketing** (batched all-reduce) | `cs336_systems/ddp_bucketed/` |
| **ZeRO-style optimizer state sharding** (ZeRO stage 1) | `cs336_systems/optimizer_sharding/` |
| **NCCL vs Gloo collective benchmarking** | `cs336_systems/distributed_communication/` |

Combined optimizations achieved **2× average / 6.35× peak** throughput improvement over the unoptimized baseline.

### 4D Parallelism Analysis
Theoretical analysis of combining Data Parallelism, **FSDP** (Fully Sharded Data Parallelism), and **Tensor Parallelism** — memory and compute trade-offs at scale.

---

## Part 3: Scaling Laws

**Goal:** Fit empirical scaling laws to predict compute-optimal model and dataset size as a function of FLOPs budget.

### Part 3a — IsoFLOPs Fitting (Fully Implemented & Executed)
- Reproduced the IsoFLOPs scaling law methodology from Hoffmann et al. (**Chinchilla**, 2022) on synthetic training run data (9 compute budgets from 6×10¹⁸ to 3×10²¹ FLOPs)
- For each compute budget C, identified the model size N_opt(C) that minimized final training loss; computed D_opt = C / (6N)
- Fitted power laws in log-log space via `np.polyfit`:
  - **N_opt = 1.163 × C^0.469** (R² fit on 9 data points; exponent ≈ 0.5 consistent with Chinchilla)
  - **D_opt = 0.143 × C^0.531**
- Extrapolations:

| Target Budget | Optimal Model Size | Optimal Dataset Size |
|---|---|---|
| 10²³ FLOPs | ~70B parameters | ~238B tokens |
| 10²⁴ FLOPs | ~206B parameters | ~809B tokens |

Scaling law plots: [`results/part1_isoflops/model_size_scaling_law.png`](./assignment3-scaling/results/part1_isoflops/model_size_scaling_law.png), [`dataset_size_scaling_law.png`](./assignment3-scaling/results/part1_isoflops/dataset_size_scaling_law.png)

### Part 3b — Live Experiment Infrastructure (Implemented, Not Executed)
Full pipeline implemented in `part2_scaling_laws/`:
- **`experiment_design.py`** — IsoFLOPs strategy: 8 compute budgets (10¹⁵–10¹⁸ FLOPs), 6 model sizes per budget, architecture search over (d_model, num_layers, num_heads) with budget tracking
- **`api_client.py`** — API client with caching, retry logic, and budget enforcement for querying training results
- **`scaling_law_fitter.py`** — power-law fitting from API results; extrapolation to 10¹⁹ FLOP target
- **`hyperparameter_selector.py`** — maps predicted N_opt to concrete architecture and training hyperparameters

Not executed: requires access to the Stanford cluster training API (VPN-gated). The experiment design, fitting, and extrapolation logic is complete and would run against any compatible training API endpoint.

Key references: [Chinchilla (arXiv:2203.15556)](https://arxiv.org/abs/2203.15556), [Kaplan et al. (arXiv:2001.08361)](https://arxiv.org/abs/2001.08361), [μP (arXiv:2203.03466)](https://arxiv.org/abs/2203.03466)

---

## Part 4: Data Filtering & Pipeline

**Goal:** Build a web-scale data processing pipeline to turn raw Common Crawl web data into a clean language modeling dataset, and measure the impact of filtering decisions on downstream LM perplexity.

### Pipeline Stages (implemented in `cs336_data/`)

| Stage | Tool / Method |
|---|---|
| WARC → text extraction | **Resiliparse** (`extract_plain_text`), **FastWARC** for record iteration, encoding detection |
| Language identification | **fastText** `lid.176.bin` classifier with confidence threshold filtering |
| PII masking | Regex-based detection and replacement of emails, phone numbers, IP addresses |
| Harmful content filtering | **fastText** NSFW + toxic speech classifiers from the **Dolma** project (Jigsaw-trained) |
| Quality heuristics | **Gopher** quality filters (word count, symbol/word ratio, stop-word presence, line length) |
| Quality classifier | **fastText** classifier trained on positive (Wikipedia-linked URLs) vs negative (raw CC) examples |
| Exact deduplication | Exact line-level deduplication across documents |
| Fuzzy deduplication | **MinHash + LSH** (locality-sensitive hashing) for near-duplicate document removal using Jaccard similarity on n-gram sets |
| Parallel processing | `concurrent.futures` for distributed processing of 5,000 WET files (~375 GB) |

### Status
Implementation and testing of the filtering pipeline is ongoing. Training and Paloma C4 100-domains evaluation will follow once the pipeline is complete.

---

## Repository Structure

```
Foundation_model_performance_and_scaling/
├── assignment1-basics/          # BPE tokenizer, Transformer, AdamW, training loop, ablations
├── assignment2-systems/         # Profiling harness, kernel optimizations, DDP, ZeRO sharding
│   ├── cs336_systems/           # All implemented modules
│   └── results/
│       ├── nsight_profiles/     # Nsight Systems GPU traces (.nsys-rep, .sqlite)
│       └── memory_profiling/    # PyTorch memory snapshots (.pickle)
├── assignment3-scaling/         # IsoFLOPs fitting, scaling law experiments, predictions
│   ├── part1_isoflops/
│   └── part2_scaling_laws/
├── assignment4-data/            # Common Crawl filtering pipeline
│   └── cs336_data/
└── transformer_workshop/        # Supplementary workshop materials
```

---

## Setup

Each part uses [`uv`](https://github.com/astral-sh/uv) for dependency management:

```bash
cd assignment1-basics   # or assignment2-systems, assignment3-scaling, assignment4-data
uv sync
uv run python <script>
uv run pytest
```

---

## Acknowledgments

Curriculum structure follows the publicly available [Stanford CS336 (Spring 2025)](https://stanford-cs336.github.io/spring2025/) course materials.
