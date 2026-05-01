# FlashAttention-2: Background and Implementation Guide

This document provides the mathematical background, algorithm details, and implementation guidance for FlashAttention-2, referenced in REQUIREMENTS.md Section 1.3.2.

## Table of Contents
1. [Recommended Reading](#recommended-reading)
2. [Understanding Inefficiencies in Vanilla Attention](#understanding-inefficiencies-in-vanilla-attention)
3. [FlashAttention Techniques](#flashattention-techniques)
4. [Backward Pass with Recomputation](#backward-pass-with-recomputation)
5. [Forward Pass Details](#forward-pass-details)
6. [Algorithm 1: FlashAttention-2 Forward Pass](#algorithm-1-flashattention-2-forward-pass)
7. [Triton Tips and Tricks](#triton-tips-and-tricks)

---

## Recommended Reading

Before implementing FlashAttention-2, review these resources:

- **[Dao et al., 2022]** - Original FlashAttention paper: provides intuition for computing softmax online across tiles
- **[Dao, 2023]** - FlashAttention-2 paper: improved version with better parallelization and efficiency
- **[Milakov and Gimelshein, 2018]** - Original online softmax technique
- **[He, 2022]** - GPU execution intuition: how GPUs actually execute PyTorch code

---

## Understanding Inefficiencies in Vanilla Attention

### Forward Pass

The standard attention forward pass (ignoring masking) is:

```
S = QK⊤/√d                    (4)
P_ij = softmax_j(S)_ij        (5)
O = PV                         (6)
```

Where:
- `Q, K, V` are query, key, value matrices
- `S` is the pre-softmax attention scores
- `P` is the attention probability matrix (post-softmax)
- `O` is the output

### Standard Backward Pass

```
dV = P⊤dO                                              (7)
dP = dOV⊤                                              (8)
dS_i = dsoftmax(dP_i) = (diag(P_i) - P_iP_i⊤) dP_i  (9)
dQ = dSK/√d                                            (10)
dK = dS⊤Q/√d                                           (11)
```

### The Problem: Memory and IO Bottlenecks

**Quadratic Memory Cost**: The attention matrix `P` has shape `(batch_size, n_heads, seq_len, seq_len)`—its size grows **quadratically** with sequence length. For example:
- seq_len=1024: ~1M elements per head
- seq_len=4096: ~16M elements per head (16× larger!)
- seq_len=16384: ~268M elements per head (268× larger!)

**High Memory IO Costs**: During forward and backward passes, we repeatedly transfer `P` and other large activations between:
- **SRAM (on-chip, fast)**: ~20-30 MB on modern GPUs, ~20 TB/s bandwidth
- **HBM (off-chip, slower)**: 40-80 GB on modern GPUs, ~2-3 TB/s bandwidth

Example: A standard backward pass reads `P` from HBM multiple times:
1. Once for computing `dV = P⊤dO` (Equation 7)
2. Again for computing `dS` via the softmax gradient (Equation 9)

Each read of a 16M element matrix (64 MB in FP32) at 2 TB/s takes ~32 μs—this adds up across layers and training steps!

---

## FlashAttention Techniques

FlashAttention-2 eliminates these bottlenecks using three core techniques:

### 1. Tiling

**Goal**: Compute attention without needing the entire attention matrix in memory at once.

**Approach**:
- Split inputs `Q`, `K`, `V` into smaller tiles
- Process tiles incrementally in nested loops
- Compute output tiles by accumulating partial results from key-value tiles

**Challenge**: Softmax requires entire rows of `S` for normalization. How do we tile when softmax needs global information?

**Solution**: Online softmax (see next section on Forward Pass Details)

### 2. Recomputation

**Goal**: Avoid storing large intermediate matrices in HBM.

**Approach**:
- **Forward pass**: Instead of saving `P` (quadratic size), save only:
  - Input tensors `Q, K, V` (linear size)
  - Logsumexp values `L` (linear size)
- **Backward pass**: Recompute `P` on-the-fly from `Q, K, L` as needed

**Logsumexp** `L` is defined as:
```
L_i = log(Σ_j exp(S_ij))   (12)
```

**Memory savings**:
- Without recomputation: Store `P` = `O(batch * heads * seq_len²)` elements
- With recomputation: Store `L` = `O(batch * heads * seq_len)` elements
- **Reduction**: `O(seq_len²)` → `O(seq_len)` storage!

### 3. Operator Fusion

**Goal**: Minimize data transfer between SRAM and HBM.

**Approach**:
- Fuse all attention operations into a **single Triton kernel**
- Load inputs from HBM once → compute in SRAM → write outputs to HBM once
- Avoid materializing intermediate results (like `P`) in HBM

**Standard PyTorch**: Each operation (matmul, softmax, etc.) is a separate kernel:
```
S = Q @ K.T  # Kernel 1: write S to HBM
P = softmax(S)  # Kernel 2: read S from HBM, write P to HBM
O = P @ V  # Kernel 3: read P from HBM, write O to HBM
```

**FlashAttention**: Single kernel does everything:
```
flash_attention(Q, K, V) → O  # One kernel: read Q,K,V, write O
```

---

## Backward Pass with Recomputation

Using the saved logsumexp `L` and a pre-computed helper vector `D`, we can compute gradients efficiently without storing `P`.

### Pre-computation (before backward kernel)

Compute and store in global memory:
```
D = rowsum(O ◦ dO)
```

where `◦` is element-wise multiplication.

**Why this works**: `D = rowsum(P ◦ dP)` because:
```
P @ dP⊤ = P @ (dO @ V⊤)⊤ = P @ (V @ dO⊤) = (P @ V) @ dO⊤ = O @ dO⊤
```
Therefore: `rowsum(P ◦ dP) = diag(P @ dP⊤) = diag(O @ dO⊤) = rowsum(O ◦ dO)`

### Modified Backward Pass

With `L` and `D`, we can recompute `P` without storing it:

```
S = QK⊤/√d                     (13)
P_ij = exp(S_ij - L_i)        (14)  ← Recompute P from S and L
dV = P⊤dO                      (15)
dP = dOV⊤                      (16)
dS_ij = P_ij ◦ (dP_ij - D_i)  (17)  ← Use precomputed D
dQ = dSK/√d                    (18)
dK = dS⊤Q/√d                   (19)
```

**Key insight**: We never need to store `P` in HBM during the forward pass. Instead:
1. Forward: Compute and discard `P`, save `L`
2. Backward: Recompute `P` from `Q, K, L` as needed (Equations 13-14)

---

## Forward Pass Details

### Online Softmax

**Problem**: Standard softmax requires two passes over data:
```
# Pass 1: Find max for numerical stability
max_val = max(x)

# Pass 2: Compute exp and sum
exp_sum = sum(exp(x - max_val))

# Pass 3: Normalize
result = exp(x - max_val) / exp_sum
```

**Solution**: Online softmax maintains running statistics while processing tiles.

### Notation

- **Query tiles**: Size `B_q`, indexed by `i ∈ {1, ..., T_q}` where `T_q = ⌈N_q / B_q⌉`
- **Key tiles**: Size `B_k`, indexed by `j ∈ {1, ..., T_k}` where `T_k = ⌈N_k / B_k⌉`
- **Superscript** `(j)`: Current key tile iteration
- **Subscript** `i`: Current query tile

### Running Statistics Per Query Tile

For each query tile `i`, maintain:

1. **`m_i^(j)` ∈ R^(B_q)**: Running row-wise maximum
   - Used for numerically stable softmax
   - Updated: `m_i^(j) = max(m_i^(j-1), rowmax(S_i^(j)))`
   - Initialize: `m_i^(0) = -∞`

2. **`l_i^(j)` ∈ R^(B_q)**: Running proxy for softmax denominator
   - Tracks the sum of exponentiated values
   - Updated: `l_i^(j) = exp(m_i^(j-1) - m_i^(j)) l_i^(j-1) + rowsum(P̃_i^(j))`
   - Initialize: `l_i^(0) = 0`

3. **`O_i^(j)` ∈ R^(B_q × d)**: Running output accumulator
   - Accumulates weighted values across key tiles
   - Updated: `O_i^(j) = diag(exp(m_i^(j-1) - m_i^(j))) O_i^(j-1) + P̃_i^(j) V^(j)`
   - Initialize: `O_i^(0) = 0`

### Unnormalized Attention Scores

```
P̃_i^(j) = exp(S_i^(j) - m_i^(j))
```

This is the unnormalized softmax (numerator only). The denominator is tracked separately in `l_i^(j)`.

### Final Normalization

After processing all key tiles (when `j = T_k`):
```
O_i = diag(l_i^(T_k))^(-1) O_i^(T_k)  ← Normalize by final denominator
L_i = m_i^(T_k) + log(l_i^(T_k))      ← Compute logsumexp for backward
```

---

## Algorithm 1: FlashAttention-2 Forward Pass

```
Input: Q ∈ R^(N_q×d), K, V ∈ R^(N_k×d), tile sizes B_q, B_k

Split Q into T_q = ⌈N_q/B_q⌉ tiles Q_1,...,Q_Tq of size B_q×d
Split K,V into T_k = ⌈N_k/B_k⌉ tiles K^(1),...,K^(Tk) and V^(1),...,V^(Tk) of size B_k×d

for i = 1,...,T_q do                          ← Outer loop: query tiles
  Load Q_i from global memory
  Initialize O_i^(0) = 0 ∈ R^(B_q×d), l_i^(0) = 0 ∈ R^(B_q), m_i^(0) = -∞ ∈ R^(B_q)

  for j = 1,...,T_k do                        ← Inner loop: key tiles
    Load K^(j), V^(j) from global memory

    # Compute attention scores for this tile
    Compute S_i^(j) = Q_i(K^(j))⊤/√d ∈ R^(B_q×B_k)

    # Update running maximum
    Compute m_i^(j) = max(m_i^(j-1), rowmax(S_i^(j))) ∈ R^(B_q)

    # Compute unnormalized attention (numerator only)
    Compute P̃_i^(j) = exp(S_i^(j) - m_i^(j)) ∈ R^(B_q×B_k)

    # Update running denominator (with correction for max change)
    Compute l_i^(j) = exp(m_i^(j-1) - m_i^(j)) l_i^(j-1) + rowsum(P̃_i^(j)) ∈ R^(B_q)

    # Update running output (with correction for max change)
    Compute O_i^(j) = diag(exp(m_i^(j-1) - m_i^(j))) O_i^(j-1) + P̃_i^(j) V^(j)
  end for

  # Final normalization
  Compute O_i = diag(l_i^(T_k))^(-1) O_i^(T_k)
  Compute L_i = m_i^(T_k) + log(l_i^(T_k))

  Write O_i to global memory as the i-th tile of O
  Write L_i to global memory as the i-th tile of L
end for

Return output O and logsumexp L
```

### Algorithm Walkthrough Example

Consider a small example with:
- `N_q = 32`, `N_k = 32`, `d = 64`
- Tile sizes: `B_q = 16`, `B_k = 16`
- This gives us: `T_q = 2` query tiles, `T_k = 2` key tiles

**Iteration `i=1, j=1`** (first query tile, first key tile):
1. Load `Q_1` (rows 0-15 of Q)
2. Initialize: `O^(0) = 0`, `l^(0) = 0`, `m^(0) = -∞`
3. Load `K^(1), V^(1)` (rows 0-15 of K, V)
4. Compute `S^(1) = Q_1 @ K^(1).T / √d` → shape (16, 16)
5. Update max: `m^(1) = rowmax(S^(1))` → shape (16,)
6. Compute unnormalized attention: `P̃^(1) = exp(S^(1) - m^(1))` → shape (16, 16)
7. Update denominator: `l^(1) = 0 + rowsum(P̃^(1))` → shape (16,)
8. Update output: `O^(1) = 0 + P̃^(1) @ V^(1)` → shape (16, 64)

**Iteration `i=1, j=2`** (first query tile, second key tile):
1. `Q_1` still loaded
2. Load `K^(2), V^(2)` (rows 16-31 of K, V)
3. Compute `S^(2) = Q_1 @ K^(2).T / √d`
4. Update max: `m^(2) = max(m^(1), rowmax(S^(2)))`
   - If max increased, previous values need correction!
5. Compute `P̃^(2) = exp(S^(2) - m^(2))`
6. Update denominator with correction: `l^(2) = exp(m^(1) - m^(2)) * l^(1) + rowsum(P̃^(2))`
   - The correction factor `exp(m^(1) - m^(2))` adjusts previous denominator
7. Update output with correction: `O^(2) = diag(exp(m^(1) - m^(2))) @ O^(1) + P̃^(2) @ V^(2)`
   - The correction factor adjusts previous output for the new max

**Finalization for `i=1`**:
1. Normalize: `O_1 = diag(1/l^(2)) @ O^(2)`
2. Compute logsumexp: `L_1 = m^(2) + log(l^(2))`
3. Write `O_1` and `L_1` to global memory

Then repeat for `i=2` (second query tile, rows 16-31 of Q).

---

## Triton Tips and Tricks

### Debugging

1. **Print statements**: Use `tl.device_print` for debugging
   ```python
   tl.device_print("Value of x:", x)
   ```
   Documentation: https://triton-lang.org/main/python-api/generated/triton.language.device_print.html

2. **CPU interpreter**: Set `TRITON_INTERPRET=1` to run on CPU
   ```bash
   TRITON_INTERPRET=1 python your_script.py
   ```
   Note: This can be buggy; use with caution

3. **Compare against PyTorch**: Implement the same algorithm in pure PyTorch first (part a), then compare results operation-by-operation when writing Triton

### Block Pointers

1. **Setup checklist**:
   - ✓ Pointer to first element
   - ✓ Overall tensor shape (for bounds checking)
   - ✓ Strides for each dimension
   - ✓ Starting offsets (usually `program_id * tile_size`)
   - ✓ Block shape (tile size)
   - ✓ Order (dimension ordering, e.g., `(1, 0)` for column-major)

2. **Offsets**: Multiply by tile size!
   ```python
   # Correct
   offsets=(query_tile_index * Q_TILE_SIZE, 0)

   # Wrong (common mistake)
   offsets=(query_tile_index, 0)  # Forgot to multiply by tile size!
   ```

3. **Advancing**: Explicitly advance pointers in loops
   ```python
   for j in range(num_tiles):
       tile = tl.load(block_ptr, boundary_check=(0, 1))
       # ... process tile ...
       block_ptr = block_ptr.advance((0, TILE_SIZE))  # Move to next tile
   ```

### Launch Grid

The launch grid determines how many parallel program instances run:

```python
# Launch grid of (T_q, batch_size)
flash_fwd_kernel[(T_q, batch_size)](
    Q, K, V, O, L,
    # ... strides and other args ...
)
```

Inside kernel:
```python
query_tile_index = tl.program_id(0)  # 0 to T_q-1
batch_index = tl.program_id(1)       # 0 to batch_size-1
```

### Common Operations

1. **Matrix multiply**: Use `tl.dot`
   ```python
   S = tl.dot(Q_tile, K_tile)  # Q_tile @ K_tile.T for appropriate shapes
   ```

2. **Reductions**: Use `tl.sum`, `tl.max` with `axis` parameter
   ```python
   row_max = tl.max(S, axis=1)         # Max along columns (per row)
   row_sum = tl.sum(P_tilde, axis=1)   # Sum along columns (per row)
   ```

3. **Broadcasting**: Add `None` to expand dimensions
   ```python
   # S has shape (B_q, B_k), m has shape (B_q,)
   P_tilde = tl.exp(S - m[:, None])  # Broadcast m to match S
   ```

### Precision Guidelines

1. **On-chip buffers**: Use `tl.float32` for accumulators
   ```python
   output = tl.zeros((TILE_SIZE, D), dtype=tl.float32)
   l = tl.zeros((TILE_SIZE,), dtype=tl.float32)
   m = tl.full((TILE_SIZE,), float('-inf'), dtype=tl.float32)
   ```

2. **Matrix multiply accumulation**: Use `acc` parameter
   ```python
   # Accumulate into existing buffer
   output = tl.dot(P_tilde, V_tile, acc=output)
   ```

3. **Type casting**: Cast before operations
   ```python
   # Get dtype of V tile
   v_dtype = V_block_ptr.type.element_ty

   # Cast P_tilde to match V before multiply
   P_tilde_casted = P_tilde.to(v_dtype)
   output_chunk = tl.dot(P_tilde_casted, V_tile)

   # Cast output to appropriate dtype before writing
   output_final = output.to(v_dtype)
   tl.store(O_block_ptr, output_final)
   ```

4. **Boundary checking**: Always check bounds when tile doesn't evenly divide
   ```python
   tl.load(block_ptr, boundary_check=(0, 1), padding_option="zero")
   tl.store(block_ptr, value, boundary_check=(0,))
   ```

---

## Additional Resources

- **Triton Language Documentation**: https://triton-lang.org/main/python-api/triton.language.html
- **Triton Tutorials**: https://triton-lang.org/main/getting-started/tutorials/index.html
- **FlashAttention GitHub**: https://github.com/Dao-AILab/flash-attention
- **Original FlashAttention Paper**: https://arxiv.org/abs/2205.14135
- **FlashAttention-2 Paper**: https://arxiv.org/abs/2307.08691

---

**Note**: This document is a companion to `REQUIREMENTS.md` Section 1.3.2. Refer to the requirements document for deliverables and grading criteria.