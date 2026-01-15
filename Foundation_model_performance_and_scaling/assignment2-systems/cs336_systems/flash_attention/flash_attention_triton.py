"""
Triton kernel implementation of FlashAttention-2.

This implementation uses fused Triton kernels for efficient GPU execution
with tiled computation and minimal memory transfers.

Forward pass uses Triton kernel. Backward pass can use either:
- PyTorch torch.compile (for 1.3.2(d) requirements)
- Triton kernel (optional, for 1.3.4 and leaderboard optimization)
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    """
    Triton kernel for FlashAttention-2 forward pass.

    Each program instance processes one query tile for one batch element.
    Launch grid: (T_q, batch_size)
    """
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # Load query tile (only once for all key tiles)
    Q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (Q_TILE_SIZE, D)

    # Initialize running statistics (on-chip buffers in float32)
    m_i = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    # Number of key tiles
    num_key_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)

    # For causal masking: compute query indices
    if is_causal:
        query_start = query_tile_index * Q_TILE_SIZE
        query_indices = query_start + tl.arange(0, Q_TILE_SIZE)

    # Loop over key tiles
    for j in range(num_key_tiles):
        # Load key and value tiles
        K_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (K_TILE_SIZE, D)
        V_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (K_TILE_SIZE, D)

        # Compute attention scores: S_ij = Q_i @ K_j^T * scale
        S_ij = tl.dot(Q_tile, tl.trans(K_tile)) * scale  # (Q_TILE_SIZE, K_TILE_SIZE)

        # Apply causal masking if needed
        if is_causal:
            key_start = j * K_TILE_SIZE
            key_indices = key_start + tl.arange(0, K_TILE_SIZE)
            # Create mask: query_idx >= key_idx for causal
            mask = query_indices[:, None] >= key_indices[None, :]
            # Add large negative value to masked positions
            S_ij = tl.where(mask, S_ij, -1e6)

        # Update running maximum
        m_ij = tl.max(S_ij, axis=1)  # (Q_TILE_SIZE,)
        m_i_new = tl.maximum(m_i, m_ij)

        # Compute unnormalized attention weights
        # P_tilde_ij = exp(S_ij - m_i_new)
        P_tilde_ij = tl.exp(S_ij - m_i_new[:, None])  # (Q_TILE_SIZE, K_TILE_SIZE)

        # Update running sum with correction
        # l_i_new = exp(m_i - m_i_new) * l_i + rowsum(P_tilde_ij)
        correction = tl.exp(m_i - m_i_new)
        l_i_new = correction * l_i + tl.sum(P_tilde_ij, axis=1)

        # Update running output with correction
        # O_i_new = diag(correction) @ O_i + P_tilde_ij @ V_j
        O_i = O_i * correction[:, None]

        # Cast P_tilde to V's dtype before matmul
        V_dtype = V_block_ptr.type.element_ty
        P_tilde_ij_casted = P_tilde_ij.to(V_dtype)

        # Accumulate: O_i += P_tilde_ij @ V_tile
        O_i = tl.dot(P_tilde_ij_casted, V_tile, acc=O_i)

        # Update running statistics
        m_i = m_i_new
        l_i = l_i_new

        # Advance block pointers to next tile
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # Normalize output by final sum
    # O_i = O_i / l_i
    O_i = O_i / l_i[:, None]

    # Compute logsumexp
    # L_i = m_i + log(l_i)
    L_i = m_i + tl.log(l_i)

    # Cast output to appropriate dtype before writing
    O_dtype = O_block_ptr.type.element_ty
    O_i = O_i.to(O_dtype)

    # Write results to global memory
    tl.store(O_block_ptr, O_i, boundary_check=(0, 1))
    tl.store(L_block_ptr, L_i, boundary_check=(0,))


@triton.jit
def flash_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, D_ptr,
    dO_ptr, dQ_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dob, stride_doq, stride_dod,
    stride_dqb, stride_dqq, stride_dqd,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS, scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    """
    Triton kernel for FlashAttention-2 backward pass (Algorithm 2).

    Each program instance processes one key tile for one batch element.
    Launch grid: (T_k, batch_size)

    Key optimization: outer loop over key tiles, inner loop over query tiles.
    This allows local accumulation of dK and dV without atomics.
    Only dQ requires atomic updates.
    """
    # Program indices
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Setup block pointers for key and value tiles (loaded once)
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # Load key and value tiles once
    K_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    V_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # Initialize gradient accumulators for this key tile
    dK_tile = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dV_tile = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

    # Number of query tiles
    num_query_tiles = tl.cdiv(N_QUERIES, Q_TILE_SIZE)

    # For causal masking: compute key indices
    if is_causal:
        key_start = key_tile_index * K_TILE_SIZE
        key_indices = key_start + tl.arange(0, K_TILE_SIZE)

    # Loop over query tiles
    for i in range(num_query_tiles):
        # Setup block pointers for this query tile
        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(i * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        O_block_ptr = tl.make_block_ptr(
            O_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(i * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        dO_block_ptr = tl.make_block_ptr(
            dO_ptr + batch_index * stride_dob,
            shape=(N_QUERIES, D),
            strides=(stride_doq, stride_dod),
            offsets=(i * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(i * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )

        D_block_ptr = tl.make_block_ptr(
            D_ptr + batch_index * stride_db,
            shape=(N_QUERIES,),
            strides=(stride_dq,),
            offsets=(i * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )

        # Load query tile data
        Q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        O_tile = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero")
        dO_tile = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
        L_tile = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        D_tile = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")

        # Compute attention scores: S = Q @ K^T / sqrt(d)
        S_tile = tl.dot(Q_tile, tl.trans(K_tile)) * scale  # (Q_TILE_SIZE, K_TILE_SIZE)

        # Apply causal masking if needed
        if is_causal:
            query_start = i * Q_TILE_SIZE
            query_indices = query_start + tl.arange(0, Q_TILE_SIZE)
            mask = query_indices[:, None] >= key_indices[None, :]
            S_tile = tl.where(mask, S_tile, -1e6)

        # Recompute attention probabilities: P = exp(S - L)
        P_tile = tl.exp(S_tile - L_tile[:, None])  # (Q_TILE_SIZE, K_TILE_SIZE)

        # Compute dV: dV += P^T @ dO
        # Cast P to dO's dtype before matmul
        dO_dtype = dO_block_ptr.type.element_ty
        P_tile_casted = P_tile.to(dO_dtype)
        dV_tile = tl.dot(tl.trans(P_tile_casted), dO_tile, acc=dV_tile)  # (K_TILE_SIZE, D)

        # Compute dP: dP = dO @ V^T
        dP_tile = tl.dot(dO_tile, tl.trans(V_tile))  # (Q_TILE_SIZE, K_TILE_SIZE)

        # Compute dS: dS = P * (dP - D) / sqrt(d)
        dS_tile = P_tile * (dP_tile - D_tile[:, None]) * scale  # (Q_TILE_SIZE, K_TILE_SIZE)

        # Compute dQ: dQ += dS @ K
        dQ_tile = tl.dot(dS_tile, K_tile)  # (Q_TILE_SIZE, D)

        # Atomic add dQ to global memory (multiple key tiles contribute to same query)
        # Need to use atomic operations for correctness
        dQ_ptrs = (
            dQ_ptr +
            batch_index * stride_dqb +
            (i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)[:, None]) * stride_dqq +
            tl.arange(0, D)[None, :] * stride_dqd
        )

        # Create boundary mask
        q_mask = (i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)[:, None]) < N_QUERIES
        d_mask = tl.arange(0, D)[None, :] < D
        mask = q_mask & d_mask

        # Atomic add for dQ
        tl.atomic_add(dQ_ptrs, dQ_tile, mask=mask)

        # Compute dK: dK += dS^T @ Q
        dK_tile = tl.dot(tl.trans(dS_tile), Q_tile, acc=dK_tile)  # (K_TILE_SIZE, D)

    # Write dK and dV to global memory (only once per key tile)
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # Cast to appropriate dtype before writing
    dK_dtype = dK_block_ptr.type.element_ty
    dV_dtype = dV_block_ptr.type.element_ty
    dK_tile = dK_tile.to(dK_dtype)
    dV_tile = dV_tile.to(dV_dtype)

    tl.store(dK_block_ptr, dK_tile, boundary_check=(0, 1))
    tl.store(dV_block_ptr, dV_tile, boundary_check=(0, 1))


class FlashAttentionTritonFunc(torch.autograd.Function):
    """
    FlashAttention-2 implementation using Triton kernel.

    Provides an efficient fused kernel implementation with minimal
    memory transfers and optimal GPU utilization.
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Forward pass using Triton kernel.

        Args:
            ctx: Context for saving tensors for backward pass
            Q: Query tensor of shape (batch_size, N_q, d)
            K: Key tensor of shape (batch_size, N_k, d)
            V: Value tensor of shape (batch_size, N_k, d)
            is_causal: Whether to apply causal masking

        Returns:
            O: Output tensor of shape (batch_size, N_q, d)
        """
        # Input validation
        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Inputs must be CUDA tensors"
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous(), \
            "Inputs must be contiguous"

        batch_size, N_q, d = Q.shape
        _, N_k, _ = K.shape

        # Tile sizes (tunable hyperparameters)
        Q_TILE_SIZE = 64
        K_TILE_SIZE = 64

        # Scale factor
        scale = 1.0 / math.sqrt(d)

        # Number of query tiles
        T_q = math.ceil(N_q / Q_TILE_SIZE)

        # Allocate output tensors
        O = torch.empty_like(Q)
        L = torch.empty(batch_size, N_q, device=Q.device, dtype=torch.float32)

        # Launch grid: (num_query_tiles, batch_size)
        grid = (T_q, batch_size)

        # Launch kernel
        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_q, N_k,
            scale,
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )

        # Save tensors for backward pass
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using Triton kernel (Algorithm 2).

        Implements tiled backward pass with atomic operations for dQ.
        Follows Algorithm 2 from FlashAttention-2 paper.

        Args:
            ctx: Context with saved tensors
            grad_output: Gradient w.r.t. output (batch_size, N_q, d)

        Returns:
            grad_Q, grad_K, grad_V, None (for is_causal flag)
        """
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal

        # Ensure contiguous
        grad_output = grad_output.contiguous()

        batch_size, N_q, d = Q.shape
        _, N_k, _ = K.shape

        # Tile sizes (smaller than forward due to more on-chip buffers in backward)
        # Backward needs to store: K_tile, V_tile, dK_tile, dV_tile, Q_tile, O_tile, dO_tile, etc.
        Q_TILE_SIZE = 32
        K_TILE_SIZE = 32

        # Scale factor
        scale = 1.0 / math.sqrt(d)

        # Pre-compute D = rowsum(O ◦ dO)
        D = torch.sum(O * grad_output, dim=-1)  # (batch_size, N_q)

        # Allocate gradient tensors
        grad_Q = torch.zeros_like(Q)
        grad_K = torch.empty_like(K)
        grad_V = torch.empty_like(V)

        # Number of key tiles
        T_k = math.ceil(N_k / K_TILE_SIZE)

        # Launch grid: (num_key_tiles, batch_size)
        grid = (T_k, batch_size)

        # Launch backward kernel
        flash_bwd_kernel[grid](
            Q, K, V, O, L, D,
            grad_output, grad_Q, grad_K, grad_V,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            D.stride(0), D.stride(1),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2),
            grad_Q.stride(0), grad_Q.stride(1), grad_Q.stride(2),
            grad_K.stride(0), grad_K.stride(1), grad_K.stride(2),
            grad_V.stride(0), grad_V.stride(1), grad_V.stride(2),
            N_q, N_k, scale,
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )

        # Return gradients for Q, K, V, and None for is_causal (no gradient needed)
        return grad_Q, grad_K, grad_V, None