"""
Optimized Triton kernel implementation of FlashAttention-2 for leaderboard.

Key optimizations:
1. Autotune tile sizes for optimal performance
2. Two-pass backward (separate dQ from dK/dV to avoid atomics)
3. Early termination for causal masking
4. Larger tile sizes for d_model=1024
"""

import torch
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        # Small tiles for large d_model (like 1024)
        triton.Config({'Q_TILE_SIZE': 16, 'K_TILE_SIZE': 16}, num_warps=2),
        triton.Config({'Q_TILE_SIZE': 16, 'K_TILE_SIZE': 16}, num_warps=4),
        triton.Config({'Q_TILE_SIZE': 16, 'K_TILE_SIZE': 32}, num_warps=4),
        triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 16}, num_warps=4),
        triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 32}, num_warps=4),
        # Medium tiles for medium d_model
        triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 64}, num_warps=4),
        triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 64}, num_warps=8),
    ],
    key=['N_QUERIES', 'N_KEYS', 'D'],
)
@triton.jit
def flash_fwd_kernel_optimized(
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
    Optimized Triton kernel for FlashAttention-2 forward pass with autotune.
    """
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Setup block pointers
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

    # Load query tile
    Q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # Initialize running statistics
    m_i = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    # Compute query indices for causal masking
    if is_causal:
        query_start = query_tile_index * Q_TILE_SIZE
        query_indices = query_start + tl.arange(0, Q_TILE_SIZE)

    # Determine number of key tiles to process
    num_key_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)

    # Early termination: for causal masking, only process tiles up to current query position
    if is_causal:
        # Maximum key index that can affect this query tile
        max_key_idx = query_start + Q_TILE_SIZE
        max_key_tiles = tl.cdiv(max_key_idx, K_TILE_SIZE)
        num_key_tiles = tl.minimum(num_key_tiles, max_key_tiles)

    # Loop over key tiles
    for j in range(num_key_tiles):
        # Load key and value tiles
        K_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # Compute attention scores
        S_ij = tl.dot(Q_tile, tl.trans(K_tile)) * scale

        # Apply causal masking
        if is_causal:
            key_start = j * K_TILE_SIZE
            key_indices = key_start + tl.arange(0, K_TILE_SIZE)
            mask = query_indices[:, None] >= key_indices[None, :]
            S_ij = tl.where(mask, S_ij, -1e6)

        # Online softmax update
        m_ij = tl.max(S_ij, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)

        P_tilde_ij = tl.exp(S_ij - m_i_new[:, None])

        correction = tl.exp(m_i - m_i_new)
        l_i_new = correction * l_i + tl.sum(P_tilde_ij, axis=1)

        O_i = O_i * correction[:, None]

        V_dtype = V_block_ptr.type.element_ty
        P_tilde_ij_casted = P_tilde_ij.to(V_dtype)

        O_i = tl.dot(P_tilde_ij_casted, V_tile, acc=O_i)

        m_i = m_i_new
        l_i = l_i_new

        # Advance to next tile
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # Normalize output
    O_i = O_i / l_i[:, None]
    L_i = m_i + tl.log(l_i)

    # Write results
    O_dtype = O_block_ptr.type.element_ty
    O_i = O_i.to(O_dtype)

    tl.store(O_block_ptr, O_i, boundary_check=(0, 1))
    tl.store(L_block_ptr, L_i, boundary_check=(0,))


@triton.autotune(
    configs=[
        # Small tiles for large d_model
        triton.Config({'Q_TILE_SIZE': 16, 'K_TILE_SIZE': 16}, num_warps=2),
        triton.Config({'Q_TILE_SIZE': 16, 'K_TILE_SIZE': 16}, num_warps=4),
        triton.Config({'Q_TILE_SIZE': 16, 'K_TILE_SIZE': 32}, num_warps=4),
        triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 16}, num_warps=4),
        # Medium tiles for smaller d_model
        triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 32}, num_warps=4),
        triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 32}, num_warps=4),
    ],
    key=['N_QUERIES', 'N_KEYS', 'D'],
)
@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, D_ptr,
    dO_ptr, dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dob, stride_doq, stride_dod,
    stride_dqb, stride_dqq, stride_dqd,
    N_QUERIES, N_KEYS, scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    """
    First pass: compute dQ only (no atomics needed).
    Launch grid: (num_query_tiles, batch_size)
    """
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Setup Q block pointer
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # Load Q, O, dO, L, D for this query tile
    Q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    O_tile = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero")

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dO_tile = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    L_tile = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    D_tile = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")

    # Initialize dQ accumulator
    dQ_tile = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    # Setup K and V block pointers (will iterate)
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

    # Query indices for causal masking
    if is_causal:
        query_start = query_tile_index * Q_TILE_SIZE
        query_indices = query_start + tl.arange(0, Q_TILE_SIZE)

    num_key_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)

    # Early termination for causal
    if is_causal:
        max_key_idx = query_start + Q_TILE_SIZE
        max_key_tiles = tl.cdiv(max_key_idx, K_TILE_SIZE)
        num_key_tiles = tl.minimum(num_key_tiles, max_key_tiles)

    # Loop over key tiles
    for j in range(num_key_tiles):
        # Load K and V tiles
        K_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # Recompute S and P
        S_tile = tl.dot(Q_tile, tl.trans(K_tile)) * scale

        if is_causal:
            key_start = j * K_TILE_SIZE
            key_indices = key_start + tl.arange(0, K_TILE_SIZE)
            mask = query_indices[:, None] >= key_indices[None, :]
            S_tile = tl.where(mask, S_tile, -1e6)

        P_tile = tl.exp(S_tile - L_tile[:, None])

        # Compute dP and dS
        dO_dtype = dO_block_ptr.type.element_ty
        dP_tile = tl.dot(dO_tile, tl.trans(V_tile))
        dS_tile = P_tile * (dP_tile - D_tile[:, None]) * scale

        # Accumulate dQ
        K_dtype = K_block_ptr.type.element_ty
        dS_tile_casted = dS_tile.to(K_dtype)
        dQ_tile = tl.dot(dS_tile_casted, K_tile, acc=dQ_tile)

        # Advance to next tile
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # Write dQ (no atomics needed!)
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D),
        strides=(stride_dqq, stride_dqd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    dQ_dtype = dQ_block_ptr.type.element_ty
    dQ_tile = dQ_tile.to(dQ_dtype)
    tl.store(dQ_block_ptr, dQ_tile, boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 32}, num_warps=4),
        triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 32}, num_warps=4),
        triton.Config({'Q_TILE_SIZE': 32, 'K_TILE_SIZE': 64}, num_warps=4),
        triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 64}, num_warps=8),
    ],
    key=['N_QUERIES', 'N_KEYS', 'D'],
)
@triton.jit
def flash_bwd_dkdv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr, D_ptr,
    dO_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dob, stride_doq, stride_dod,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS, scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    """
    Second pass: compute dK and dV only (no atomics needed).
    Launch grid: (num_key_tiles, batch_size)
    """
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Setup K and V block pointers
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

    # Load K and V tiles
    K_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    V_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # Initialize dK and dV accumulators
    dK_tile = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dV_tile = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

    # Key indices for causal masking
    if is_causal:
        key_start = key_tile_index * K_TILE_SIZE
        key_indices = key_start + tl.arange(0, K_TILE_SIZE)

    num_query_tiles = tl.cdiv(N_QUERIES, Q_TILE_SIZE)

    # Loop over query tiles
    for i in range(num_query_tiles):
        # Early termination: skip query tiles that don't affect this key tile
        if is_causal:
            query_start = i * Q_TILE_SIZE
            # If all queries in this tile are before all keys, skip
            if query_start + Q_TILE_SIZE <= key_start:
                continue

        # Setup Q, O, dO, L, D block pointers
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

        # Load tiles
        Q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        O_tile = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero")
        dO_tile = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
        L_tile = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        D_tile = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")

        # Recompute S and P
        S_tile = tl.dot(Q_tile, tl.trans(K_tile)) * scale

        if is_causal:
            query_start = i * Q_TILE_SIZE
            query_indices = query_start + tl.arange(0, Q_TILE_SIZE)
            mask = query_indices[:, None] >= key_indices[None, :]
            S_tile = tl.where(mask, S_tile, -1e6)

        P_tile = tl.exp(S_tile - L_tile[:, None])

        # Compute dV
        dO_dtype = dO_block_ptr.type.element_ty
        P_tile_casted = P_tile.to(dO_dtype)
        dV_tile = tl.dot(tl.trans(P_tile_casted), dO_tile, acc=dV_tile)

        # Compute dP and dS
        dP_tile = tl.dot(dO_tile, tl.trans(V_tile))
        dS_tile = P_tile * (dP_tile - D_tile[:, None]) * scale

        # Compute dK
        K_dtype = K_block_ptr.type.element_ty
        dS_tile_casted = dS_tile.to(K_dtype)
        dK_tile = tl.dot(tl.trans(dS_tile_casted), Q_tile, acc=dK_tile)

    # Write dK and dV
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

    dK_dtype = dK_block_ptr.type.element_ty
    dV_dtype = dV_block_ptr.type.element_ty
    dK_tile = dK_tile.to(dK_dtype)
    dV_tile = dV_tile.to(dV_dtype)

    tl.store(dK_block_ptr, dK_tile, boundary_check=(0, 1))
    tl.store(dV_block_ptr, dV_tile, boundary_check=(0, 1))


class FlashAttentionTritonOptimizedFunc(torch.autograd.Function):
    """
    Optimized FlashAttention-2 for leaderboard with two-pass backward.
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """Forward pass with autotuned kernel."""
        assert Q.is_cuda and K.is_cuda and V.is_cuda
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()

        batch_size, N_q, d = Q.shape
        _, N_k, _ = K.shape

        scale = 1.0 / math.sqrt(d)

        # Allocate outputs
        O = torch.empty_like(Q)
        L = torch.empty(batch_size, N_q, device=Q.device, dtype=torch.float32)

        # Use autotuned kernel (tile sizes determined by autotune)
        T_q = math.ceil(N_q / 64)  # Default, will be overridden by autotune
        grid = (T_q, batch_size)

        flash_fwd_kernel_optimized[grid](
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
            is_causal=is_causal,
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, grad_output):
        """
        Two-pass backward: dQ in first pass, dK/dV in second pass.
        NO ATOMICS - much faster!
        """
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal

        grad_output = grad_output.contiguous()

        batch_size, N_q, d = Q.shape
        _, N_k, _ = K.shape

        scale = 1.0 / math.sqrt(d)

        # Pre-compute D
        D = torch.sum(O * grad_output, dim=-1)

        # Allocate gradients
        grad_Q = torch.empty_like(Q)
        grad_K = torch.empty_like(K)
        grad_V = torch.empty_like(V)

        # Pass 1: Compute dQ (launch over query tiles)
        T_q = math.ceil(N_q / 64)
        grid_q = (T_q, batch_size)

        flash_bwd_dq_kernel[grid_q](
            Q, K, V, O, L, D,
            grad_output, grad_Q,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            D.stride(0), D.stride(1),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2),
            grad_Q.stride(0), grad_Q.stride(1), grad_Q.stride(2),
            N_q, N_k, scale,
            D=d,
            is_causal=is_causal,
        )

        # Pass 2: Compute dK and dV (launch over key tiles)
        T_k = math.ceil(N_k / 64)
        grid_k = (T_k, batch_size)

        flash_bwd_dkdv_kernel[grid_k](
            Q, K, V, O, L, D,
            grad_output, grad_K, grad_V,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            D.stride(0), D.stride(1),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2),
            grad_K.stride(0), grad_K.stride(1), grad_K.stride(2),
            grad_V.stride(0), grad_V.stride(1), grad_V.stride(2),
            N_q, N_k, scale,
            D=d,
            is_causal=is_causal,
        )

        return grad_Q, grad_K, grad_V, None