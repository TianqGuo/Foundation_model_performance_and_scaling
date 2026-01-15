"""
Triton kernel implementation of FlashAttention-2 forward pass.

This implementation uses a fused Triton kernel for efficient GPU execution
with tiled computation and minimal memory transfers.
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
        Backward pass - not implemented yet.

        This will be implemented in a later section of the assignment.
        """
        raise NotImplementedError("Backward pass not yet implemented")