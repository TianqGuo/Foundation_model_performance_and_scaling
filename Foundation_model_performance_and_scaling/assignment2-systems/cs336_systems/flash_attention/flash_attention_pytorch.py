"""
Pure PyTorch implementation of FlashAttention-2 forward pass.

This implementation uses tiling and online softmax to compute attention
without materializing the full attention matrix. It's slower than the
Triton version but useful for debugging.
"""

import torch
import math


class FlashAttentionPyTorchFunc(torch.autograd.Function):
    """
    Pure PyTorch implementation of FlashAttention-2.

    Implements Algorithm 1 from FlashAttention-2 paper using tiled computation
    and online softmax to avoid storing the full attention matrix.
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Forward pass of FlashAttention-2.

        Args:
            ctx: Context for saving tensors for backward pass
            Q: Query tensor of shape (batch_size, N_q, d)
            K: Key tensor of shape (batch_size, N_k, d)
            V: Value tensor of shape (batch_size, N_k, d)
            is_causal: Whether to apply causal masking (ignored in this version)

        Returns:
            O: Output tensor of shape (batch_size, N_q, d)
        """
        batch_size, N_q, d = Q.shape
        _, N_k, _ = K.shape

        # Tile sizes (must be >= 16)
        B_q = 16  # Query tile size
        B_k = 16  # Key tile size

        # Number of tiles
        T_q = math.ceil(N_q / B_q)
        T_k = math.ceil(N_k / B_k)

        # Scale factor for attention scores
        scale = 1.0 / math.sqrt(d)

        # Initialize output and logsumexp
        O = torch.zeros_like(Q)
        L = torch.zeros(batch_size, N_q, device=Q.device, dtype=torch.float32)

        # Process each query tile
        for i in range(T_q):
            # Query tile indices
            q_start = i * B_q
            q_end = min((i + 1) * B_q, N_q)
            Q_i = Q[:, q_start:q_end, :]  # (batch_size, B_q, d)

            # Initialize running statistics for this query tile
            # m: running maximum, l: running sum, O_i: running output
            m_i = torch.full((batch_size, q_end - q_start), float('-inf'),
                           device=Q.device, dtype=torch.float32)
            l_i = torch.zeros(batch_size, q_end - q_start,
                            device=Q.device, dtype=torch.float32)
            O_i = torch.zeros(batch_size, q_end - q_start, d,
                            device=Q.device, dtype=torch.float32)

            # Process each key tile
            for j in range(T_k):
                # Key/Value tile indices
                k_start = j * B_k
                k_end = min((j + 1) * B_k, N_k)
                K_j = K[:, k_start:k_end, :]  # (batch_size, B_k, d)
                V_j = V[:, k_start:k_end, :]  # (batch_size, B_k, d)

                # Compute attention scores for this tile
                # S_ij = Q_i @ K_j^T / sqrt(d)
                S_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) * scale  # (batch_size, B_q, B_k)

                # Update running maximum
                m_ij = torch.max(S_ij, dim=-1)[0]  # (batch_size, B_q)
                m_i_new = torch.maximum(m_i, m_ij)

                # Compute unnormalized attention weights (numerator only)
                # P_tilde_ij = exp(S_ij - m_i_new)
                P_tilde_ij = torch.exp(S_ij - m_i_new.unsqueeze(-1))  # (batch_size, B_q, B_k)

                # Update running sum with correction for max change
                # l_i_new = exp(m_i - m_i_new) * l_i + rowsum(P_tilde_ij)
                correction = torch.exp(m_i - m_i_new)
                l_i_new = correction * l_i + torch.sum(P_tilde_ij, dim=-1)

                # Update running output with correction for max change
                # O_i_new = diag(exp(m_i - m_i_new)) @ O_i + P_tilde_ij @ V_j
                O_i = correction.unsqueeze(-1) * O_i + torch.matmul(P_tilde_ij, V_j)

                # Update running statistics
                m_i = m_i_new
                l_i = l_i_new

            # Normalize output by final sum
            # O_i = O_i / l_i
            O[:, q_start:q_end, :] = O_i / l_i.unsqueeze(-1)

            # Compute logsumexp for this query tile
            # L_i = m_i + log(l_i)
            L[:, q_start:q_end] = m_i + torch.log(l_i)

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