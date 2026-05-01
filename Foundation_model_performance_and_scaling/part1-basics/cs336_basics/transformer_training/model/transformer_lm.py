"""
Transformer Language Model implementation.
"""

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int

from .embedding import Embedding
from .transformer_block import TransformerBlock
from .rmsnorm import RMSNorm
from .linear import Linear


class TransformerLM(nn.Module):
    """
    Transformer Language Model implementation.

    Full autoregressive language model consisting of:
    1. Token embedding layer
    2. Stack of Transformer blocks with causal self-attention and RoPE
    3. Final layer normalization
    4. Language modeling head for next-token prediction

    Args:
        vocab_size: Size of the vocabulary
        context_length: Maximum context length (for RoPE precomputation)
        d_model: Dimensionality of the model
        num_layers: Number of Transformer blocks
        num_heads: Number of attention heads per block
        d_ff: Dimensionality of feed-forward inner layer
        rope_theta: RoPE theta parameter (default: 10000.0)
        eps: Epsilon for RMSNorm (default: 1e-5)
        device: Device to place the module on
        dtype: Data type for the module parameters
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        # Token embedding layer
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        # Stack of Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                rope_theta=rope_theta,
                eps=eps,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])

        # Final layer normalization
        self.ln_final = RMSNorm(d_model=d_model, eps=eps, device=device, dtype=dtype)

        # Language modeling head
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        input_ids: Int[Tensor, "batch_size seq_len"],
    ) -> Float[Tensor, "batch_size seq_len vocab_size"]:
        """
        Forward pass of the Transformer language model.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)

        Returns:
            Logits over vocabulary for each position, shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        # Validate sequence length
        if seq_len > self.context_length:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds context length ({self.context_length})"
            )

        # Generate token positions for RoPE
        # Shape: (batch_size, seq_len)
        token_positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
        token_positions = token_positions.unsqueeze(0).expand(batch_size, seq_len)

        # Token embeddings
        # Shape: (batch_size, seq_len, d_model)
        x = self.token_embeddings(input_ids)

        # Pass through Transformer blocks
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)

        # Final layer normalization
        x = self.ln_final(x)

        # Language modeling head
        # Shape: (batch_size, seq_len, vocab_size)
        logits = self.lm_head(x)

        return logits

    def generate(
        self,
        input_ids: Int[Tensor, "batch_size seq_len"],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Int[Tensor, "batch_size new_seq_len"]:
        """Generate new tokens with optional temperature, top-k, and top-p sampling."""
        from cs336_basics.transformer_decode import decode as decode_tokens

        return decode_tokens(
            self,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
            generator=generator,
        )

