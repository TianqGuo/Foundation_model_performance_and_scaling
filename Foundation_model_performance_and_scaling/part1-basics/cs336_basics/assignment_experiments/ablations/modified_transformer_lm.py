"""
Modified Transformer Language Model for ablation experiments.
"""

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int

from cs336_basics.transformer_training.model.embedding import Embedding
from cs336_basics.transformer_training.model.rmsnorm import RMSNorm
from cs336_basics.transformer_training.model.linear import Linear

try:
    from .modified_transformer_block import FlexibleTransformerBlock
except ImportError:
    from modified_transformer_block import FlexibleTransformerBlock


class FlexibleTransformerLM(nn.Module):
    """
    Configurable Transformer Language Model for ablation experiments.

    Args:
        vocab_size: Size of the vocabulary
        context_length: Maximum context length
        d_model: Dimensionality of the model
        num_layers: Number of Transformer blocks
        num_heads: Number of attention heads per block
        d_ff: Dimensionality of feed-forward inner layer
        rope_theta: RoPE theta parameter (default: 10000.0)
        eps: Epsilon for RMSNorm (default: 1e-5)
        use_layer_norm: Whether to use layer normalization (default: True)
        post_norm: Whether to use post-norm instead of pre-norm (default: False)
        use_swiglu: Whether to use SwiGLU or SiLU FFN (default: True)
        use_rope: Whether to use RoPE position embeddings (default: True)
        use_final_norm: Whether to use final layer normalization (default: True)
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
        use_layer_norm: bool = True,
        post_norm: bool = False,
        use_swiglu: bool = True,
        use_rope: bool = True,
        use_final_norm: bool = True,
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
        self.use_layer_norm = use_layer_norm
        self.post_norm = post_norm
        self.use_rope = use_rope

        # Token embedding layer
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        # Stack of Transformer blocks
        self.layers = nn.ModuleList([
            FlexibleTransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                rope_theta=rope_theta,
                eps=eps,
                use_layer_norm=use_layer_norm,
                post_norm=post_norm,
                use_swiglu=use_swiglu,
                use_rope=use_rope,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])

        # Final layer normalization (may be disabled for ablation)
        if use_final_norm and use_layer_norm:
            self.ln_final = RMSNorm(d_model=d_model, eps=eps, device=device, dtype=dtype)
        else:
            self.ln_final = None

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

        # Generate token positions for RoPE (if used)
        if self.use_rope:
            token_positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
            token_positions = token_positions.unsqueeze(0).expand(batch_size, seq_len)
        else:
            token_positions = None

        # Token embeddings
        x = self.token_embeddings(input_ids)

        # Pass through Transformer blocks
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)

        # Final layer normalization (if enabled)
        if self.ln_final is not None:
            x = self.ln_final(x)

        # Language modeling head
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
