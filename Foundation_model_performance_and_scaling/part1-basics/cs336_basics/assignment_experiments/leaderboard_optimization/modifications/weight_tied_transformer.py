"""
Weight-Tied Transformer Language Model

Implements weight tying between input embeddings and output projection layer.
This is a classic technique from the original Transformer paper (Vaswani et al., 2017)
and used in modern LLMs like PaLM (Chowdhery et al., 2022).

Benefits:
1. Reduces parameters significantly (~30-40% for small models)
2. Often improves performance by enforcing input/output consistency
3. Faster training due to fewer parameters
4. Better memory efficiency
"""

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int

from cs336_basics.transformer_training.model.embedding import Embedding
from cs336_basics.transformer_training.model.rmsnorm import RMSNorm
from cs336_basics.transformer_training.model.transformer_block import TransformerBlock


class WeightTiedTransformerLM(nn.Module):
    """
    Transformer Language Model with weight tying.

    The output projection layer (lm_head) shares weights with the input embedding layer.
    This reduces parameters and can improve performance on smaller models.

    Args:
        vocab_size: Size of the vocabulary
        context_length: Maximum context length
        d_model: Dimensionality of the model
        num_layers: Number of Transformer blocks
        num_heads: Number of attention heads per block
        d_ff: Dimensionality of feed-forward inner layer
        rope_theta: RoPE theta parameter (default: 10000.0)
        eps: Epsilon for RMSNorm (default: 1e-5)
        tie_weights: Whether to tie embedding and LM head weights (default: True)
        embedding_scale: Scale factor for embeddings (default: True, uses sqrt(d_model))
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
        tie_weights: bool = True,
        embedding_scale: bool = True,
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
        self.tie_weights = tie_weights
        self.embedding_scale = embedding_scale

        # Token embedding layer with custom initialization for weight tying
        # Use lower std when tying weights (typical: std = 1/sqrt(d_model) instead of default)
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        # Re-initialize embeddings with lower std for weight tying
        if tie_weights:
            # Standard practice: use 1/sqrt(d_model) for tied weights
            std = 1.0 / (d_model ** 0.5)
            with torch.no_grad():
                self.token_embeddings.weight.normal_(mean=0.0, std=std)

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
        # If weight tying is enabled, this will share parameters with token_embeddings
        if tie_weights:
            # Don't create separate parameters, will use token_embeddings.weight directly
            self.lm_head = None
            print(f"Weight tying enabled: sharing {vocab_size * d_model:,} parameters")
        else:
            # Create separate linear layer
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False, device=device, dtype=dtype)

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
        token_positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
        token_positions = token_positions.unsqueeze(0).expand(batch_size, seq_len)

        # Token embeddings
        x = self.token_embeddings(input_ids)

        # Optional embedding scaling (common with weight tying)
        if self.embedding_scale:
            x = x * (self.d_model ** 0.5)

        # Pass through Transformer blocks
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)

        # Final layer normalization
        x = self.ln_final(x)

        # Language modeling head
        if self.tie_weights:
            # Use transposed embedding weights as output projection
            # x: (batch, seq, d_model) @ weight.T: (d_model, vocab) -> (batch, seq, vocab)
            logits = torch.matmul(x, self.token_embeddings.weight.T)
        else:
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

    def get_num_params(self, non_embedding: bool = False):
        """
        Return the number of parameters in the model.

        Args:
            non_embedding: If True, exclude embedding parameters
        """
        n_params = sum(p.numel() for p in self.parameters())

        if non_embedding:
            # Subtract embedding parameters
            n_params -= self.token_embeddings.weight.numel()

        return n_params


def compare_parameter_counts():
    """
    Compare parameter counts between weight-tied and non-tied models.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

    from cs336_basics.transformer_training.model.transformer_lm import TransformerLM

    # Standard config
    config = {
        "vocab_size": 32000,
        "context_length": 256,
        "d_model": 512,
        "num_layers": 4,
        "num_heads": 16,
        "d_ff": 1344,
    }

    # Create both models
    tied_model = WeightTiedTransformerLM(**config, tie_weights=True)
    untied_model = WeightTiedTransformerLM(**config, tie_weights=False)
    standard_model = TransformerLM(**config)

    tied_params = sum(p.numel() for p in tied_model.parameters())
    untied_params = sum(p.numel() for p in untied_model.parameters())
    standard_params = sum(p.numel() for p in standard_model.parameters())

    print("Parameter Comparison:")
    print(f"  Standard model:     {standard_params:,}")
    print(f"  Weight-tied model:  {tied_params:,}")
    print(f"  Untied model:       {untied_params:,}")
    print(f"\nSavings from tying: {standard_params - tied_params:,} parameters")
    print(f"Reduction:          {(1 - tied_params/standard_params)*100:.1f}%")

    # Check embedding size
    emb_size = config["vocab_size"] * config["d_model"]
    print(f"\nEmbedding layer size: {emb_size:,} parameters")
    print(f"That's {emb_size/tied_params*100:.1f}% of total model size")


if __name__ == "__main__":
    compare_parameter_counts()
