"""
Embedding module for the CS336 Transformer implementation.

This module implements an embedding layer that maps integer token IDs to dense vectors
without using PyTorch's built-in nn.Embedding.
"""

import torch
import torch.nn as nn
from einops import einsum
import torch.nn.functional as F

class Embedding(nn.Module):
    """
    Embedding layer that maps token IDs to dense vectors.

    This implementation creates a learnable embedding matrix and performs lookup
    operations by indexing into the matrix using token IDs.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        """
        Initialize the Embedding module.

        Args:
            num_embeddings (int): Size of the vocabulary
            embedding_dim (int): Dimension of the embedding vectors (d_model)
            device (torch.device | None): Device to store parameters on
            dtype (torch.dtype | None): Data type of the parameters
        """
        super().__init__()

        # Store dimensions
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Create embedding matrix parameter with shape (vocab_size, d_model)
        # This ensures d_model is the final dimension as required
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )

        # Initialize weights using truncated normal distribution
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using truncated normal distribution."""
        # Embedding initialization: N(μ=0, σ²=1) truncated at [-3, 3]
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup embedding vectors for the given token IDs.

        Args:
            token_ids (torch.Tensor): Token IDs with shape (batch_size, sequence_length)
                                     or any shape (...) containing integer token IDs

        Returns:
            torch.Tensor: Embedding vectors with shape (..., embedding_dim)
                         where ... matches the shape of token_ids
        """
        # Perform embedding lookup by indexing into the weight matrix
        # This is equivalent to one-hot encoding followed by matrix multiplication
        # but much more efficient
        # print(token_ids.shape)  # DEBUG: Commented out for benchmarking
        # print(self.weight.shape)  # DEBUG: Commented out for benchmarking
        return self.weight[token_ids]

        # this is much less efficient—don’t use in real code.
        # one_hot = F.one_hot(token_ids, num_classes=self.num_embeddings).to(self.weight.dtype)
        # print(one_hot.shape)
        # return einsum(one_hot, self.weight, "... v, v d -> ... d")

    def extra_repr(self) -> str:
        """Return extra representation string for debugging."""
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"