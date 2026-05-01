"""
Model Modifications for Leaderboard Optimization

- weight_tied_transformer.py: Transformer with weight tying between embeddings and LM head
"""

from .weight_tied_transformer import WeightTiedTransformerLM

__all__ = ["WeightTiedTransformerLM"]
