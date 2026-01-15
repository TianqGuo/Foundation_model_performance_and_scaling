"""
FlashAttention-2 implementations.

This module provides both PyTorch and Triton implementations of FlashAttention-2.
"""

from .flash_attention_pytorch import FlashAttentionPyTorchFunc
from .flash_attention_triton import FlashAttentionTritonFunc

__all__ = [
    "FlashAttentionPyTorchFunc",
    "FlashAttentionTritonFunc",
]