"""
Nsight Systems profiling tools for CS336 Assignment 2.

This module provides tools for profiling Transformer models with NVIDIA Nsight Systems.
"""

from .annotated_attention import annotated_scaled_dot_product_attention

__all__ = [
    "annotated_scaled_dot_product_attention",
]
