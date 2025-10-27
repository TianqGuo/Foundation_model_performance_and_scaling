"""
Profiling and benchmarking utilities for CS336 Assignment 2.

This module contains tools for benchmarking Transformer models,
analyzing warmup effects, and profiling GPU performance.
"""

from .benchmark import (
    MODEL_CONFIGS,
    ModelConfig,
    create_model,
    generate_random_batch,
    benchmark_forward,
    benchmark_forward_backward,
)

__all__ = [
    "MODEL_CONFIGS",
    "ModelConfig",
    "create_model",
    "generate_random_batch",
    "benchmark_forward",
    "benchmark_forward_backward",
]
