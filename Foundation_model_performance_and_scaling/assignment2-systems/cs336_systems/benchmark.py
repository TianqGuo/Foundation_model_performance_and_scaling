"""
Benchmarking script for Transformer models.

This script provides utilities to benchmark forward and backward passes
of Transformer models with proper CUDA synchronization and warmup steps.
"""

import argparse
import timeit
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import nn

from cs336_basics.transformer_training.model import TransformerLM


@dataclass
class ModelConfig:
    """Configuration for different model sizes."""
    name: str
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int
    vocab_size: int = 10000
    batch_size: int = 4


# Model configurations from Table 1 in the assignment
MODEL_CONFIGS = {
    "small": ModelConfig(
        name="small",
        d_model=768,
        d_ff=3072,
        num_layers=12,
        num_heads=12,
    ),
    "medium": ModelConfig(
        name="medium",
        d_model=1024,
        d_ff=4096,
        num_layers=24,
        num_heads=16,
    ),
    "large": ModelConfig(
        name="large",
        d_model=1280,
        d_ff=5120,
        num_layers=36,
        num_heads=20,
    ),
    "xl": ModelConfig(
        name="xl",
        d_model=1600,
        d_ff=6400,
        num_layers=48,
        num_heads=25,
    ),
    "2.7B": ModelConfig(
        name="2.7B",
        d_model=2560,
        d_ff=10240,
        num_layers=32,
        num_heads=32,
    ),
}


def create_model(config: ModelConfig, context_length: int, device: str = "cuda") -> nn.Module:
    """
    Create a TransformerLM model with the given configuration.

    Args:
        config: Model configuration
        context_length: Maximum sequence length
        device: Device to place the model on

    Returns:
        Initialized TransformerLM model
    """
    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        device=device,
    )
    return model


def generate_random_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Generate a random batch of token IDs.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        vocab_size: Vocabulary size
        device: Device to place the tensor on

    Returns:
        Random token IDs of shape (batch_size, seq_len)
    """
    return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)


def benchmark_forward(
    model: nn.Module,
    input_ids: torch.Tensor,
    warmup_steps: int = 5,
    measure_steps: int = 10,
    use_sync: bool = True,
) -> tuple[float, float]:
    """
    Benchmark forward pass.

    Args:
        model: Model to benchmark
        input_ids: Input token IDs
        warmup_steps: Number of warmup steps before timing
        measure_steps: Number of steps to measure
        use_sync: Whether to call torch.cuda.synchronize() after each step

    Returns:
        Tuple of (mean_time, std_time) in seconds
    """
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = model(input_ids)
            if use_sync:
                torch.cuda.synchronize()

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(measure_steps):
            start = timeit.default_timer()
            _ = model(input_ids)
            if use_sync:
                torch.cuda.synchronize()
            end = timeit.default_timer()
            times.append(end - start)

    return np.mean(times), np.std(times)


def benchmark_forward_backward(
    model: nn.Module,
    input_ids: torch.Tensor,
    warmup_steps: int = 5,
    measure_steps: int = 10,
    use_sync: bool = True,
) -> tuple[float, float]:
    """
    Benchmark forward and backward pass.

    Args:
        model: Model to benchmark
        input_ids: Input token IDs
        warmup_steps: Number of warmup steps before timing
        measure_steps: Number of steps to measure
        use_sync: Whether to call torch.cuda.synchronize() after each step

    Returns:
        Tuple of (mean_time, std_time) in seconds
    """
    model.train()

    # Warmup
    for _ in range(warmup_steps):
        model.zero_grad()
        logits = model(input_ids)
        # Simple loss: sum of all logits (doesn't matter for benchmarking)
        loss = logits.sum()
        loss.backward()
        if use_sync:
            torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(measure_steps):
        model.zero_grad()
        start = timeit.default_timer()
        logits = model(input_ids)
        loss = logits.sum()
        loss.backward()
        if use_sync:
            torch.cuda.synchronize()
        end = timeit.default_timer()
        times.append(end - start)

    return np.mean(times), np.std(times)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Transformer models")
    parser.add_argument(
        "--model-size",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        default="small",
        help="Model size to benchmark",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
        help="Context length (sequence length)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--measure-steps",
        type=int,
        default=10,
        help="Number of measurement steps",
    )
    parser.add_argument(
        "--pass-type",
        type=str,
        choices=["forward", "forward_backward"],
        default="forward_backward",
        help="Type of pass to benchmark",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Disable torch.cuda.synchronize() (for testing)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )

    args = parser.parse_args()

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Get model configuration
    config = MODEL_CONFIGS[args.model_size]

    print(f"Benchmarking {config.name} model")
    print(f"  d_model: {config.d_model}")
    print(f"  d_ff: {config.d_ff}")
    print(f"  num_layers: {config.num_layers}")
    print(f"  num_heads: {config.num_heads}")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  context_length: {args.context_length}")
    print(f"  warmup_steps: {args.warmup_steps}")
    print(f"  measure_steps: {args.measure_steps}")
    print(f"  pass_type: {args.pass_type}")
    print(f"  use_sync: {not args.no_sync}")
    print(f"  device: {args.device}")
    print()

    # Create model
    print("Creating model...")
    model = create_model(config, args.context_length, device=args.device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters ({num_params / 1e6:.2f}M)")
    print()

    # Generate random batch
    print("Generating random batch...")
    input_ids = generate_random_batch(
        args.batch_size,
        args.context_length,
        config.vocab_size,
        device=args.device,
    )
    print()

    # Benchmark
    print(f"Running benchmark ({args.pass_type})...")
    if args.pass_type == "forward":
        mean_time, std_time = benchmark_forward(
            model,
            input_ids,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
            use_sync=not args.no_sync,
        )
    else:
        mean_time, std_time = benchmark_forward_backward(
            model,
            input_ids,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
            use_sync=not args.no_sync,
        )

    print()
    print("Results:")
    print(f"  Mean time: {mean_time * 1000:.2f} ms ({mean_time:.6f} s)")
    print(f"  Std dev: {std_time * 1000:.2f} ms ({std_time:.6f} s)")
    print(f"  Coefficient of variation: {(std_time / mean_time * 100):.2f}%")


if __name__ == "__main__":
    main()
