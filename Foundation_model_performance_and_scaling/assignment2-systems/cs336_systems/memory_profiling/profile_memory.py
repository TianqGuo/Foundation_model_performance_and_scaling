"""
Memory profiling script for Transformer models.

This script profiles GPU memory usage during forward pass, backward pass, and optimizer step
using PyTorch's built-in memory profiler. The output can be visualized at:
https://pytorch.org/memory_viz

Usage:
    # Profile forward pass only
    python -m cs336_systems.memory_profiling.profile_memory \
        --model-size 2.7B --context-length 512 --profile-type forward

    # Profile full training step (forward + backward + optimizer)
    python -m cs336_systems.memory_profiling.profile_memory \
        --model-size 2.7B --context-length 512 --profile-type training

    # Profile with mixed precision
    python -m cs336_systems.memory_profiling.profile_memory \
        --model-size 2.7B --context-length 512 --profile-type training \
        --use-mixed-precision

    # Custom output path
    python -m cs336_systems.memory_profiling.profile_memory \
        --model-size 2.7B --context-length 512 \
        --output results/memory_profiling/custom_snapshot.pickle
"""

import argparse
import timeit
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from cs336_systems.profiling_benchmarking.benchmark import (
    MODEL_CONFIGS,
    create_model,
    generate_random_batch,
)


def profile_memory_forward_only(
    model: nn.Module,
    input_ids: torch.Tensor,
    warmup_steps: int = 5,
    measure_steps: int = 10,
    autocast_ctx=None,
) -> dict:
    """
    Profile memory usage for forward pass only.

    Args:
        model: Model to profile
        input_ids: Input token IDs
        warmup_steps: Number of warmup steps before profiling
        measure_steps: Number of steps to profile
        autocast_ctx: Autocast context for mixed precision (or nullcontext)

    Returns:
        Dictionary with profiling statistics
    """
    model.eval()

    if autocast_ctx is None:
        autocast_ctx = nullcontext()

    print(f"Running {warmup_steps} warmup steps...")
    # Warmup (not profiled)
    with torch.no_grad():
        for _ in range(warmup_steps):
            with autocast_ctx:
                _ = model(input_ids)
            torch.cuda.synchronize()

    # Get initial memory stats
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB

    print(f"Starting memory profiling for {measure_steps} forward passes...")

    # Start recording memory history
    torch.cuda.memory._record_memory_history(max_entries=1000000)

    # Measure forward passes with memory profiling
    start_time = timeit.default_timer()
    with torch.no_grad():
        for step in range(measure_steps):
            with autocast_ctx:
                output = model(input_ids)
            torch.cuda.synchronize()
    end_time = timeit.default_timer()

    # Get peak memory stats
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

    elapsed_time = end_time - start_time
    avg_time = elapsed_time / measure_steps

    stats = {
        "profile_type": "forward",
        "initial_memory_mb": initial_memory,
        "peak_memory_mb": peak_memory,
        "memory_increase_mb": peak_memory - initial_memory,
        "total_time_s": elapsed_time,
        "avg_time_s": avg_time,
        "num_steps": measure_steps,
    }

    return stats


def profile_memory_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    warmup_steps: int = 5,
    measure_steps: int = 10,
    autocast_ctx=None,
) -> dict:
    """
    Profile memory usage for full training step (forward + backward + optimizer).

    Args:
        model: Model to profile
        optimizer: Optimizer to use
        input_ids: Input token IDs
        warmup_steps: Number of warmup steps before profiling
        measure_steps: Number of steps to profile
        autocast_ctx: Autocast context for mixed precision (or nullcontext)

    Returns:
        Dictionary with profiling statistics
    """
    model.train()

    if autocast_ctx is None:
        autocast_ctx = nullcontext()

    print(f"Running {warmup_steps} warmup steps...")
    # Warmup (not profiled)
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        with autocast_ctx:
            logits = model(input_ids)
            loss = logits.sum()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

    # Get initial memory stats
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB

    print(f"Starting memory profiling for {measure_steps} training steps...")

    # Start recording memory history
    torch.cuda.memory._record_memory_history(max_entries=1000000)

    # Measure training steps with memory profiling
    start_time = timeit.default_timer()
    for step in range(measure_steps):
        optimizer.zero_grad()
        with autocast_ctx:
            logits = model(input_ids)
            loss = logits.sum()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
    end_time = timeit.default_timer()

    # Get peak memory stats
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

    elapsed_time = end_time - start_time
    avg_time = elapsed_time / measure_steps

    stats = {
        "profile_type": "training",
        "initial_memory_mb": initial_memory,
        "peak_memory_mb": peak_memory,
        "memory_increase_mb": peak_memory - initial_memory,
        "total_time_s": elapsed_time,
        "avg_time_s": avg_time,
        "num_steps": measure_steps,
    }

    return stats


def save_memory_snapshot(output_path: str):
    """Save memory snapshot to pickle file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    torch.cuda.memory._dump_snapshot(str(output_file))

    # Stop recording history
    torch.cuda.memory._record_memory_history(enabled=None)

    print(f"\nMemory snapshot saved to: {output_file}")
    print("\nTo visualize:")
    print("  1. Open https://pytorch.org/memory_viz in your browser")
    print("  2. Drag and drop the pickle file onto the page")
    print("  3. Explore the 'Active memory timeline' and allocation details")


def main():
    parser = argparse.ArgumentParser(
        description="Profile GPU memory usage of Transformer models"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        required=True,
        help="Model size to profile",
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
        help="Number of warmup steps (not profiled)",
    )
    parser.add_argument(
        "--measure-steps",
        type=int,
        default=10,
        help="Number of measurement steps",
    )
    parser.add_argument(
        "--profile-type",
        type=str,
        choices=["forward", "training"],
        default="forward",
        help="What to profile: 'forward' for forward-only, 'training' for full training step",
    )
    parser.add_argument(
        "--use-mixed-precision",
        action="store_true",
        help="Use mixed precision (BF16)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16"],
        default="bf16",
        help="Data type for mixed precision",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output pickle file path (default: results/memory_profiling/<model>_<context>_<type>.pickle)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a GPU.")
        return

    # Get model config
    config = MODEL_CONFIGS[args.model_size]

    # Setup mixed precision
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    if args.use_mixed_precision:
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            print("WARNING: BF16 not supported on this GPU, falling back to FP16")
            dtype = torch.float16
        autocast_ctx = torch.autocast(device_type="cuda", dtype=dtype)
        precision_str = args.dtype.upper()
    else:
        autocast_ctx = nullcontext()
        precision_str = "FP32"

    # Set default output path
    if args.output is None:
        mp_suffix = f"_{args.dtype}" if args.use_mixed_precision else ""
        output_filename = (
            f"{args.model_size}_ctx{args.context_length}_"
            f"{args.profile_type}{mp_suffix}_snapshot.pickle"
        )
        args.output = f"results/memory_profiling/{output_filename}"

    print("=" * 80)
    print(f"Memory Profiling: {config.name} model")
    print("=" * 80)
    print(f"Context length: {args.context_length}")
    print(f"Batch size: {args.batch_size}")
    print(f"Profile type: {args.profile_type}")
    print(f"Precision: {precision_str}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Measure steps: {args.measure_steps}")
    print(f"Output: {args.output}")
    print()

    # Create model
    print("Creating model...")
    try:
        model = create_model(config, args.context_length, device=args.device)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {num_params:,} parameters ({num_params / 1e6:.2f}M)")
        print()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("ERROR: Out of memory during model creation")
            torch.cuda.empty_cache()
            return
        else:
            raise

    # Generate input
    print("Generating random input batch...")
    input_ids = generate_random_batch(
        args.batch_size,
        args.context_length,
        config.vocab_size,
        device=args.device,
    )
    print()

    # Profile
    try:
        if args.profile_type == "forward":
            stats = profile_memory_forward_only(
                model, input_ids, args.warmup_steps, args.measure_steps, autocast_ctx
            )
        else:  # training
            # Create optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            stats = profile_memory_training_step(
                model, optimizer, input_ids, args.warmup_steps, args.measure_steps, autocast_ctx
            )

        # Save snapshot
        save_memory_snapshot(args.output)

        # Print statistics
        print("\n" + "=" * 80)
        print("MEMORY PROFILING STATISTICS")
        print("=" * 80)
        print(f"Profile type: {stats['profile_type']}")
        print(f"Initial memory: {stats['initial_memory_mb']:.2f} MB")
        print(f"Peak memory: {stats['peak_memory_mb']:.2f} MB")
        print(f"Memory increase: {stats['memory_increase_mb']:.2f} MB")
        print(f"Average time per step: {stats['avg_time_s'] * 1000:.2f} ms")
        print(f"Total time ({stats['num_steps']} steps): {stats['total_time_s']:.2f} s")
        print("=" * 80)
        print()

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\nERROR: Out of memory during profiling")
            print(f"Try reducing --context-length or using --use-mixed-precision")
            torch.cuda.empty_cache()
        else:
            raise


if __name__ == "__main__":
    main()
