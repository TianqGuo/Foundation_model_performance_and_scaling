"""
Nsight Systems profiling script for Transformer models.

This script profiles forward pass, backward pass, and optimizer step with NVTX annotations.
Usage:
    nsys profile -o output.nsys-rep python profile_model.py --model-size small
"""

import argparse
import torch
import torch.cuda.nvtx as nvtx
from torch import nn

from cs336_basics.transformer_training.model import TransformerLM
from cs336_systems.profiling_benchmarking.benchmark import MODEL_CONFIGS, generate_random_batch


def profile_forward_only(
    model: nn.Module,
    input_ids: torch.Tensor,
    warmup_steps: int = 5,
    measure_steps: int = 10,
):
    """Profile forward pass only."""
    model.eval()

    # Warmup (not profiled)
    with nvtx.range("warmup"):
        with torch.no_grad():
            for _ in range(warmup_steps):
                _ = model(input_ids)
                torch.cuda.synchronize()

    # Measured forward passes
    with torch.no_grad():
        for step in range(measure_steps):
            with nvtx.range(f"forward_step_{step}"):
                output = model(input_ids)
                torch.cuda.synchronize()


def profile_forward_backward(
    model: nn.Module,
    input_ids: torch.Tensor,
    warmup_steps: int = 5,
    measure_steps: int = 10,
):
    """Profile forward and backward pass."""
    model.train()

    # Warmup (not profiled)
    with nvtx.range("warmup"):
        for _ in range(warmup_steps):
            model.zero_grad()
            logits = model(input_ids)
            loss = logits.sum()
            loss.backward()
            torch.cuda.synchronize()

    # Measured forward+backward passes
    for step in range(measure_steps):
        model.zero_grad()

        with nvtx.range(f"step_{step}"):
            with nvtx.range("forward"):
                logits = model(input_ids)
                loss = logits.sum()
                torch.cuda.synchronize()

            with nvtx.range("backward"):
                loss.backward()
                torch.cuda.synchronize()


def profile_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    warmup_steps: int = 5,
    measure_steps: int = 10,
):
    """Profile complete training step: forward + backward + optimizer."""
    model.train()

    # Warmup (not profiled)
    with nvtx.range("warmup"):
        for _ in range(warmup_steps):
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = logits.sum()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

    # Measured training steps
    for step in range(measure_steps):
        with nvtx.range(f"training_step_{step}"):
            with nvtx.range("forward"):
                optimizer.zero_grad()
                logits = model(input_ids)
                loss = logits.sum()
                torch.cuda.synchronize()

            with nvtx.range("backward"):
                loss.backward()
                torch.cuda.synchronize()

            with nvtx.range("optimizer_step"):
                optimizer.step()
                torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description="Profile Transformer model with Nsight Systems")
    parser.add_argument(
        "--model-size",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        default="small",
        help="Model size to profile",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
        help="Context length (128, 256, 512, or 1024)",
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
        choices=["forward", "forward_backward", "training"],
        default="forward_backward",
        help="What to profile",
    )
    parser.add_argument(
        "--use-annotated-attention",
        action="store_true",
        help="Use NVTX-annotated attention implementation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    # Get config
    config = MODEL_CONFIGS[args.model_size]

    print(f"Profiling {config.name} model")
    print(f"  Context length: {args.context_length}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Profile type: {args.profile_type}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Measure steps: {args.measure_steps}")
    print()

    # Create model
    with nvtx.range("model_creation"):
        model = TransformerLM(
            vocab_size=config.vocab_size,
            context_length=args.context_length,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            device=args.device,
        )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters ({num_params / 1e6:.2f}M)")
    print()

    # Optionally use annotated attention
    if args.use_annotated_attention:
        print("Using NVTX-annotated attention implementation")
        from cs336_systems.nsight_systems_profiler.annotated_attention import (
            annotated_scaled_dot_product_attention
        )
        import cs336_basics.transformer_training.model as model_module
        model_module.scaled_dot_product_attention = annotated_scaled_dot_product_attention
        print()

    # Generate random batch
    with nvtx.range("data_generation"):
        input_ids = generate_random_batch(
            args.batch_size,
            args.context_length,
            config.vocab_size,
            device=args.device,
        )

    # Profile
    print(f"Starting profiling ({args.profile_type})...")
    print("Note: Warmup steps are marked with NVTX and can be filtered out in nsys viewer")
    print()

    if args.profile_type == "forward":
        profile_forward_only(model, input_ids, args.warmup_steps, args.measure_steps)

    elif args.profile_type == "forward_backward":
        profile_forward_backward(model, input_ids, args.warmup_steps, args.measure_steps)

    elif args.profile_type == "training":
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        profile_training_step(model, optimizer, input_ids, args.warmup_steps, args.measure_steps)

    print("Profiling complete!")
    print()
    print("To view the profile:")
    print("  1. Open Nsight Systems GUI on your local machine")
    print("  2. Load the .nsys-rep file")
    print("  3. Use NVTX ranges to filter out warmup steps")
    print("  4. Check 'CUDA GPU Kernel Summary' in Stats System View")


if __name__ == "__main__":
    main()
