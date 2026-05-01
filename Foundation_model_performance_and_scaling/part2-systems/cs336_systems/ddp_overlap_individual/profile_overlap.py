#!/usr/bin/env python3
# ==============================================================================
# Profile DDP with Overlapping Computation and Communication
# ==============================================================================
#
# DESCRIPTION:
#   Script for profiling DDP implementations with Nsight Systems to visualize
#   whether communication overlaps with backward computation.
#
# USAGE:
#   # Profile naive DDP (no overlap)
#   nsys profile -o naive_ddp --trace=cuda,nvtx \
#       uv run python profile_overlap.py --implementation naive
#
#   # Profile overlap DDP (with overlap)
#   nsys profile -o overlap_ddp --trace=cuda,nvtx \
#       uv run python profile_overlap.py --implementation overlap
#
# OUTPUT:
#   - naive_ddp.nsys-rep: Profiling report for naive DDP
#   - overlap_ddp.nsys-rep: Profiling report for overlap DDP
#
# VIEW RESULTS:
#   Open the .nsys-rep files in Nsight Systems GUI to compare traces
#
# ==============================================================================

import argparse
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from naive_ddp.naive_ddp_trainer import setup_distributed, cleanup_distributed

# Model configuration (XL model as specified)
MODEL_CONFIG = {
    "d_model": 1600,
    "d_ff": 6400,
    "num_layers": 48,
    "num_heads": 25,
}


def create_model():
    """Create a Transformer model for profiling."""
    try:
        from cs336_basics.transformer_training.model import TransformerLM

        model = TransformerLM(
            vocab_size=10000,
            context_length=512,
            d_model=MODEL_CONFIG["d_model"],
            num_layers=MODEL_CONFIG["num_layers"],
            num_heads=MODEL_CONFIG["num_heads"],
            d_ff=MODEL_CONFIG["d_ff"],
        )
        return model

    except Exception as exc:
        print(f"Warning: cs336_basics import failed ({exc}), using fallback model")

        class FallbackTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(10000, MODEL_CONFIG["d_model"])
                self.layers = nn.ModuleList([
                    nn.TransformerEncoderLayer(
                        d_model=MODEL_CONFIG["d_model"],
                        nhead=MODEL_CONFIG["num_heads"],
                        dim_feedforward=MODEL_CONFIG["d_ff"],
                        batch_first=True,
                    )
                    for _ in range(MODEL_CONFIG["num_layers"])
                ])
                self.head = nn.Linear(MODEL_CONFIG["d_model"], 10000)

            def forward(self, x):
                x = self.embed(x)
                for layer in self.layers:
                    x = layer(x)
                return self.head(x)

        return FallbackTransformer()


def profile_naive_ddp(
    rank: int,
    world_size: int,
    num_steps: int,
    batch_size: int,
):
    """Profile naive DDP (no overlap)."""
    try:
        # Setup distributed
        setup_distributed(rank, world_size, backend="nccl")
        device = f"cuda:{rank}"
        torch.cuda.set_device(device)

        # Create model
        model = create_model().to(device)

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Loss function
        class SequenceCrossEntropyLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.loss_fn = nn.CrossEntropyLoss()

            def forward(self, logits, targets):
                vocab_size = logits.shape[-1]
                return self.loss_fn(
                    logits.view(-1, vocab_size),
                    targets.view(-1),
                )

        loss_fn = SequenceCrossEntropyLoss()

        # Generate synthetic data
        torch.manual_seed(42 + rank)
        seq_len = 512
        num_samples = num_steps * batch_size

        inputs = torch.randint(0, 10000, (num_samples, seq_len))
        targets = torch.randint(0, 10000, (num_samples, seq_len))

        # Profile training steps
        for step in range(num_steps):
            start_idx = step * batch_size
            end_idx = start_idx + batch_size

            batch_inputs = inputs[start_idx:end_idx].to(device)
            batch_targets = targets[start_idx:end_idx].to(device)

            # Mark NVTX range for this iteration
            torch.cuda.nvtx.range_push(f"Training Step {step}")

            optimizer.zero_grad()

            # Forward pass
            torch.cuda.nvtx.range_push("Forward")
            outputs = model(batch_inputs)
            loss = loss_fn(outputs, batch_targets)
            torch.cuda.nvtx.range_pop()

            # Backward pass
            torch.cuda.nvtx.range_push("Backward")
            loss.backward()
            torch.cuda.nvtx.range_pop()

            # All-reduce gradients (naive implementation - individual params)
            torch.cuda.nvtx.range_push("All-Reduce (Naive)")
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                        param.grad.data /= world_size
            torch.cuda.nvtx.range_pop()

            # Optimizer step
            torch.cuda.nvtx.range_push("Optimizer Step")
            optimizer.step()
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_pop()  # Training Step

        cleanup_distributed()

    except Exception as e:
        if rank == 0:
            print(f"Error in naive DDP profiling (rank {rank}): {e}")
            import traceback
            traceback.print_exc()


def profile_overlap_ddp(
    rank: int,
    world_size: int,
    num_steps: int,
    batch_size: int,
):
    """Profile overlap DDP (with overlap)."""
    try:
        from ddp_overlap_individual.ddp import DDP

        # Setup distributed
        setup_distributed(rank, world_size, backend="nccl")
        device = f"cuda:{rank}"
        torch.cuda.set_device(device)

        # Create model and wrap with DDP
        model = create_model().to(device)
        ddp_model = DDP(model)

        # Optimizer
        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)

        # Loss function
        class SequenceCrossEntropyLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.loss_fn = nn.CrossEntropyLoss()

            def forward(self, logits, targets):
                vocab_size = logits.shape[-1]
                return self.loss_fn(
                    logits.view(-1, vocab_size),
                    targets.view(-1),
                )

        loss_fn = SequenceCrossEntropyLoss()

        # Generate synthetic data
        torch.manual_seed(42 + rank)
        seq_len = 512
        num_samples = num_steps * batch_size

        inputs = torch.randint(0, 10000, (num_samples, seq_len))
        targets = torch.randint(0, 10000, (num_samples, seq_len))

        # Profile training steps
        for step in range(num_steps):
            start_idx = step * batch_size
            end_idx = start_idx + batch_size

            batch_inputs = inputs[start_idx:end_idx].to(device)
            batch_targets = targets[start_idx:end_idx].to(device)

            # Mark NVTX range for this iteration
            torch.cuda.nvtx.range_push(f"Training Step {step}")

            optimizer.zero_grad()

            # Forward pass
            torch.cuda.nvtx.range_push("Forward")
            outputs = ddp_model(batch_inputs)
            loss = loss_fn(outputs, batch_targets)
            torch.cuda.nvtx.range_pop()

            # Backward pass (async all-reduce triggered automatically!)
            torch.cuda.nvtx.range_push("Backward (with async all-reduce)")
            loss.backward()
            torch.cuda.nvtx.range_pop()

            # Wait for all communication to finish
            torch.cuda.nvtx.range_push("Wait for All-Reduce")
            ddp_model.finish_gradient_synchronization()
            torch.cuda.nvtx.range_pop()

            # Optimizer step
            torch.cuda.nvtx.range_push("Optimizer Step")
            optimizer.step()
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_pop()  # Training Step

        cleanup_distributed()

    except Exception as e:
        if rank == 0:
            print(f"Error in overlap DDP profiling (rank {rank}): {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Profile DDP implementations with Nsight Systems"
    )

    parser.add_argument(
        "--implementation",
        type=str,
        required=True,
        choices=["naive", "overlap"],
        help="Which implementation to profile",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=3,
        help="Number of training steps to profile (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size per GPU (default: 2)",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
        help="Number of GPUs to use (default: 2)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print(f"Profiling {args.implementation.upper()} DDP with Nsight Systems")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Implementation: {args.implementation}")
    print(f"  World size: {args.world_size} GPUs")
    print(f"  Batch size per GPU: {args.batch_size}")
    print(f"  Profiling steps: {args.num_steps}")
    print()

    # Check GPU availability
    if not torch.cuda.is_available():
        print("✗ Error: CUDA not available")
        return 1

    if torch.cuda.device_count() < args.world_size:
        print(f"✗ Error: Need {args.world_size} GPUs, only {torch.cuda.device_count()} available")
        return 1

    # Select profiling function
    if args.implementation == "naive":
        profile_fn = profile_naive_ddp
    else:
        profile_fn = profile_overlap_ddp

    # Spawn workers
    ctx = mp.get_context("spawn")
    processes = []

    for rank in range(args.world_size):
        p = ctx.Process(
            target=profile_fn,
            args=(
                rank,
                args.world_size,
                args.num_steps,
                args.batch_size,
            ),
        )
        p.start()
        processes.append(p)

    # Wait for completion
    for p in processes:
        p.join()

    print()
    print("=" * 80)
    print("✓ Profiling complete!")
    print("=" * 80)
    print()
    print("To view results:")
    print(f"  Open the generated .nsys-rep file in Nsight Systems GUI")
    print()
    print("What to look for:")
    if args.implementation == "naive":
        print("  - Backward pass completes BEFORE all-reduce starts")
        print("  - All-reduce happens AFTER backward (no overlap)")
    else:
        print("  - NCCL operations start DURING backward pass")
        print("  - Communication overlaps with backward computation")
    print()

    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())