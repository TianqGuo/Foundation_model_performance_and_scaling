#!/usr/bin/env python3
# ==============================================================================
# Benchmark Naïve DDP Training
# ==============================================================================
#
# DESCRIPTION:
#   Benchmarks naïve DDP training to measure communication overhead.
#   Tests with XL model (d_model=1600, 48 layers) on 2 GPUs.
#
# USAGE:
#   # Run benchmark with default XL model
#   uv run python benchmark_naive_ddp.py
#
#   # Custom model size
#   uv run python benchmark_naive_ddp.py --model-size large --num-steps 20
#
# OUTPUT:
#   - CSV results: ../../results/naive_ddp/benchmark_results.csv
#   - Text summary: Console output with timing breakdown
#
# ==============================================================================

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from naive_ddp_trainer import NaiveDDPTrainer, setup_distributed, cleanup_distributed, shard_batch


# Ensure Assignment 1 package is importable without installation
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CS336_BASICS_DIR = PROJECT_ROOT / "cs336-basics"
if CS336_BASICS_DIR.exists() and str(CS336_BASICS_DIR) not in sys.path:
    sys.path.append(str(CS336_BASICS_DIR))


# Model configurations from Section 1.1.2
MODEL_CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


def create_transformer_model(
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
    vocab_size: int = 10000,
    context_length: int = 512,
):
    """Create a Transformer language model.

    Uses the cs336_basics implementation if available, otherwise creates
    a simple transformer for benchmarking.

    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        vocab_size: Vocabulary size
        context_length: Maximum sequence length

    Returns:
        Transformer model
    """
    try:
        # Try to import from cs336_basics (Assignment 1)
        from cs336_basics.model import TransformerLM

        model = TransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            attn_pdrop=0.0,
            residual_pdrop=0.0,
        )
    except ImportError:
        # Fallback: create simple transformer for benchmarking
        print("Warning: cs336_basics not found, using simple transformer")

        model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=0.0,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        # Wrap with embedding and output layers
        class SimpleTransformerLM(nn.Module):
            def __init__(self, vocab_size, d_model, encoder):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.encoder = encoder
                self.output = nn.Linear(d_model, vocab_size)

            def forward(self, x):
                x = self.embedding(x)
                x = self.encoder(x)
                x = self.output(x)
                return x

        model = SimpleTransformerLM(vocab_size, d_model, model)

    return model


def benchmark_ddp_worker(
    rank: int,
    world_size: int,
    model_config: Dict[str, int],
    batch_size: int,
    context_length: int,
    num_warmup: int,
    num_steps: int,
    results_queue: mp.Queue,
):
    """Worker process for benchmarking DDP.

    Args:
        rank: Global rank
        world_size: Total number of processes
        model_config: Model configuration
        batch_size: Global batch size
        context_length: Sequence length
        num_warmup: Number of warmup steps
        num_steps: Number of measured steps
        results_queue: Queue to send results
    """
    try:
        # Setup distributed
        setup_distributed(rank, world_size, backend="nccl")

        device = f"cuda:{rank}"
        torch.cuda.set_device(rank)

        # Create model
        model = create_transformer_model(
            d_model=model_config["d_model"],
            d_ff=model_config["d_ff"],
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            vocab_size=10000,
            context_length=context_length,
        )

        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Create trainer
        trainer = NaiveDDPTrainer(model, optimizer, rank, world_size, device=device)

        # Loss function (flatten sequence tokens for cross-entropy)
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

        # Generate random data (same on all ranks for reproducibility)
        torch.manual_seed(42)
        global_inputs = torch.randint(0, 10000, (batch_size, context_length))
        global_targets = torch.randint(0, 10000, (batch_size, context_length))

        if rank == 0:
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"Model size: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9:.2f} GB")
            print()

        # Warmup
        if rank == 0:
            print(f"Running {num_warmup} warmup steps...")

        for step in range(num_warmup):
            # Shard batch
            local_inputs, local_targets = shard_batch(
                (global_inputs, global_targets),
                rank,
                world_size,
            )

            # Training step
            trainer.train_step(local_inputs, local_targets, loss_fn)

        if rank == 0:
            print("✓ Warmup complete")
            print()

        # Benchmark
        if rank == 0:
            print(f"Running {num_steps} measured steps...")

        step_results = []

        for step in range(num_steps):
            # Shard batch
            local_inputs, local_targets = shard_batch(
                (global_inputs, global_targets),
                rank,
                world_size,
            )

            # Training step
            step_info = trainer.train_step(local_inputs, local_targets, loss_fn)
            step_results.append(step_info)

        # Gather results from all ranks to rank 0
        all_results = [None] * world_size
        dist.all_gather_object(all_results, step_results)

        # Only rank 0 reports
        if rank == 0:
            # Average across ranks and steps
            avg_results = {
                "total_time": 0.0,
                "compute_time": 0.0,
                "comm_time": 0.0,
            }

            for rank_results in all_results:
                for step_info in rank_results:
                    avg_results["total_time"] += step_info["total_time"]
                    avg_results["compute_time"] += step_info["compute_time"]
                    avg_results["comm_time"] += step_info["comm_time"]

            # Average
            total_samples = world_size * num_steps
            for key in avg_results:
                avg_results[key] /= total_samples

            # Calculate communication fraction
            avg_results["comm_fraction"] = (
                avg_results["comm_time"] / avg_results["total_time"]
                if avg_results["total_time"] > 0 else 0
            )

            # Get communication stats
            comm_stats = trainer.get_communication_stats()

            results_queue.put({
                "avg_results": avg_results,
                "comm_stats": comm_stats,
                "num_parameters": sum(p.numel() for p in model.parameters()),
            })

        cleanup_distributed()

    except Exception as e:
        if rank == 0:
            print(f"Error in benchmark worker (rank {rank}): {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark naïve DDP training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model-size",
        type=str,
        default="xl",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model size to benchmark (default: xl)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Global batch size (default: 4)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
        help="Sequence length (default: 512)",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=3,
        help="Number of warmup steps (default: 3)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Number of measured steps (default: 10)",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
        help="Number of GPUs (default: 2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../../results/naive_ddp/benchmark_results.csv",
        help="Output CSV file",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Naïve DDP Training Benchmark")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Model size: {args.model_size}")
    print(f"  Model config: {MODEL_CONFIGS[args.model_size]}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Context length: {args.context_length}")
    print(f"  World size: {args.world_size}")
    print(f"  Warmup steps: {args.num_warmup}")
    print(f"  Measured steps: {args.num_steps}")
    print()

    # Check GPU availability
    if not torch.cuda.is_available():
        print("✗ Error: CUDA not available")
        return 1

    if torch.cuda.device_count() < args.world_size:
        print(f"✗ Error: Need {args.world_size} GPUs, found {torch.cuda.device_count()}")
        return 1

    # Get model config
    model_config = MODEL_CONFIGS[args.model_size]

    # Spawn workers
    ctx = mp.get_context("spawn")
    results_queue = ctx.Queue()

    processes = []
    for rank in range(args.world_size):
        p = ctx.Process(
            target=benchmark_ddp_worker,
            args=(
                rank,
                args.world_size,
                model_config,
                args.batch_size,
                args.context_length,
                args.num_warmup,
                args.num_steps,
                results_queue,
            ),
        )
        p.start()
        processes.append(p)

    # Wait for completion
    for p in processes:
        p.join()

    # Get results
    if not results_queue.empty():
        results = results_queue.get()

        print("=" * 80)
        print("Benchmark Results")
        print("=" * 80)
        print()

        avg = results["avg_results"]
        comm_stats = results["comm_stats"]

        print(f"Timing per training step:")
        print(f"  Total time:      {avg['total_time']*1000:.2f} ms")
        print(f"  Compute time:    {avg['compute_time']*1000:.2f} ms ({avg['compute_time']/avg['total_time']*100:.1f}%)")
        print(f"  Communication:   {avg['comm_time']*1000:.2f} ms ({avg['comm_fraction']*100:.1f}%)")
        print()

        print(f"Communication statistics:")
        print(f"  Total comm ops:  {comm_stats['num_comm_ops']}")
        print(f"  Avg comm time:   {comm_stats['avg_comm_time']*1000:.3f} ms per operation")
        print()

        print(f"Model:")
        print(f"  Parameters:      {results['num_parameters']:,}")
        print()

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame([{
            "model_size": args.model_size,
            "d_model": model_config["d_model"],
            "num_layers": model_config["num_layers"],
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "world_size": args.world_size,
            "num_parameters": results["num_parameters"],
            "total_time_ms": avg["total_time"] * 1000,
            "compute_time_ms": avg["compute_time"] * 1000,
            "comm_time_ms": avg["comm_time"] * 1000,
            "comm_fraction": avg["comm_fraction"],
            "num_comm_ops": comm_stats["num_comm_ops"],
        }])

        df.to_csv(args.output, index=False)
        print(f"✓ Results saved to {args.output}")
        print()

        print("=" * 80)
        print("Analysis")
        print("=" * 80)
        print(f"The naïve DDP implementation spends {avg['comm_fraction']*100:.1f}% of time on")
        print(f"gradient communication. This high overhead is due to all-reducing each")
        print(f"parameter gradient individually ({comm_stats['num_comm_ops']} separate operations),")
        print(f"causing high communication latency and preventing overlap with computation.")
        print()
        print(f"Optimized DDP implementations bucket gradients to reduce communication overhead.")
        print()

        return 0
    else:
        print("✗ Error: No results from benchmark")
        return 1


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
