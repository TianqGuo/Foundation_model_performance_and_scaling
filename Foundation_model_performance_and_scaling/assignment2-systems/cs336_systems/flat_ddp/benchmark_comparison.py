#!/usr/bin/env python3
# ==============================================================================
# DDP Implementation Comparison Benchmark
# ==============================================================================
#
# DESCRIPTION:
#   Compares performance of naive DDP (individual parameter all-reduces) vs.
#   flat DDP (single batched all-reduce) to measure the impact of batching
#   communication calls.
#
# USAGE:
#   # Run comparison with default settings (XL model, 2 GPUs)
#   uv run python benchmark_comparison.py
#
#   # Custom configuration
#   uv run python benchmark_comparison.py --model-size large --num-steps 20
#
# OUTPUT:
#   Comparison table showing time per iteration and communication overhead
#   for both implementations
#
# ==============================================================================

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

# Import from naive_ddp module
import sys
from pathlib import Path
# Add parent directory to path to import from sibling modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from naive_ddp.naive_ddp_trainer import NaiveDDPTrainer, setup_distributed, cleanup_distributed
from flat_ddp.flat_ddp_trainer import FlatDDPTrainer


# Model size configurations (from §1.1.2)
MODEL_CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
}


def create_model(model_size: str):
    """Create a Transformer model for benchmarking.

    Args:
        model_size: Model size (small/medium/large/xl)

    Returns:
        Transformer model instance
    """
    config = MODEL_CONFIGS[model_size]

    # Try to import from cs336_basics
    try:
        from cs336_basics.transformer_training.model import TransformerLM

        model = TransformerLM(
            vocab_size=10000,
            context_length=512,
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            d_ff=config["d_ff"],
        )
        return model

    except Exception as exc:
        # Fallback to simple model
        print(f"Warning: cs336_basics import failed ({exc}), using fallback model")

        class FallbackTransformer(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.embed = nn.Embedding(10000, config["d_model"])
                self.layers = nn.ModuleList([
                    nn.TransformerEncoderLayer(
                        d_model=config["d_model"],
                        nhead=config["num_heads"],
                        dim_feedforward=config["d_ff"],
                        batch_first=True,
                    )
                    for _ in range(config["num_layers"])
                ])
                self.head = nn.Linear(config["d_model"], 10000)

            def forward(self, x):
                x = self.embed(x)
                for layer in self.layers:
                    x = layer(x)
                return self.head(x[:, -1, :])

        return FallbackTransformer(config)


def benchmark_worker(
    rank: int,
    world_size: int,
    model_size: str,
    num_steps: int,
    batch_size: int,
    warmup_steps: int,
    implementation: str,
    results_queue: mp.Queue,
):
    """Worker process for benchmarking a DDP implementation.

    Args:
        rank: Global rank
        world_size: Total number of processes
        model_size: Model size to benchmark
        num_steps: Number of training steps to measure
        batch_size: Batch size per GPU
        warmup_steps: Number of warm-up steps
        implementation: "naive" or "flat"
        results_queue: Queue to send results back
    """
    try:
        # Setup distributed
        setup_distributed(rank, world_size, backend="nccl")
        device = f"cuda:{rank}"
        torch.cuda.set_device(device)

        # Create model
        model = create_model(model_size)

        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Create trainer based on implementation type
        if implementation == "naive":
            trainer = NaiveDDPTrainer(model, optimizer, rank, world_size, device=device)
        elif implementation == "flat":
            trainer = FlatDDPTrainer(model, optimizer, rank, world_size, device=device)
        else:
            raise ValueError(f"Unknown implementation: {implementation}")

        # Generate synthetic data
        torch.manual_seed(42 + rank)
        seq_len = 512
        num_samples = (warmup_steps + num_steps) * batch_size

        inputs = torch.randint(0, 10000, (num_samples, seq_len))
        targets = torch.randint(0, 10000, (num_samples,))

        # Warm-up steps
        for step in range(warmup_steps):
            start_idx = step * batch_size
            end_idx = start_idx + batch_size

            batch_inputs = inputs[start_idx:end_idx]
            batch_targets = targets[start_idx:end_idx]

            trainer.train_step(batch_inputs, batch_targets)

        # Measured steps
        step_times = []
        comm_times = []

        for step in range(num_steps):
            start_idx = (warmup_steps + step) * batch_size
            end_idx = start_idx + batch_size

            batch_inputs = inputs[start_idx:end_idx]
            batch_targets = targets[start_idx:end_idx]

            step_info = trainer.train_step(batch_inputs, batch_targets)

            step_times.append(step_info["total_time"])
            comm_times.append(step_info["comm_time"])

        # Gather statistics
        stats = trainer.get_communication_stats()

        # Synchronize before reporting
        dist.barrier()

        # Only rank 0 reports results
        if rank == 0:
            results_queue.put({
                "implementation": implementation,
                "avg_step_time": sum(step_times) / len(step_times),
                "avg_comm_time": sum(comm_times) / len(comm_times),
                "num_comm_ops": stats["num_comm_ops"],
                "total_comm_time": stats["total_comm_time"],
            })

        cleanup_distributed()

    except Exception as e:
        if rank == 0:
            print(f"Error in benchmark worker (rank {rank}): {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()


def run_benchmark(
    model_size: str,
    num_steps: int,
    batch_size: int,
    warmup_steps: int,
    world_size: int,
    implementation: str,
) -> dict:
    """Run benchmark for a specific implementation.

    Args:
        model_size: Model size to benchmark
        num_steps: Number of training steps to measure
        batch_size: Batch size per GPU
        warmup_steps: Number of warm-up steps
        world_size: Number of GPUs to use
        implementation: "naive" or "flat"

    Returns:
        Dictionary with benchmark results
    """
    # Spawn workers
    ctx = mp.get_context("spawn")
    results_queue = ctx.Queue()

    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=benchmark_worker,
            args=(
                rank,
                world_size,
                model_size,
                num_steps,
                batch_size,
                warmup_steps,
                implementation,
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
        return results_queue.get()
    else:
        raise RuntimeError(f"No results from {implementation} benchmark")


def main():
    parser = argparse.ArgumentParser(
        description="Compare naive vs. flat DDP implementations"
    )

    parser.add_argument(
        "--model-size",
        type=str,
        default="xl",
        choices=["small", "medium", "large", "xl"],
        help="Model size to benchmark (default: xl)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Number of measured training steps (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size per GPU (default: 2)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Number of warm-up steps (default: 5)",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
        help="Number of GPUs to use (default: 2)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("DDP Implementation Comparison Benchmark")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Model size: {args.model_size}")
    print(f"  World size: {args.world_size} GPUs")
    print(f"  Batch size per GPU: {args.batch_size}")
    print(f"  Warm-up steps: {args.warmup_steps}")
    print(f"  Measured steps: {args.num_steps}")
    print()

    # Check GPU availability
    if not torch.cuda.is_available():
        print("✗ Error: CUDA not available")
        return 1

    if torch.cuda.device_count() < args.world_size:
        print(f"✗ Error: Need {args.world_size} GPUs, only {torch.cuda.device_count()} available")
        return 1

    # Run benchmarks
    print("Running benchmarks...")
    print("-" * 80)

    # Benchmark naive implementation
    print(f"Benchmarking naive DDP (individual parameter all-reduces)...")
    naive_results = run_benchmark(
        args.model_size,
        args.num_steps,
        args.batch_size,
        args.warmup_steps,
        args.world_size,
        "naive",
    )
    print(f"✓ Naive DDP complete")

    # Benchmark flat implementation
    print(f"Benchmarking flat DDP (single batched all-reduce)...")
    flat_results = run_benchmark(
        args.model_size,
        args.num_steps,
        args.batch_size,
        args.warmup_steps,
        args.world_size,
        "flat",
    )
    print(f"✓ Flat DDP complete")
    print()

    # Display results
    print("=" * 80)
    print("Results")
    print("=" * 80)
    print()

    print(f"{'Implementation':<20} {'Avg Time/Iter':<15} {'Avg Comm Time':<15} {'Num Comm Ops':<15}")
    print("-" * 80)

    print(
        f"{'Naive DDP':<20} "
        f"{naive_results['avg_step_time']*1000:>13.2f} ms "
        f"{naive_results['avg_comm_time']*1000:>13.2f} ms "
        f"{naive_results['num_comm_ops']:>15}"
    )

    print(
        f"{'Flat DDP':<20} "
        f"{flat_results['avg_step_time']*1000:>13.2f} ms "
        f"{flat_results['avg_comm_time']*1000:>13.2f} ms "
        f"{flat_results['num_comm_ops']:>15}"
    )

    print()
    print("-" * 80)
    print()

    # Calculate speedup
    naive_time = naive_results['avg_step_time']
    flat_time = flat_results['avg_step_time']
    speedup = naive_time / flat_time

    naive_comm = naive_results['avg_comm_time']
    flat_comm = flat_results['avg_comm_time']
    comm_speedup = naive_comm / flat_comm

    print(f"Speedup (overall): {speedup:.2f}x")
    print(f"Speedup (communication): {comm_speedup:.2f}x")
    print()

    # Analysis
    print("Analysis:")
    print("-" * 80)

    naive_comm_pct = (naive_comm / naive_time) * 100
    flat_comm_pct = (flat_comm / flat_time) * 100

    print(f"Naive DDP: Communication overhead = {naive_comm_pct:.1f}% of total time")
    print(f"           Number of all-reduce calls = {naive_results['num_comm_ops']}")
    print()
    print(f"Flat DDP:  Communication overhead = {flat_comm_pct:.1f}% of total time")
    print(f"           Number of all-reduce calls = {flat_results['num_comm_ops']}")
    print()

    if speedup > 1.0:
        print(f"Batching gradients provides {speedup:.2f}x speedup by reducing from")
        print(f"{naive_results['num_comm_ops']} communication calls to {flat_results['num_comm_ops']} call(s).")
    else:
        print(f"Batching shows minimal benefit (speedup: {speedup:.2f}x), likely due to")
        print(f"efficient pipelining in NCCL or small model size where overhead is negligible.")
    print()

    # Save results to CSV
    results_dir = Path(__file__).parent.parent.parent / "results" / "flat_ddp"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "comparison_results.csv"

    model_config = MODEL_CONFIGS[args.model_size]

    # Write CSV with both naive and flat results
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "implementation",
            "model_size",
            "d_model",
            "num_layers",
            "world_size",
            "avg_step_time_ms",
            "avg_comm_time_ms",
            "comm_overhead_pct",
            "num_comm_ops",
            "speedup_vs_naive",
            "comm_speedup_vs_naive",
        ])
        writer.writeheader()

        # Write naive results
        writer.writerow({
            "implementation": "naive",
            "model_size": args.model_size,
            "d_model": model_config["d_model"],
            "num_layers": model_config["num_layers"],
            "world_size": args.world_size,
            "avg_step_time_ms": f"{naive_time * 1000:.2f}",
            "avg_comm_time_ms": f"{naive_comm * 1000:.2f}",
            "comm_overhead_pct": f"{naive_comm_pct:.1f}",
            "num_comm_ops": naive_results["num_comm_ops"],
            "speedup_vs_naive": "1.00",
            "comm_speedup_vs_naive": "1.00",
        })

        # Write flat results
        writer.writerow({
            "implementation": "flat",
            "model_size": args.model_size,
            "d_model": model_config["d_model"],
            "num_layers": model_config["num_layers"],
            "world_size": args.world_size,
            "avg_step_time_ms": f"{flat_time * 1000:.2f}",
            "avg_comm_time_ms": f"{flat_comm * 1000:.2f}",
            "comm_overhead_pct": f"{flat_comm_pct:.1f}",
            "num_comm_ops": flat_results["num_comm_ops"],
            "speedup_vs_naive": f"{speedup:.2f}",
            "comm_speedup_vs_naive": f"{comm_speedup:.2f}",
        })

    print("=" * 80)
    print(f"✓ Results saved to: {csv_path}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())