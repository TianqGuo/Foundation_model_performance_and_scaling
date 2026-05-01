#!/usr/bin/env python3
# ==============================================================================
# Benchmark DDP with Bucketed Gradients
# ==============================================================================
#
# DESCRIPTION:
#   Benchmarks bucketed DDP with various bucket sizes and compares with
#   previous DDP implementations.
#
# USAGE:
#   uv run python benchmark_bucketed.py
#   uv run python benchmark_bucketed.py --bucket-sizes 1 10 100
#
# OUTPUT:
#   Comparison table and CSV results
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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from naive_ddp.naive_ddp_trainer import setup_distributed, cleanup_distributed
from ddp_bucketed.ddp import DDP


# Model size configurations (from §1.1.2)
MODEL_CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
}


def create_model(model_size: str):
    """Create a Transformer model for benchmarking."""
    config = MODEL_CONFIGS[model_size]

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
                return self.head(x)

        return FallbackTransformer(config)


def benchmark_worker(
    rank: int,
    world_size: int,
    model_size: str,
    bucket_size_mb: float,
    num_steps: int,
    batch_size: int,
    warmup_steps: int,
    results_queue: mp.Queue,
):
    """Worker process for benchmarking bucketed DDP."""
    try:
        # Setup distributed
        setup_distributed(rank, world_size, backend="nccl")
        device = f"cuda:{rank}"
        torch.cuda.set_device(device)

        # Create model and wrap with DDP
        model = create_model(model_size).to(device)
        ddp_model = DDP(model, bucket_size_mb=bucket_size_mb)

        # Create optimizer
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
        num_samples = (warmup_steps + num_steps) * batch_size

        inputs = torch.randint(0, 10000, (num_samples, seq_len))
        targets = torch.randint(0, 10000, (num_samples, seq_len))

        # Warm-up steps
        for step in range(warmup_steps):
            start_idx = step * batch_size
            end_idx = start_idx + batch_size

            batch_inputs = inputs[start_idx:end_idx].to(device)
            batch_targets = targets[start_idx:end_idx].to(device)

            optimizer.zero_grad()
            outputs = ddp_model(batch_inputs)
            loss = loss_fn(outputs, batch_targets)
            loss.backward()
            ddp_model.finish_gradient_synchronization()
            optimizer.step()

        # Measured steps
        import time
        step_times = []

        for step in range(num_steps):
            start_idx = (warmup_steps + step) * batch_size
            end_idx = start_idx + batch_size

            batch_inputs = inputs[start_idx:end_idx].to(device)
            batch_targets = targets[start_idx:end_idx].to(device)

            torch.cuda.synchronize()
            step_start = time.perf_counter()

            optimizer.zero_grad()
            outputs = ddp_model(batch_inputs)
            loss = loss_fn(outputs, batch_targets)
            loss.backward()
            ddp_model.finish_gradient_synchronization()
            optimizer.step()

            torch.cuda.synchronize()
            step_time = time.perf_counter() - step_start
            step_times.append(step_time)

        # Synchronize before reporting
        dist.barrier()

        # Only rank 0 reports results
        if rank == 0:
            results_queue.put({
                "avg_step_time": sum(step_times) / len(step_times),
                "num_buckets": len(ddp_model.buckets),
            })

        cleanup_distributed()

    except Exception as e:
        if rank == 0:
            print(f"Error in benchmark worker (rank {rank}): {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()


def run_benchmark(
    model_size: str,
    bucket_size_mb: float,
    num_steps: int,
    batch_size: int,
    warmup_steps: int,
    world_size: int,
) -> dict:
    """Run benchmark for bucketed DDP with given bucket size."""
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
                bucket_size_mb,
                num_steps,
                batch_size,
                warmup_steps,
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
        raise RuntimeError("No results from bucketed DDP benchmark")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark bucketed DDP with various bucket sizes"
    )

    parser.add_argument(
        "--model-size",
        type=str,
        default="xl",
        choices=["small", "medium", "large", "xl"],
        help="Model size to benchmark (default: xl)",
    )
    parser.add_argument(
        "--bucket-sizes",
        type=float,
        nargs="+",
        default=[1.0, 10.0, 100.0, 1000.0],
        help="Bucket sizes in MB to test (default: 1 10 100 1000)",
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
    print("Bucketed DDP Benchmark")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Model size: {args.model_size}")
    print(f"  World size: {args.world_size} GPUs")
    print(f"  Batch size per GPU: {args.batch_size}")
    print(f"  Warm-up steps: {args.warmup_steps}")
    print(f"  Measured steps: {args.num_steps}")
    print(f"  Bucket sizes: {args.bucket_sizes} MB")
    print()

    # Check GPU availability
    if not torch.cuda.is_available():
        print("✗ Error: CUDA not available")
        return 1

    if torch.cuda.device_count() < args.world_size:
        print(f"✗ Error: Need {args.world_size} GPUs, only {torch.cuda.device_count()} available")
        return 1

    # Benchmark each bucket size
    results = []

    for bucket_size_mb in args.bucket_sizes:
        print(f"Benchmarking with bucket size: {bucket_size_mb} MB...")
        result = run_benchmark(
            args.model_size,
            bucket_size_mb,
            args.num_steps,
            args.batch_size,
            args.warmup_steps,
            args.world_size,
        )
        result["bucket_size_mb"] = bucket_size_mb
        results.append(result)
        print(f"  Time: {result['avg_step_time'] * 1000:.2f} ms, Buckets: {result['num_buckets']}")

    # Display results table
    print()
    print("=" * 80)
    print("Results")
    print("=" * 80)
    print()
    print(f"{'Bucket Size (MB)':<20} {'Num Buckets':<15} {'Time/Iter (ms)':<20}")
    print("-" * 80)
    for r in results:
        print(f"{r['bucket_size_mb']:<20.1f} {r['num_buckets']:<15} {r['avg_step_time'] * 1000:<20.2f}")

    # Save results
    output_dir = Path(__file__).parent.parent.parent / "results" / "ddp_bucketed"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "bucket_size_comparison.csv"

    model_config = MODEL_CONFIGS[args.model_size]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "bucket_size_mb",
            "num_buckets",
            "model_size",
            "d_model",
            "num_layers",
            "world_size",
            "avg_step_time_ms",
        ])
        writer.writeheader()

        for r in results:
            writer.writerow({
                "bucket_size_mb": r["bucket_size_mb"],
                "num_buckets": r["num_buckets"],
                "model_size": args.model_size,
                "d_model": model_config["d_model"],
                "num_layers": model_config["num_layers"],
                "world_size": args.world_size,
                "avg_step_time_ms": f"{r['avg_step_time'] * 1000:.2f}",
            })

    print()
    print("=" * 80)
    print(f"✓ Results saved to: {csv_path}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())