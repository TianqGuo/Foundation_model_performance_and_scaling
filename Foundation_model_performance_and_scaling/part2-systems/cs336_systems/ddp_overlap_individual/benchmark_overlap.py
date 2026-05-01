#!/usr/bin/env python3
# ==============================================================================
# Benchmark DDP with Overlapping Computation and Communication
# ==============================================================================
#
# DESCRIPTION:
#   Compares performance of three DDP implementations:
#   1. Naive DDP (individual parameter all-reduces, no overlap)
#   2. Flat DDP (single batched all-reduce, no overlap)
#   3. Overlap DDP (individual parameter async all-reduces, WITH overlap)
#
# USAGE:
#   uv run python benchmark_overlap.py
#   uv run python benchmark_overlap.py --model-size large
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
from ddp_overlap_individual.ddp import DDP


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
    num_steps: int,
    batch_size: int,
    warmup_steps: int,
    results_queue: mp.Queue,
):
    """Worker process for benchmarking overlap DDP."""
    try:
        # Setup distributed
        setup_distributed(rank, world_size, backend="nccl")
        device = f"cuda:{rank}"
        torch.cuda.set_device(device)

        # Create model and wrap with DDP
        model = create_model(model_size).to(device)
        ddp_model = DDP(model)

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
) -> dict:
    """Run benchmark for overlap DDP."""
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
        raise RuntimeError("No results from overlap benchmark")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DDP with overlapping computation and communication"
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
    print("DDP with Overlapping Computation and Communication Benchmark")
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

    # Run benchmark
    print("Running benchmark...")
    print("-" * 80)

    print(f"Benchmarking overlap DDP (individual params, async all-reduce)...")
    overlap_results = run_benchmark(
        args.model_size,
        args.num_steps,
        args.batch_size,
        args.warmup_steps,
        args.world_size,
    )
    print(f"✓ Overlap DDP complete")
    print()

    # Display results
    print("=" * 80)
    print("Results")
    print("=" * 80)
    print()

    overlap_time = overlap_results['avg_step_time']
    print(f"Overlap DDP: {overlap_time * 1000:.2f} ms per step")
    print()

    # Load previous results for comparison if they exist
    results_dir = Path(__file__).parent.parent.parent / "results"
    flat_csv = results_dir / "flat_ddp" / "comparison_results.csv"

    if flat_csv.exists():
        import pandas as pd
        df = pd.read_csv(flat_csv)

        naive_time = float(df[df['implementation'] == 'naive']['avg_step_time_ms'].iloc[0]) / 1000
        flat_time = float(df[df['implementation'] == 'flat']['avg_step_time_ms'].iloc[0]) / 1000

        print("Comparison with previous implementations:")
        print("-" * 80)
        print(f"{'Implementation':<30} {'Avg Time/Iter':<15} {'Speedup':<15}")
        print("-" * 80)
        print(f"{'Naive DDP (no overlap)':<30} {naive_time * 1000:>13.2f} ms {'1.00x':>15}")
        print(f"{'Flat DDP (no overlap)':<30} {flat_time * 1000:>13.2f} ms {naive_time/flat_time:>14.2f}x")
        print(f"{'Overlap DDP (with overlap)':<30} {overlap_time * 1000:>13.2f} ms {naive_time/overlap_time:>14.2f}x")
        print()

        # Save combined results
        output_dir = results_dir / "ddp_overlap_individual"
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "benchmark_results.csv"

        model_config = MODEL_CONFIGS[args.model_size]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "implementation",
                "model_size",
                "d_model",
                "num_layers",
                "world_size",
                "avg_step_time_ms",
                "speedup_vs_naive",
            ])
            writer.writeheader()

            writer.writerow({
                "implementation": "naive",
                "model_size": args.model_size,
                "d_model": model_config["d_model"],
                "num_layers": model_config["num_layers"],
                "world_size": args.world_size,
                "avg_step_time_ms": f"{naive_time * 1000:.2f}",
                "speedup_vs_naive": "1.00",
            })

            writer.writerow({
                "implementation": "flat",
                "model_size": args.model_size,
                "d_model": model_config["d_model"],
                "num_layers": model_config["num_layers"],
                "world_size": args.world_size,
                "avg_step_time_ms": f"{flat_time * 1000:.2f}",
                "speedup_vs_naive": f"{naive_time/flat_time:.2f}",
            })

            writer.writerow({
                "implementation": "overlap_individual",
                "model_size": args.model_size,
                "d_model": model_config["d_model"],
                "num_layers": model_config["num_layers"],
                "world_size": args.world_size,
                "avg_step_time_ms": f"{overlap_time * 1000:.2f}",
                "speedup_vs_naive": f"{naive_time/overlap_time:.2f}",
            })

        print("=" * 80)
        print(f"✓ Results saved to: {csv_path}")
        print("=" * 80)

    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())