"""
Training speed benchmarking script for optimizer state sharding (Part b).

This script measures time per iteration with and without optimizer state sharding.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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


def benchmark_speed(rank, world_size, args):
    """Run speed benchmarking on a single rank."""
    # Setup distributed
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.master_port

    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    # Results storage
    results = {}

    # ============================================================
    # Benchmark WITHOUT sharding
    # ============================================================
    if rank == 0:
        print("\n" + "="*80)
        print(f"Benchmarking WITHOUT optimizer state sharding")
        print("="*80)

    torch.manual_seed(42)
    model = create_model(args.model_size).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Warmup
    if rank == 0:
        print(f"Warming up for {args.warmup_steps} steps...")

    for _ in range(args.warmup_steps):
        dummy_input = torch.randint(0, 10000, (args.batch_size, 512)).to(device)
        dummy_target = torch.randint(0, 10000, (args.batch_size, 512)).to(device)

        optimizer.zero_grad()
        logits = model(dummy_input)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, 10000),
            dummy_target.view(-1)
        )
        loss.backward()
        optimizer.step()

    # Benchmark
    if rank == 0:
        print(f"Benchmarking for {args.num_steps} steps...")

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()

    for _ in range(args.num_steps):
        dummy_input = torch.randint(0, 10000, (args.batch_size, 512)).to(device)
        dummy_target = torch.randint(0, 10000, (args.batch_size, 512)).to(device)

        optimizer.zero_grad()
        logits = model(dummy_input)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, 10000),
            dummy_target.view(-1)
        )
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time_per_step = (total_time / args.num_steps) * 1000  # Convert to ms

    results['non_sharded'] = {
        'total_time_sec': total_time,
        'avg_time_per_step_ms': avg_time_per_step
    }

    if rank == 0:
        print(f"Non-sharded: {avg_time_per_step:.2f} ms per step")

    # Clean up
    del model, optimizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ============================================================
    # Benchmark WITH sharding
    # ============================================================
    if rank == 0:
        print("\n" + "="*80)
        print(f"Benchmarking WITH optimizer state sharding")
        print("="*80)

    torch.manual_seed(42)
    model = create_model(args.model_size).to(device)

    from cs336_systems.optimizer_sharding.optimizer import ShardedOptimizer
    optimizer_sharded = ShardedOptimizer(
        model.parameters(),
        torch.optim.AdamW,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Warmup
    if rank == 0:
        print(f"Warming up for {args.warmup_steps} steps...")

    for _ in range(args.warmup_steps):
        dummy_input = torch.randint(0, 10000, (args.batch_size, 512)).to(device)
        dummy_target = torch.randint(0, 10000, (args.batch_size, 512)).to(device)

        optimizer_sharded.zero_grad()
        logits = model(dummy_input)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, 10000),
            dummy_target.view(-1)
        )
        loss.backward()
        optimizer_sharded.step()

    # Benchmark
    if rank == 0:
        print(f"Benchmarking for {args.num_steps} steps...")

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()

    for _ in range(args.num_steps):
        dummy_input = torch.randint(0, 10000, (args.batch_size, 512)).to(device)
        dummy_target = torch.randint(0, 10000, (args.batch_size, 512)).to(device)

        optimizer_sharded.zero_grad()
        logits = model(dummy_input)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, 10000),
            dummy_target.view(-1)
        )
        loss.backward()
        optimizer_sharded.step()

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time_per_step_sharded = (total_time / args.num_steps) * 1000  # Convert to ms

    results['sharded'] = {
        'total_time_sec': total_time,
        'avg_time_per_step_ms': avg_time_per_step_sharded
    }

    if rank == 0:
        print(f"Sharded: {avg_time_per_step_sharded:.2f} ms per step")

    # Print summary from rank 0
    if rank == 0:
        print_results(results, args)

    dist.destroy_process_group()


def print_results(results, args):
    """Print formatted results."""
    print("\n" + "="*80)
    print("SPEED BENCHMARKING RESULTS")
    print("="*80)
    print(f"Configuration: {args.model_size} model, world_size={args.world_size}, batch_size={args.batch_size}")
    print()

    non_sharded_time = results['non_sharded']['avg_time_per_step_ms']
    sharded_time = results['sharded']['avg_time_per_step_ms']

    overhead = sharded_time - non_sharded_time
    overhead_pct = (overhead / non_sharded_time) * 100

    print("-" * 80)
    print("Time per Iteration:")
    print("-" * 80)
    print(f"Without sharding: {non_sharded_time:.2f} ms")
    print(f"With sharding:    {sharded_time:.2f} ms")
    print(f"Overhead:         {overhead:.2f} ms ({overhead_pct:+.2f}%)")

    # Save to CSV
    output_dir = Path(__file__).resolve().parents[2] / "results" / "optimizer_sharding"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "speed_comparison.csv"
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'model_size', 'd_model', 'num_layers', 'world_size', 'batch_size',
            'non_sharded_ms', 'sharded_ms', 'overhead_ms', 'overhead_pct'
        ])

        config = MODEL_CONFIGS[args.model_size]
        writer.writerow([
            args.model_size,
            config["d_model"],
            config["num_layers"],
            args.world_size,
            args.batch_size,
            f"{non_sharded_time:.2f}",
            f"{sharded_time:.2f}",
            f"{overhead:.2f}",
            f"{overhead_pct:+.2f}"
        ])

    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark training speed with/without optimizer sharding")
    parser.add_argument('--model-size', type=str, default='xl', choices=['large', 'xl'],
                       help='Model size')
    parser.add_argument('--world-size', type=int, default=2,
                       help='Number of GPUs')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Batch size per GPU')
    parser.add_argument('--num-steps', type=int, default=10,
                       help='Number of steps to benchmark')
    parser.add_argument('--warmup-steps', type=int, default=5,
                       help='Number of warmup steps')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--master-port', type=str, default='29501',
                       help='Master port for distributed training')

    args = parser.parse_args()

    # Run multi-process
    mp.spawn(
        benchmark_speed,
        args=(args.world_size, args),
        nprocs=args.world_size,
        join=True
    )


if __name__ == '__main__':
    main()
