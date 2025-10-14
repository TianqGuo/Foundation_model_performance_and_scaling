#!/usr/bin/env python3
"""
Diagnose Learning Rate Sweep Performance Issues

This script checks:
1. GPU availability and usage
2. Current training progress
3. Actual throughput vs expected
4. Bottleneck identification
"""

import sys
from pathlib import Path
import json
import pandas as pd
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def check_gpu_status():
    """Check GPU availability and current utilization."""
    print("="*80)
    print("GPU STATUS CHECK")
    print("="*80)

    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA Available: {cuda_available}")

    if cuda_available:
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Capability: {torch.cuda.get_device_capability(i)}")

            # Memory info
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3

            print(f"  Memory Allocated: {mem_allocated:.2f} GB")
            print(f"  Memory Reserved: {mem_reserved:.2f} GB")
            print(f"  Memory Total: {mem_total:.2f} GB")
    else:
        print("\n⚠️  WARNING: CUDA is not available!")
        print("Training is running on CPU, which is MUCH slower.")

    print()


def analyze_experiment_progress(exp_dir: Path):
    """Analyze progress of a single experiment."""
    exp_name = exp_dir.name

    # Load summary
    summary_path = exp_dir / "summary.json"
    if not summary_path.exists():
        return None

    with open(summary_path) as f:
        summary = json.load(f)

    # Load metrics
    metrics_path = exp_dir / "metrics.csv"
    if not metrics_path.exists():
        return None

    metrics = pd.read_csv(metrics_path)

    stats = summary["statistics"]
    config = summary["config"]

    # Calculate progress
    current_steps = stats["total_steps"]
    max_steps = config["max_iterations"]
    progress_pct = (current_steps / max_steps) * 100

    # Calculate throughput
    wallclock_hours = stats["total_wallclock_hours"]
    total_tokens = current_steps * config["batch_size"] * config["context_length"]
    tokens_per_sec = total_tokens / (wallclock_hours * 3600)

    # Calculate time per step
    time_per_step = (wallclock_hours * 3600) / current_steps if current_steps > 0 else 0

    # Estimate remaining time
    remaining_steps = max_steps - current_steps
    remaining_hours = (remaining_steps * time_per_step) / 3600

    return {
        "exp_name": exp_name,
        "learning_rate": config["learning_rate"],
        "progress_pct": progress_pct,
        "current_steps": current_steps,
        "max_steps": max_steps,
        "wallclock_hours": wallclock_hours,
        "tokens_per_sec": tokens_per_sec,
        "time_per_step": time_per_step,
        "remaining_hours": remaining_hours,
        "train_loss_final": stats["train_loss"]["final"],
        "val_loss_best": stats["val_loss"]["best"],
    }


def main():
    sweep_dir = Path("cs336_basics/basics/runs/lr_sweep")

    if not sweep_dir.exists():
        print(f"Error: Sweep directory not found: {sweep_dir}")
        return 1

    # Check GPU status
    check_gpu_status()

    # Analyze all experiments
    print("="*80)
    print("EXPERIMENT PROGRESS")
    print("="*80)
    print()

    exp_dirs = [d for d in sweep_dir.iterdir() if d.is_dir() and d.name.startswith("lr_")]

    if not exp_dirs:
        print("No experiments found!")
        return 1

    all_results = []

    for exp_dir in sorted(exp_dirs):
        result = analyze_experiment_progress(exp_dir)
        if result:
            all_results.append(result)

    # Print summary table
    print(f"{'Experiment':<15} {'LR':<10} {'Progress':<10} {'Steps':<15} {'Hours':<8} {'Tok/s':<10} {'Loss':<8}")
    print("-"*90)

    for result in all_results:
        print(
            f"{result['exp_name']:<15} "
            f"{result['learning_rate']:<10.2e} "
            f"{result['progress_pct']:>6.1f}%   "
            f"{result['current_steps']:>6}/{result['max_steps']:<6} "
            f"{result['wallclock_hours']:>6.1f}h  "
            f"{result['tokens_per_sec']:>8.0f}  "
            f"{result['val_loss_best']:>6.3f}"
        )

    print()

    # Identify issues
    print("="*80)
    print("DIAGNOSIS")
    print("="*80)
    print()

    if not torch.cuda.is_available():
        print("❌ CRITICAL: Training is running on CPU!")
        print("   Expected throughput on GPU: 5,000-10,000 tok/s")
        print("   Expected throughput on CPU: 100-500 tok/s")
        print()
        print("   SOLUTION: Run on a machine with CUDA GPU")
        print()

    # Check throughput
    avg_throughput = sum(r["tokens_per_sec"] for r in all_results) / len(all_results) if all_results else 0

    print(f"Average throughput: {avg_throughput:.0f} tokens/sec")
    print()

    if avg_throughput < 1000:
        print("⚠️  WARNING: Very low throughput!")
        print("   This indicates CPU training or a major bottleneck.")
        print()
        if torch.cuda.is_available():
            print("   Possible causes:")
            print("   - Data loading bottleneck (check if using memmap)")
            print("   - Model not moved to GPU (check device in run_experiment.py)")
            print("   - Compilation not working (check torch.compile)")
            print("   - Validation running too frequently")
            print()
    elif avg_throughput < 3000:
        print("⚠️  WARNING: Low throughput!")
        print("   Expected 5,000-10,000 tok/s on H100 GPU.")
        print()
        print("   Possible causes:")
        print("   - Slower GPU (not H100)")
        print("   - Sub-optimal compilation")
        print("   - Data loading overhead")
        print()
    else:
        print("✓ Throughput looks reasonable for GPU training")
        print()

    # Estimate total time
    if all_results:
        # Get most recent experiment
        most_recent = max(all_results, key=lambda x: x["current_steps"])

        print(f"\nMost recent experiment: {most_recent['exp_name']}")
        print(f"  Progress: {most_recent['progress_pct']:.1f}%")
        print(f"  Time elapsed: {most_recent['wallclock_hours']:.1f} hours")
        print(f"  Estimated remaining: {most_recent['remaining_hours']:.1f} hours")
        print()

        # Estimate total sweep time
        total_experiments = 7  # Default number of LRs
        completed_experiments = len(all_results)

        if most_recent['progress_pct'] > 0:
            time_per_experiment = most_recent['wallclock_hours'] * (100 / most_recent['progress_pct'])
            total_sweep_time = time_per_experiment * total_experiments
            elapsed_sweep_time = sum(r["wallclock_hours"] for r in all_results)
            remaining_sweep_time = total_sweep_time - elapsed_sweep_time

            print(f"Total sweep estimate:")
            print(f"  Time per experiment: ~{time_per_experiment:.1f} hours")
            print(f"  Total sweep time: ~{total_sweep_time:.1f} hours")
            print(f"  Elapsed so far: {elapsed_sweep_time:.1f} hours")
            print(f"  Remaining: ~{remaining_sweep_time:.1f} hours")
            print()

            if total_sweep_time > 10:
                print("⚠️  WARNING: Total sweep time is much longer than expected 4 hours!")
                print()

    print("="*80)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
