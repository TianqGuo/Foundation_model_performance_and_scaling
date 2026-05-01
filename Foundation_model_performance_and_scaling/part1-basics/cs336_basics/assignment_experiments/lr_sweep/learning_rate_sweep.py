#!/usr/bin/env python3
"""
Learning Rate Sweep Experiment

Problem (learning_rate): Tune the learning rate (3 points)

This script runs a hyperparameter sweep over learning rates to find the optimal value.
Target: Achieve validation loss of at most 1.45 on TinyStories.

Hyperparameters (from assignment):
- vocab_size: 10000
- context_length: 256
- d_model: 512
- d_ff: 1344 (roughly 8/3 * d_model, multiple of 64)
- num_layers: 4
- num_heads: 16
- RoPE theta: 10000
- Total tokens: ~327,680,000 (batch_size × steps × context_length)
"""

import sys
from pathlib import Path
import json
import argparse

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cs336_basics.basics import create_experiment_config
from cs336_basics.basics.run_experiment import run_experiment


def calculate_training_steps(
    total_tokens: int = 327_680_000,
    batch_size: int = 128,
    context_length: int = 256,
) -> int:
    """
    Calculate number of training steps to process target total tokens.

    Formula: batch_size × steps × context_length = total_tokens
    Therefore: steps = total_tokens / (batch_size × context_length)
    """
    steps = total_tokens // (batch_size * context_length)
    actual_tokens = batch_size * steps * context_length
    print(f"Calculated {steps:,} steps for {actual_tokens:,} tokens")
    print(f"  (target was {total_tokens:,} tokens)")
    return steps


def create_base_config(
    learning_rate: float,
    experiment_name: str,
    total_tokens: int = 327_680_000,
    batch_size: int = 128,
    use_low_resource: bool = False,
) -> dict:
    """
    Create base configuration for learning rate experiments.

    Args:
        learning_rate: Peak learning rate
        experiment_name: Unique name for this experiment
        total_tokens: Target total tokens to process
        batch_size: Batch size
        use_low_resource: If True, use settings for CPU/MPS
    """
    # Adjust for low resource settings
    if use_low_resource:
        total_tokens = 40_000_000
        batch_size = 32
        print("Using low-resource settings:")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Batch size: {batch_size}")

    context_length = 256
    max_iterations = calculate_training_steps(total_tokens, batch_size, context_length)

    # Base hyperparameters from assignment
    config = {
        "description": f"Learning rate sweep: lr={learning_rate}",
        "dataset": "TinyStories",

        # Model architecture (17M parameters non-embedding)
        "vocab_size": 10000,
        "context_length": context_length,
        "d_model": 512,
        "num_layers": 4,
        "num_heads": 16,
        "d_ff": 1344,  # 8/3 * 512 ≈ 1365, but 1344 is multiple of 64
        "rope_theta": 10000.0,

        # Training hyperparameters
        "batch_size": batch_size,
        "max_iterations": max_iterations,
        "learning_rate": learning_rate,
        "min_learning_rate": learning_rate * 0.1,  # Decay to 10% of peak
        "warmup_iters": max(100, max_iterations // 100),  # 1% warmup

        # AdamW hyperparameters (from Kingma & Ba 2015)
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.1,
        "grad_clip": 1.0,

        # Data paths
        "train_data_path": "cs336_basics/artifacts/datasets/tinystories_train_tokens.npy",
        "val_data_path": "cs336_basics/artifacts/datasets/tinystories_tokens.npy",

        # System settings
        "device": None,  # Auto-detect
        "dtype": "float32",
        "seed": 42,

        # Logging (more frequent for analysis)
        "log_interval": 50,
        "eval_interval": 200,
        "checkpoint_interval": max(1000, max_iterations // 10),
    }

    return config


def run_learning_rate_sweep(
    learning_rates: list[float],
    output_base_dir: Path,
    use_low_resource: bool = False,
    device: str = None,
):
    """
    Run learning rate sweep experiments.

    Args:
        learning_rates: List of learning rates to try
        output_base_dir: Base directory for all experiments
        use_low_resource: If True, use CPU/MPS settings
        device: Device to use (cpu, cuda, mps)
    """
    print("="*80)
    print("LEARNING RATE SWEEP EXPERIMENT")
    print("="*80)
    print(f"\nLearning rates to test: {learning_rates}")
    print(f"Output directory: {output_base_dir}")
    if use_low_resource:
        print("Mode: Low-resource (CPU/MPS)")
        print("Target validation loss: 2.00")
    else:
        print("Mode: Full (GPU)")
        print("Target validation loss: 1.45")
    print()

    results = []

    for lr in learning_rates:
        exp_name = f"lr_{lr:.0e}".replace(".", "_").replace("-", "_")

        print("\n" + "="*80)
        print(f"Running experiment: {exp_name} (learning_rate={lr})")
        print("="*80)

        # Create config
        config = create_base_config(
            learning_rate=lr,
            experiment_name=exp_name,
            use_low_resource=use_low_resource,
        )

        if device:
            config["device"] = device

        # Create output directory
        output_dir = output_base_dir / exp_name

        # Save config for reference
        config_path = output_dir / "config_template.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Run experiment
        try:
            run_experiment(
                experiment_name=exp_name,
                config_dict=config,
                output_dir=output_dir,
            )

            # Record result
            results.append({
                "learning_rate": lr,
                "experiment_name": exp_name,
                "status": "completed",
            })

        except Exception as e:
            print(f"\n❌ Experiment {exp_name} failed: {e}")
            import traceback
            traceback.print_exc()

            results.append({
                "learning_rate": lr,
                "experiment_name": exp_name,
                "status": "failed",
                "error": str(e),
            })

    # Save sweep summary
    summary_path = output_base_dir / "sweep_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("LEARNING RATE SWEEP COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_base_dir}")
    print(f"Summary: {summary_path}")
    print("\nExperiments:")
    for result in results:
        status_symbol = "✓" if result["status"] == "completed" else "✗"
        print(f"  {status_symbol} lr={result['learning_rate']:.2e}: {result['status']}")


def main():
    parser = argparse.ArgumentParser(
        description="Learning rate sweep experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full sweep on GPU (default)
  uv run python -m cs336_basics.assignment_experiments.learning_rate_sweep

  # Low-resource mode (CPU/MPS)
  uv run python -m cs336_basics.assignment_experiments.learning_rate_sweep --low-resource

  # Custom learning rates
  uv run python -m cs336_basics.assignment_experiments.learning_rate_sweep \\
      --learning-rates 1e-4 3e-4 1e-3 3e-3

  # Specify device
  uv run python -m cs336_basics.assignment_experiments.learning_rate_sweep --device cuda
        """
    )

    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        help="Learning rates to test (default: sweep from 1e-4 to 1e-2)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cs336_basics/basics/runs/lr_sweep"),
        help="Output directory for all experiments"
    )
    parser.add_argument(
        "--low-resource",
        action="store_true",
        help="Use low-resource settings (40M tokens, smaller batch)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test with minimal iterations (for debugging)"
    )

    args = parser.parse_args()

    # Default learning rates: logarithmic sweep
    if args.learning_rates:
        learning_rates = sorted(args.learning_rates)
    else:
        # Reasonable range based on literature
        learning_rates = [
            1e-4,   # Conservative
            3e-4,   # Common starting point
            6e-4,   # GPT-3 style
            1e-3,   # Aggressive
            3e-3,   # Very aggressive
            6e-3,   # Near stability edge
            1e-2,   # Likely to diverge
        ]

    # Quick test mode
    if args.quick_test:
        print("⚠️  Quick test mode: using only 100 iterations")
        learning_rates = [3e-4, 1e-3]  # Just two rates
        # Will need to modify configs for quick test

    # Run the sweep
    run_learning_rate_sweep(
        learning_rates=learning_rates,
        output_base_dir=args.output_dir,
        use_low_resource=args.low_resource,
        device=args.device,
    )


if __name__ == "__main__":
    main()
