#!/usr/bin/env python3
"""
Performance Test Script

Tests training throughput with optimizations enabled.
Runs a small number of steps to verify performance without full training.

Usage:
    uv run python -m cs336_basics.assignment_experiments.lr_sweep.test_performance
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from cs336_basics.basics import create_experiment_config
from cs336_basics.basics.run_experiment import run_experiment


def test_performance():
    """Run quick performance test."""

    print("="*80)
    print("PERFORMANCE TEST")
    print("="*80)
    print("\nThis will run 100 training steps to verify performance optimizations.")
    print("Expected throughput on RTX 4090: 5,000-10,000 tokens/sec")
    print("Previous (broken) throughput: 400-500 tokens/sec")
    print()
    print("Watch for:")
    print("  ✓ TF32 enabled message")
    print("  ✓ Model compiled successfully message")
    print("  ✓ Tok/s > 3000 in training logs")
    print()
    input("Press Enter to start test (or Ctrl+C to cancel)...")
    print()

    # Create test configuration
    # Use smaller settings for quick test
    config = {
        "description": "Performance test with TF32 and compilation",
        "dataset": "TinyStories",

        # Model architecture (same as full sweep)
        "vocab_size": 10000,
        "context_length": 256,
        "d_model": 512,
        "num_layers": 4,
        "num_heads": 16,
        "d_ff": 1344,
        "rope_theta": 10000.0,

        # Training hyperparameters
        "batch_size": 128,
        "max_iterations": 100,  # Just 100 steps for testing
        "learning_rate": 3e-4,
        "min_learning_rate": 3e-5,
        "warmup_iters": 10,

        # AdamW hyperparameters
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

        # Logging (more frequent for test)
        "log_interval": 10,
        "eval_interval": 50,
        "checkpoint_interval": 1000,  # No checkpoints for test
    }

    # Run test
    output_dir = Path("cs336_basics/basics/runs/performance_test")

    print("="*80)
    print("Starting performance test...")
    print("="*80)
    print()

    run_experiment(
        experiment_name="performance_test",
        config_dict=config,
        output_dir=output_dir,
    )

    print()
    print("="*80)
    print("PERFORMANCE TEST COMPLETE")
    print("="*80)
    print()
    print("Results:")
    print("  - Check the 'Tok/s' values in the output above")
    print("  - Expected: 5,000-10,000 tokens/sec on RTX 4090")
    print("  - If you see 'TF32 enabled' and 'Model compiled', optimizations are working")
    print()
    print("Next steps:")
    print("  1. If throughput is good (>3000 tok/s), run full sweep:")
    print("     uv run python -m cs336_basics.assignment_experiments.lr_sweep.learning_rate_sweep")
    print()
    print("  2. If throughput is still low (<2000 tok/s), check:")
    print("     - GPU temperature (nvidia-smi)")
    print("     - GPU utilization (nvidia-smi)")
    print("     - See PERFORMANCE_ISSUE.md for more troubleshooting")
    print()


if __name__ == "__main__":
    test_performance()
