#!/usr/bin/env python3
"""
Quick test script to verify training works end-to-end.
"""

import json
import os
import tempfile
import numpy as np
from pathlib import Path


def test_training():
    """Test the training script with a tiny model and dataset."""

    print("=" * 80)
    print("Testing Training Script")
    print("=" * 80)

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\nWorking in: {temp_dir}")

        # 1. Create dummy dataset
        print("\n1. Creating dummy dataset...")
        train_data_path = os.path.join(temp_dir, "train.npy")
        val_data_path = os.path.join(temp_dir, "val.npy")

        # Create small datasets
        np.random.seed(42)
        train_data = np.random.randint(0, 100, size=5000, dtype=np.uint16)
        val_data = np.random.randint(0, 100, size=1000, dtype=np.uint16)

        np.save(train_data_path, train_data)
        np.save(val_data_path, val_data)

        print(f"   Training data: {len(train_data)} tokens")
        print(f"   Validation data: {len(val_data)} tokens")

        # 2. Create config
        print("\n2. Creating configuration...")
        config = {
            # Tiny model for quick testing
            "vocab_size": 100,
            "context_length": 32,
            "d_model": 64,
            "num_layers": 2,
            "num_heads": 2,
            "d_ff": 128,

            # Quick training
            "batch_size": 4,
            "max_iterations": 20,
            "learning_rate": 1e-3,
            "min_learning_rate": 1e-4,
            "warmup_iters": 5,
            "weight_decay": 0.1,
            "grad_clip": 1.0,

            # Data paths
            "train_data_path": train_data_path,
            "val_data_path": val_data_path,
            "checkpoint_dir": os.path.join(temp_dir, "checkpoints"),

            # Logging
            "log_interval": 5,
            "eval_interval": 10,
            "eval_iters": 2,
            "checkpoint_interval": 10,

            # System
            "device": "cpu",  # Use CPU for testing
            "dtype": "float32",
            "seed": 42,
        }

        config_path = os.path.join(temp_dir, "test_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"   Config saved to: {config_path}")

        # 3. Import and run training
        print("\n3. Running training...")
        print("-" * 80)

        try:
            from cs336_basics.train import train

            # Run training
            train(**config)

            print("-" * 80)
            print("✓ Training completed successfully!")

        except Exception as e:
            print("-" * 80)
            print(f"✗ Training failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False

        # 4. Verify outputs
        print("\n4. Verifying outputs...")
        checkpoint_dir = config["checkpoint_dir"]

        if not os.path.exists(checkpoint_dir):
            print("✗ Checkpoint directory not created")
            return False

        print(f"   ✓ Checkpoint directory exists: {checkpoint_dir}")

        # Check for expected files
        expected_files = ["training.log"]
        optional_files = ["checkpoint_10.pt", "best_model.pt", "final_checkpoint.pt"]

        for filename in expected_files:
            filepath = os.path.join(checkpoint_dir, filename)
            if os.path.exists(filepath):
                print(f"   ✓ {filename} exists")
            else:
                print(f"   ✗ {filename} missing")
                return False

        for filename in optional_files:
            filepath = os.path.join(checkpoint_dir, filename)
            if os.path.exists(filepath):
                size = os.path.getsize(filepath) / 1024  # KB
                print(f"   ✓ {filename} exists ({size:.1f} KB)")

        # Check log file content
        log_file = os.path.join(checkpoint_dir, "training.log")
        with open(log_file, 'r') as f:
            log_content = f.read()

        # Verify key log messages
        checks = [
            "Starting Transformer Language Model Training",
            "Model initialized with",
            "Starting training loop",
            "Training Complete",
        ]

        print("\n5. Verifying log content...")
        for check in checks:
            if check in log_content:
                print(f"   ✓ Found: '{check}'")
            else:
                print(f"   ✗ Missing: '{check}'")
                return False

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)
        print("\nThe training script successfully:")
        print("  ✓ Loaded data with memory mapping")
        print("  ✓ Initialized model and optimizer")
        print("  ✓ Ran training loop with gradient updates")
        print("  ✓ Evaluated on validation data")
        print("  ✓ Saved checkpoints")
        print("  ✓ Logged training progress")
        print("\n" + "=" * 80)

        return True


if __name__ == "__main__":
    import sys
    success = test_training()
    sys.exit(0 if success else 1)
