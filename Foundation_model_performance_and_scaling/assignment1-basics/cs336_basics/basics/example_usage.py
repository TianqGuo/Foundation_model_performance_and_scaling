#!/usr/bin/env python3
"""
Example: Using the Experiment Tracking Infrastructure

This file demonstrates how to use the experiment tracking components.
Run with: uv run python assignment_experiments/example_usage.py
"""

import sys
from pathlib import Path
import tempfile
import shutil

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cs336_basics.basics.experiment_tracker import (
    ExperimentTracker,
    ExperimentConfig,
    create_experiment_config
)
from cs336_basics.basics.experiment_logger import ExperimentLogger


def example_basic_tracking():
    """Example 1: Basic metric tracking."""
    print("="*60)
    print("EXAMPLE 1: Basic Metric Tracking")
    print("="*60)

    # Create temporary directory for output
    temp_dir = Path(tempfile.mkdtemp())
    print(f"\nOutput directory: {temp_dir}")

    try:
        # Create experiment configuration
        config = create_experiment_config(
            experiment_name="example_basic",
            description="Basic tracking example",
            vocab_size=10000,
            d_model=512,
            num_layers=6,
            max_iterations=100,
            batch_size=32,
        )

        # Initialize tracker
        tracker = ExperimentTracker(
            experiment_name="example_basic",
            config=config,
            output_dir=temp_dir
        )

        print("\n1. Logging training steps...")
        # Simulate training steps
        for step in range(0, 100, 10):
            # Simulated loss that decreases
            train_loss = 5.0 - (step / 100) * 2.0

            tracker.log_step(
                step=step,
                train_loss=train_loss,
                learning_rate=0.001 * (1 - step / 100),
                tokens_per_sec=8000.0,
            )

        print("2. Logging validation steps...")
        # Simulate validation steps
        for step in [20, 40, 60, 80]:
            val_loss = 4.8 - (step / 100) * 1.5
            val_ppl = 2.718 ** val_loss

            tracker.log_step(
                step=step,
                val_loss=val_loss,
                val_perplexity=val_ppl,
            )

        print("3. Finalizing and computing statistics...")
        # Finalize
        tracker.finalize()

        # Show statistics
        stats = tracker.compute_statistics()
        print("\nStatistics:")
        print(f"  Total steps: {stats['total_steps']}")
        print(f"  Training loss (final): {stats['train_loss']['final']:.4f}")
        print(f"  Validation loss (best): {stats['val_loss']['best']:.4f}")

        print(f"\n✓ Example 1 complete. Check {temp_dir} for outputs.")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"✓ Cleaned up {temp_dir}")


def example_with_visualization():
    """Example 2: Using ExperimentLogger with visualization."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Logger with Visualization")
    print("="*60)

    temp_dir = Path(tempfile.mkdtemp())
    print(f"\nOutput directory: {temp_dir}")

    try:
        # Create configuration
        config = create_experiment_config(
            experiment_name="example_viz",
            description="Visualization example",
            vocab_size=10000,
            max_iterations=200,
        )

        # Initialize logger
        logger = ExperimentLogger(
            experiment_name="example_viz",
            config=config,
            output_dir=temp_dir,
        )

        print("\n1. Logging training and validation...")
        # Simulate training with some noise
        import random
        random.seed(42)

        for step in range(0, 200, 5):
            # Training loss with noise
            base_loss = 5.0 - (step / 200) * 2.5
            noise = random.gauss(0, 0.1)
            train_loss = base_loss + noise

            logger.log_training_step(
                step=step,
                train_loss=train_loss,
                learning_rate=0.001 * (1 - step / 200),
                tokens_per_sec=8000.0 + random.gauss(0, 500),
            )

            # Validation every 20 steps
            if step % 20 == 0 and step > 0:
                val_loss = base_loss - 0.2 + random.gauss(0, 0.05)
                val_ppl = 2.718 ** val_loss

                logger.log_validation_step(
                    step=step,
                    val_loss=val_loss,
                    val_perplexity=val_ppl,
                )

        print("2. Finalizing and generating plots...")
        # Finalize (generates plots automatically)
        logger.finalize()

        # Check generated files
        print("\n3. Generated files:")
        for file in temp_dir.glob("*"):
            if file.is_file():
                print(f"   - {file.name}")

        print(f"\n✓ Example 2 complete. Check {temp_dir} for plots.")
        print(f"  Plots: loss_curves.png, lr_schedule.png")

    finally:
        # Keep this one for inspection
        print(f"\n⚠ Output kept at: {temp_dir}")
        print(f"   (Delete manually when done inspecting)")


def example_programmatic_config():
    """Example 3: Creating configs programmatically."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Programmatic Configuration")
    print("="*60)

    # Method 1: Using create_experiment_config helper
    config1 = create_experiment_config(
        experiment_name="small_model",
        description="Small model for testing",
        vocab_size=5000,
        d_model=256,
        num_layers=4,
        num_heads=4,
        d_ff=1024,
        batch_size=16,
        max_iterations=1000,
    )

    print("\nMethod 1: Using helper function")
    print(f"  Experiment: {config1.experiment_name}")
    print(f"  Model: {config1.num_layers} layers, {config1.d_model} dim")
    print(f"  Training: {config1.max_iterations} iterations, batch {config1.batch_size}")

    # Method 2: Direct instantiation
    config2 = ExperimentConfig(
        experiment_name="large_model",
        experiment_id="manual_001",
        description="Large model experiment",
        vocab_size=10000,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=3072,
        batch_size=64,
        max_iterations=10000,
        learning_rate=0.0006,
        tags=["baseline", "17m"],
    )

    print("\nMethod 2: Direct instantiation")
    print(f"  Experiment: {config2.experiment_name}")
    print(f"  Model: {config2.num_layers} layers, {config2.d_model} dim")
    print(f"  Tags: {config2.tags}")

    # Save to JSON
    temp_dir = Path(tempfile.mkdtemp())
    config_path = temp_dir / "my_config.json"
    config2.to_json(config_path)
    print(f"\n✓ Config saved to: {config_path}")

    # Load from JSON
    loaded_config = ExperimentConfig.from_json(config_path)
    print(f"✓ Config loaded: {loaded_config.experiment_name}")

    # Cleanup
    shutil.rmtree(temp_dir)


def example_comparison():
    """Example 4: Comparing experiments."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Experiment Comparison")
    print("="*60)

    from cs336_basics.basics.experiment_logger import compare_experiments

    # Create multiple experiment directories
    base_dir = Path(tempfile.mkdtemp())
    exp_dirs = []

    try:
        print("\n1. Creating mock experiments...")
        import random
        random.seed(42)

        for i, exp_name in enumerate(["baseline", "higher_lr", "lower_wd"]):
            exp_dir = base_dir / exp_name
            exp_dir.mkdir()

            config = create_experiment_config(
                experiment_name=exp_name,
                vocab_size=10000,
                max_iterations=100,
            )

            tracker = ExperimentTracker(exp_name, config, exp_dir)

            # Different loss curves
            offset = i * 0.3
            for step in range(0, 100, 5):
                train_loss = (5.0 - offset) - (step / 100) * 2.0 + random.gauss(0, 0.1)
                tracker.log_step(step=step, train_loss=train_loss)

                if step % 20 == 0:
                    val_loss = train_loss - 0.2
                    tracker.log_step(step=step, val_loss=val_loss)

            tracker.finalize()
            exp_dirs.append(exp_dir)
            print(f"   Created {exp_name}")

        print("\n2. Generating comparison plot...")
        # Compare experiments
        compare_path = base_dir / "comparison.png"
        compare_experiments(
            exp_dirs,
            compare_path,
            experiment_names=["Baseline", "Higher LR", "Lower WD"]
        )

        print(f"✓ Comparison plot saved to: {compare_path}")
        print(f"\n⚠ Output kept at: {base_dir}")
        print(f"   (Delete manually when done)")

    except Exception as e:
        print(f"Error: {e}")
        shutil.rmtree(base_dir)


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("EXPERIMENT TRACKING EXAMPLES")
    print("="*60)
    print("\nThese examples demonstrate how to use the infrastructure.")
    print("Temporary outputs will be created and cleaned up (mostly).\n")

    try:
        # Run examples
        example_basic_tracking()
        example_with_visualization()
        example_programmatic_config()
        example_comparison()

        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETE")
        print("="*60)
        print("\nNext steps:")
        print("  1. Review the code in this file")
        print("  2. Read QUICK_START.md for real usage")
        print("  3. Run actual experiments with run_experiment.py")
        print()

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
