"""
Quick Validation on TinyStories

Tests weight-tied model on TinyStories dataset for ~15 minutes.
This is a sanity check before running expensive OpenWebText experiments.

Expected results:
- Training should complete without errors
- Final validation loss should be ~1.3-1.5 (similar to baseline)
- Weight tying should reduce parameters by ~30-40%

Usage:
    uv run python experiments/test_on_tinystories.py --config configs/optimized_1.5hr.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from cs336_basics.basics.run_experiment import run_experiment


def create_tinystories_config(base_config_path: Path) -> dict:
    """
    Create TinyStories test configuration from base config.

    Adapts the OpenWebText config for quick TinyStories testing:
    - Smaller vocabulary (10k vs 32k)
    - Shorter training (10k iterations vs 60k)
    - TinyStories dataset paths
    """
    with open(base_config_path) as f:
        config = json.load(f)

    # Adapt for TinyStories
    config["experiment_name"] = "tinystories_weight_tied_test"
    config["description"] = "Quick validation of weight tying on TinyStories"
    config["vocab_size"] = 10000
    config["max_iterations"] = 10000  # ~15 minutes
    config["batch_size"] = 8  # Smaller batch for TinyStories
    config["dataset"] = "TinyStories"
    config["train_data_path"] = "cs336_basics/artifacts/datasets/tinystories_train_tokens.npy"
    config["val_data_path"] = "cs336_basics/artifacts/datasets/tinystories_tokens.npy"

    # Keep optimizations
    config["use_weight_tying"] = True
    config["embedding_scale"] = True
    config["dtype"] = "bfloat16"

    return config


def main():
    parser = argparse.ArgumentParser(description="Test weight tying on TinyStories")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("cs336_basics/assignment_experiments/leaderboard_optimization/configs/optimized_1.5hr.json"),
        help="Base configuration file to adapt for TinyStories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cs336_basics/basics/runs/leaderboard_test_tinystories"),
        help="Output directory for results",
    )
    args = parser.parse_args()

    # Create adapted config
    print("Creating TinyStories test configuration...")
    config = create_tinystories_config(args.config)

    # Save adapted config
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved test config to: {config_path}")
    print(f"\nConfiguration:")
    print(f"  - Vocabulary: {config['vocab_size']:,}")
    print(f"  - Batch size: {config['batch_size']}")
    print(f"  - Max iterations: {config['max_iterations']:,}")
    print(f"  - Weight tying: {config['use_weight_tying']}")
    print(f"  - Data type: {config['dtype']}")
    print(f"  - Expected time: ~15 minutes")

    # Run experiment
    print("\n" + "="*80)
    print("Starting TinyStories validation...")
    print("="*80 + "\n")

    try:
        run_experiment(
            experiment_name=config["experiment_name"],
            config_dict=config,
            output_dir=output_dir,
        )

        print("\n" + "="*80)
        print("✓ TinyStories validation completed successfully!")
        print("="*80)

        # Load and display results
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)

            stats = summary.get("statistics", {})
            val_loss = stats.get("val_loss", {})

            print("\nResults:")
            print(f"  Best validation loss: {val_loss.get('best', 'N/A')}")
            print(f"  Final validation loss: {val_loss.get('final', 'N/A')}")
            print(f"  Training time: {stats.get('total_wallclock_hours', 'N/A'):.2f} hours")

            # Check if results are reasonable
            best_loss = val_loss.get("best", float("inf"))
            if best_loss < 2.0:
                print("\n✓ Loss looks good! Ready for OpenWebText testing.")
            else:
                print("\n⚠ Loss higher than expected. Check for issues.")

    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
