#!/usr/bin/env python3
"""
Verify Experiment Setup

Check that all components are properly installed and configured
before running experiments.
"""

import sys
from pathlib import Path
import importlib

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_imports():
    """Check that all required modules can be imported."""
    print("Checking imports...")
    modules = [
        "torch",
        "numpy",
        "matplotlib",
        "cs336_basics.transformer_training.model.transformer_lm",
        "cs336_basics.transformer_training.optimizer.adamw",
        "cs336_basics.data.data_loader",
        "cs336_basics.tokenizer.bpe_tokenizer",
        "assignment_experiments.experiment_tracker",
        "assignment_experiments.experiment_logger",
    ]

    all_ok = True
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module}: {e}")
            all_ok = False

    return all_ok


def check_data_files():
    """Check that tokenized datasets exist."""
    print("\nChecking data files...")
    base_dir = Path(__file__).parent.parent / "cs336_basics" / "artifacts"

    vocab_files = [
        base_dir / "vocabularies" / "tinystories_vocab.json",
        base_dir / "vocabularies" / "tinystories_merges.pkl",
    ]

    dataset_files = [
        base_dir / "datasets" / "tinystories_train_tokens.npy",
        base_dir / "datasets" / "tinystories_tokens.npy",
    ]

    all_ok = True

    print("  Vocabulary files:")
    for f in vocab_files:
        if f.exists():
            print(f"    ✓ {f.name}")
        else:
            print(f"    ✗ {f.name} (not found)")
            all_ok = False

    print("  Dataset files:")
    for f in dataset_files:
        if f.exists():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"    ✓ {f.name} ({size_mb:.1f} MB)")
        else:
            print(f"    ✗ {f.name} (not found)")
            all_ok = False

    return all_ok


def check_gpu():
    """Check GPU availability."""
    print("\nChecking GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"    {torch.cuda.device_count()} GPU(s)")
            print(f"    CUDA version: {torch.version.cuda}")
            return True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print(f"  ✓ MPS (Apple Silicon) available")
            return True
        else:
            print(f"  ⚠ No GPU available - will use CPU (training will be slow)")
            return True
    except Exception as e:
        print(f"  ✗ Error checking GPU: {e}")
        return False


def check_config_files():
    """Check that config files exist."""
    print("\nChecking configuration files...")
    config_dir = Path(__file__).parent / "configs"

    if not config_dir.exists():
        print(f"  ✗ Config directory not found: {config_dir}")
        return False

    configs = list(config_dir.glob("*.json"))
    if configs:
        print(f"  ✓ Found {len(configs)} configuration file(s):")
        for config in configs:
            print(f"    - {config.name}")
        return True
    else:
        print(f"  ⚠ No configuration files found in {config_dir}")
        print(f"    Create configs or use the baseline template")
        return True


def test_minimal_model():
    """Test that a minimal model can be created."""
    print("\nTesting minimal model creation...")
    try:
        import torch
        from cs336_basics.transformer_training.model.transformer_lm import TransformerLM

        # Create tiny model
        model = TransformerLM(
            vocab_size=100,
            context_length=32,
            d_model=64,
            num_layers=2,
            num_heads=2,
            d_ff=128,
            device="cpu",
        )

        # Test forward pass
        x = torch.randint(0, 100, (2, 32))
        y = model(x)

        assert y.shape == (2, 32, 100), f"Unexpected output shape: {y.shape}"
        print("  ✓ Model creation and forward pass successful")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_experiment_tracker():
    """Test experiment tracker functionality."""
    print("\nTesting experiment tracker...")
    try:
        from cs336_basics.basics.experiment_tracker import (
            create_experiment_config,
            ExperimentTracker
        )
        import tempfile
        import shutil

        # Create temp directory
        temp_dir = Path(tempfile.mkdtemp())

        try:
            # Create config
            config = create_experiment_config(
                experiment_name="test",
                description="Test experiment",
                vocab_size=100,
                max_iterations=10,
            )

            # Create tracker
            tracker = ExperimentTracker("test", config, temp_dir)

            # Log some metrics
            tracker.log_step(step=1, train_loss=5.0, learning_rate=1e-3)
            tracker.log_step(step=2, train_loss=4.5, val_loss=4.8, learning_rate=9e-4)

            # Finalize
            tracker.finalize()

            # Check files were created
            assert (temp_dir / "config.json").exists()
            assert (temp_dir / "metrics.csv").exists()
            assert (temp_dir / "summary.json").exists()

            print("  ✓ Experiment tracker working correctly")
            return True

        finally:
            # Cleanup
            shutil.rmtree(temp_dir)

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary(results):
    """Print summary of checks."""
    print("\n" + "="*60)
    print("SETUP VERIFICATION SUMMARY")
    print("="*60)

    all_passed = all(results.values())

    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check}")

    print("="*60)

    if all_passed:
        print("\n✓ All checks passed! You're ready to run experiments.")
        print("\nNext steps:")
        print("  1. Review: assignment_experiments/QUICK_START.md")
        print("  2. Run baseline: uv run python -m assignment_experiments.run_experiment \\")
        print("                       --name baseline_17m \\")
        print("                       --config assignment_experiments/configs/tinystories_17m_baseline.json")
    else:
        print("\n⚠ Some checks failed. Please resolve the issues above.")
        print("\nCommon fixes:")
        print("  - Missing data: Run tokenization scripts (see QUICK_START.md)")
        print("  - Import errors: Install dependencies with 'uv sync'")
        print("  - Config files: They're in assignment_experiments/configs/")

    print()
    return all_passed


def main():
    """Run all verification checks."""
    print("="*60)
    print("EXPERIMENT SETUP VERIFICATION")
    print("="*60)
    print()

    results = {
        "Python imports": check_imports(),
        "Data files": check_data_files(),
        "GPU/device": check_gpu(),
        "Config files": check_config_files(),
        "Model creation": test_minimal_model(),
        "Experiment tracker": test_experiment_tracker(),
    }

    success = print_summary(results)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
