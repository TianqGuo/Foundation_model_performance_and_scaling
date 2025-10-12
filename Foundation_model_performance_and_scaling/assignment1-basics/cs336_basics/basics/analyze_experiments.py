#!/usr/bin/env python3
"""
Experiment Analysis Utilities

Tools for analyzing and comparing experiments:
- Load and display experiment summaries
- Compare multiple experiments
- Generate comparison tables
- Plot loss curves side-by-side
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any
import argparse
import pandas as pd


def load_experiment_summary(exp_dir: Path) -> Dict[str, Any]:
    """Load experiment summary from directory."""
    summary_path = exp_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")

    with open(summary_path, 'r') as f:
        return json.load(f)


def load_experiment_metrics(exp_dir: Path) -> pd.DataFrame:
    """Load experiment metrics CSV as pandas DataFrame."""
    metrics_path = exp_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics not found: {metrics_path}")

    return pd.read_csv(metrics_path)


def print_experiment_summary(exp_dir: Path) -> None:
    """Print a formatted summary of an experiment."""
    summary = load_experiment_summary(exp_dir)

    print("\n" + "="*80)
    print(f"EXPERIMENT: {summary['experiment_name']}")
    print("="*80)

    # Configuration
    config = summary['config']
    print("\nModel Configuration:")
    print(f"  Architecture: {config['num_layers']} layers, {config['d_model']} hidden dim, {config['num_heads']} heads")
    print(f"  Parameters: ~{(config['d_model']**2 * config['num_layers'] * 12) / 1e6:.1f}M")
    print(f"  Context Length: {config['context_length']}")
    print(f"  Vocabulary Size: {config['vocab_size']}")

    print("\nTraining Configuration:")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Max Iterations: {config['max_iterations']}")
    print(f"  Learning Rate: {config['learning_rate']:.2e} → {config['min_learning_rate']:.2e}")
    print(f"  Warmup Iterations: {config['warmup_iters']}")
    print(f"  Weight Decay: {config['weight_decay']}")

    # Statistics
    stats = summary['statistics']
    print("\nTraining Statistics:")
    print(f"  Total Steps: {stats['total_steps']:,}")
    print(f"  Total Time: {stats['total_wallclock_hours']:.2f} hours")

    if 'train_loss' in stats:
        train_stats = stats['train_loss']
        print(f"\nTraining Loss:")
        print(f"  Initial: {train_stats['initial']:.4f}")
        print(f"  Final: {train_stats['final']:.4f}")
        print(f"  Min: {train_stats['min']:.4f}")
        print(f"  Mean: {train_stats['mean']:.4f}")

    if 'val_loss' in stats:
        val_stats = stats['val_loss']
        print(f"\nValidation Loss:")
        print(f"  Final: {val_stats['final']:.4f}")
        print(f"  Best: {val_stats['best']:.4f}")
        print(f"  Mean: {val_stats['mean']:.4f}")

    print("="*80 + "\n")


def compare_experiments_table(exp_dirs: List[Path], exp_names: List[str] = None) -> pd.DataFrame:
    """Create a comparison table of multiple experiments."""
    if exp_names is None:
        exp_names = [d.name for d in exp_dirs]

    data = []
    for exp_dir, name in zip(exp_dirs, exp_names):
        try:
            summary = load_experiment_summary(exp_dir)
            config = summary['config']
            stats = summary['statistics']

            row = {
                'Experiment': name,
                'Layers': config['num_layers'],
                'd_model': config['d_model'],
                'Heads': config['num_heads'],
                'Batch Size': config['batch_size'],
                'LR (peak)': config['learning_rate'],
                'Steps': stats['total_steps'],
                'Time (hrs)': f"{stats['total_wallclock_hours']:.2f}",
            }

            if 'train_loss' in stats:
                row['Train Loss (final)'] = f"{stats['train_loss']['final']:.4f}"

            if 'val_loss' in stats:
                row['Val Loss (final)'] = f"{stats['val_loss']['final']:.4f}"
                row['Val Loss (best)'] = f"{stats['val_loss']['best']:.4f}"

            data.append(row)

        except Exception as e:
            print(f"Warning: Could not load {exp_dir}: {e}")

    return pd.DataFrame(data)


def find_all_experiments(runs_dir: Path) -> List[Path]:
    """Find all experiment directories in runs folder."""
    if not runs_dir.exists():
        return []

    experiments = []
    for exp_dir in runs_dir.iterdir():
        if exp_dir.is_dir() and (exp_dir / "summary.json").exists():
            experiments.append(exp_dir)

    return sorted(experiments, key=lambda x: x.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="Analyze experiments")
    parser.add_argument(
        "command",
        choices=["summary", "compare", "list"],
        help="Command to run"
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        help="Experiment names or paths"
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path(__file__).parent / "runs",
        help="Directory containing experiment runs"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for comparison table (CSV)"
    )

    args = parser.parse_args()

    if args.command == "list":
        # List all experiments
        experiments = find_all_experiments(args.runs_dir)
        if not experiments:
            print(f"No experiments found in {args.runs_dir}")
            return

        print(f"\nFound {len(experiments)} experiments in {args.runs_dir}:\n")
        for exp_dir in experiments:
            try:
                summary = load_experiment_summary(exp_dir)
                stats = summary['statistics']
                print(f"  {exp_dir.name}")
                print(f"    Steps: {stats['total_steps']:,}, Time: {stats['total_wallclock_hours']:.2f}h")
                if 'val_loss' in stats:
                    print(f"    Val Loss: {stats['val_loss']['best']:.4f} (best)")
            except Exception as e:
                print(f"  {exp_dir.name} (error: {e})")
            print()

    elif args.command == "summary":
        # Print detailed summary
        if not args.experiments:
            parser.error("--experiments required for summary command")

        for exp in args.experiments:
            exp_path = Path(exp) if Path(exp).exists() else args.runs_dir / exp
            if not exp_path.exists():
                print(f"Error: Experiment not found: {exp_path}")
                continue
            print_experiment_summary(exp_path)

    elif args.command == "compare":
        # Compare multiple experiments
        if not args.experiments:
            parser.error("--experiments required for compare command")

        exp_paths = []
        for exp in args.experiments:
            exp_path = Path(exp) if Path(exp).exists() else args.runs_dir / exp
            if not exp_path.exists():
                print(f"Warning: Experiment not found: {exp_path}")
                continue
            exp_paths.append(exp_path)

        if not exp_paths:
            print("Error: No valid experiments to compare")
            return

        # Create comparison table
        comparison_df = compare_experiments_table(exp_paths)

        print("\n" + "="*80)
        print("EXPERIMENT COMPARISON")
        print("="*80 + "\n")
        print(comparison_df.to_string(index=False))
        print()

        # Save to CSV if requested
        if args.output:
            comparison_df.to_csv(args.output, index=False)
            print(f"Comparison table saved to: {args.output}\n")

        # Generate comparison plot
        from cs336_basics.basics.experiment_logger import compare_experiments
        plot_path = args.runs_dir / "experiment_comparison.png"
        try:
            compare_experiments(exp_paths, plot_path)
            print(f"Comparison plot saved to: {plot_path}\n")
        except Exception as e:
            print(f"Warning: Could not generate comparison plot: {e}\n")


if __name__ == "__main__":
    main()
