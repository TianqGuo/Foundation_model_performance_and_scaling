#!/usr/bin/env python3
"""
Analyze Learning Rate Sweep Results

This script analyzes the results of learning rate sweep experiments and generates:
- Comparison plots of all learning rates
- Analysis of divergence vs convergence
- Identification of optimal learning rate
- "Edge of stability" analysis
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_experiment_metrics(exp_dir: Path) -> pd.DataFrame:
    """Load metrics CSV from experiment directory."""
    metrics_path = exp_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics not found: {metrics_path}")
    return pd.read_csv(metrics_path)


def extract_learning_rate(exp_name: str) -> float:
    """Extract learning rate from experiment name like 'lr_1e-03'."""
    # Format: lr_1e_03 or lr_0_001
    parts = exp_name.replace("lr_", "").replace("_", ".")
    # Handle scientific notation
    if "e" in parts:
        return float(parts.replace(".", "e", 1))
    else:
        return float("0." + parts)


def check_divergence(metrics: pd.DataFrame, threshold: float = 10.0) -> bool:
    """
    Check if training diverged.

    Criteria for divergence:
    - Training loss exceeds threshold (default 10.0)
    - NaN values in loss
    - Loss increases significantly after initial decrease
    """
    train_losses = metrics['train_loss'].dropna()

    if len(train_losses) == 0:
        return True  # No data = diverged

    # Check for NaN
    if train_losses.isna().any():
        return True

    # Check if loss exceeds threshold
    if (train_losses > threshold).any():
        return True

    # Check if loss increases significantly after first 20% of training
    if len(train_losses) > 100:
        early_min = train_losses[:len(train_losses)//5].min()
        late_max = train_losses[len(train_losses)//2:].max()
        if late_max > early_min * 2:  # Loss doubled
            return True

    return False


def analyze_convergence_rate(metrics: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze convergence rate.

    Returns:
    - initial_loss: Loss at start
    - final_loss: Loss at end
    - min_loss: Minimum achieved loss
    - convergence_rate: How quickly loss decreased
    """
    train_losses = metrics['train_loss'].dropna()

    if len(train_losses) < 10:
        return None

    # Use first 10% and last 10% to measure convergence
    n_early = max(10, len(train_losses) // 10)
    n_late = max(10, len(train_losses) // 10)

    initial_loss = train_losses[:n_early].mean()
    final_loss = train_losses[-n_late:].mean()
    min_loss = train_losses.min()

    # Convergence rate: how much loss decreased per step
    steps = metrics['step'].dropna()
    if len(steps) > 1:
        total_steps = steps.iloc[-1] - steps.iloc[0]
        convergence_rate = (initial_loss - final_loss) / total_steps
    else:
        convergence_rate = 0.0

    return {
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'min_loss': min_loss,
        'convergence_rate': convergence_rate,
    }


def plot_lr_comparison(
    results: Dict[float, Dict],
    output_path: Path,
    title: str = "Learning Rate Sweep - Loss Curves"
):
    """
    Plot comparison of all learning rates.

    Args:
        results: Dict mapping learning_rate -> {'metrics': df, 'diverged': bool, ...}
        output_path: Where to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Sort by learning rate
    sorted_lrs = sorted(results.keys())

    # Color map
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_lrs)))

    for i, lr in enumerate(sorted_lrs):
        result = results[lr]
        metrics = result['metrics']
        diverged = result['diverged']

        # Plot by steps
        steps = metrics['step'].dropna()
        train_losses = metrics['train_loss'].dropna()

        linestyle = '--' if diverged else '-'
        alpha = 0.5 if diverged else 1.0
        label = f"lr={lr:.2e}"
        if diverged:
            label += " (diverged)"

        axes[0].plot(steps, train_losses, label=label, color=colors[i],
                    linestyle=linestyle, alpha=alpha, linewidth=2)

        # Plot by time (if available)
        if 'elapsed_time' in metrics.columns:
            times = metrics['elapsed_time'].dropna()
            axes[1].plot(times, train_losses[:len(times)], label=label,
                        color=colors[i], linestyle=linestyle, alpha=alpha, linewidth=2)

    # Format axes
    axes[0].set_xlabel('Training Steps')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Loss vs Steps')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    axes[1].set_xlabel('Wallclock Time (minutes)')
    axes[1].set_ylabel('Training Loss')
    axes[1].set_title('Loss vs Time')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Comparison plot saved to: {output_path}")


def plot_final_loss_vs_lr(
    results: Dict[float, Dict],
    output_path: Path,
):
    """Plot final validation loss vs learning rate."""
    lrs = []
    final_losses = []
    min_losses = []
    diverged_lrs = []

    for lr, result in sorted(results.items()):
        lrs.append(lr)

        if result['diverged']:
            diverged_lrs.append(lr)
            final_losses.append(None)
            min_losses.append(None)
        else:
            stats = result['stats']
            final_losses.append(stats['final_loss'])
            min_losses.append(stats['min_loss'])

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot final loss
    valid_lrs = [lr for lr, loss in zip(lrs, final_losses) if loss is not None]
    valid_finals = [loss for loss in final_losses if loss is not None]
    valid_mins = [loss for loss in min_losses if loss is not None]

    ax.plot(valid_lrs, valid_finals, 'o-', label='Final Loss', linewidth=2, markersize=8)
    ax.plot(valid_lrs, valid_mins, 's--', label='Min Loss', linewidth=2, markersize=8)

    # Mark diverged learning rates
    if diverged_lrs:
        ax.scatter(diverged_lrs, [ax.get_ylim()[1]] * len(diverged_lrs),
                  color='red', marker='x', s=200, label='Diverged', zorder=5)

    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Loss')
    ax.set_title('Final Loss vs Learning Rate')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add target line (1.45 for full, 2.00 for low-resource)
    target = 1.45  # Adjust based on your setting
    ax.axhline(y=target, color='green', linestyle=':', label=f'Target ({target})')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Final loss plot saved to: {output_path}")


def analyze_edge_of_stability(results: Dict[float, Dict]) -> Dict:
    """
    Analyze the "edge of stability" - the relationship between divergence point
    and optimal learning rate.

    Folk wisdom: Best LR is just below the divergence point.
    """
    # Find divergence point
    sorted_lrs = sorted(results.keys())

    diverged_lrs = [lr for lr in sorted_lrs if results[lr]['diverged']]
    converged_lrs = [lr for lr in sorted_lrs if not results[lr]['diverged']]

    if not diverged_lrs:
        print("⚠️  No divergent runs found. Consider testing higher learning rates.")
        edge_lr = None
    else:
        edge_lr = min(diverged_lrs)  # First divergent LR

    # Find best learning rate (lowest final loss)
    if converged_lrs:
        best_lr = min(converged_lrs,
                     key=lambda lr: results[lr]['stats']['final_loss'])
        best_loss = results[best_lr]['stats']['final_loss']
    else:
        print("⚠️  No converged runs found!")
        best_lr = None
        best_loss = None

    analysis = {
        'edge_lr': edge_lr,
        'best_lr': best_lr,
        'best_loss': best_loss,
        'ratio': edge_lr / best_lr if (edge_lr and best_lr) else None,
        'all_converged': converged_lrs,
        'all_diverged': diverged_lrs,
    }

    return analysis


def generate_analysis_report(
    results: Dict[float, Dict],
    edge_analysis: Dict,
    output_path: Path,
):
    """Generate markdown analysis report."""
    report = []

    report.append("# Learning Rate Sweep Analysis\n")
    report.append("## Summary\n")

    # Count results
    n_total = len(results)
    n_converged = sum(1 for r in results.values() if not r['diverged'])
    n_diverged = n_total - n_converged

    report.append(f"- Total experiments: {n_total}\n")
    report.append(f"- Converged: {n_converged}\n")
    report.append(f"- Diverged: {n_diverged}\n\n")

    # Best learning rate
    if edge_analysis['best_lr']:
        report.append("## Best Learning Rate\n")
        report.append(f"- Learning rate: `{edge_analysis['best_lr']:.2e}`\n")
        report.append(f"- Final loss: `{edge_analysis['best_loss']:.4f}`\n")
        report.append(f"- Target loss: `1.45` (full) or `2.00` (low-resource)\n\n")

    # Edge of stability
    report.append("## Edge of Stability Analysis\n")
    if edge_analysis['edge_lr'] and edge_analysis['best_lr']:
        report.append(f"- First divergent LR: `{edge_analysis['edge_lr']:.2e}`\n")
        report.append(f"- Best LR: `{edge_analysis['best_lr']:.2e}`\n")
        report.append(f"- Ratio (edge/best): `{edge_analysis['ratio']:.2f}`\n\n")

        report.append("**Analysis:**\n")
        if edge_analysis['ratio'] < 2:
            report.append("- Best LR is close to divergence point (within 2x)\n")
            report.append("- Confirms 'edge of stability' folk wisdom\n\n")
        else:
            report.append("- Best LR is more conservative (>2x below divergence)\n")
            report.append("- May indicate room for more aggressive tuning\n\n")
    else:
        report.append("- Could not determine edge (no divergence or no convergence)\n\n")

    # Detailed results table
    report.append("## Detailed Results\n\n")
    report.append("| Learning Rate | Status | Final Loss | Min Loss | Conv. Rate |\n")
    report.append("|---------------|--------|------------|----------|------------|\n")

    for lr in sorted(results.keys()):
        result = results[lr]
        status = "❌ Diverged" if result['diverged'] else "✅ Converged"

        if result['diverged']:
            report.append(f"| {lr:.2e} | {status} | - | - | - |\n")
        else:
            stats = result['stats']
            report.append(
                f"| {lr:.2e} | {status} | "
                f"{stats['final_loss']:.4f} | "
                f"{stats['min_loss']:.4f} | "
                f"{stats['convergence_rate']:.2e} |\n"
            )

    report.append("\n## Recommendations\n\n")

    if edge_analysis['best_lr']:
        best = edge_analysis['best_lr']
        report.append(f"1. **Use learning rate: `{best:.2e}`**\n")
        report.append(f"2. Achieved loss: `{edge_analysis['best_loss']:.4f}`\n")

        if edge_analysis['best_loss'] > 1.45:
            report.append(f"3. ⚠️  Did not meet target of 1.45. Consider:\n")
            report.append(f"   - Training for more steps\n")
            report.append(f"   - Tuning other hyperparameters (warmup, weight decay)\n")
            report.append(f"   - Adjusting learning rate schedule\n")
        else:
            report.append(f"3. ✅ Met target validation loss!\n")

    # Save report
    with open(output_path, 'w') as f:
        f.writelines(report)

    print(f"Analysis report saved to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze learning rate sweep results")
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path("cs336_basics/basics/runs/lr_sweep"),
        help="Directory containing sweep experiments"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for analysis (default: sweep_dir)"
    )

    args = parser.parse_args()

    sweep_dir = args.sweep_dir
    output_dir = args.output_dir or sweep_dir

    if not sweep_dir.exists():
        print(f"Error: Sweep directory not found: {sweep_dir}")
        return 1

    print("="*80)
    print("LEARNING RATE SWEEP ANALYSIS")
    print("="*80)
    print(f"Sweep directory: {sweep_dir}")
    print(f"Output directory: {output_dir}\n")

    # Find all experiment directories
    exp_dirs = [d for d in sweep_dir.iterdir() if d.is_dir() and d.name.startswith("lr_")]

    if not exp_dirs:
        print(f"Error: No experiment directories found in {sweep_dir}")
        return 1

    print(f"Found {len(exp_dirs)} experiments\n")

    # Load and analyze each experiment
    results = {}

    for exp_dir in sorted(exp_dirs):
        exp_name = exp_dir.name

        try:
            # Extract learning rate
            lr = extract_learning_rate(exp_name)

            print(f"Loading {exp_name} (lr={lr:.2e})...")

            # Load metrics
            metrics = load_experiment_metrics(exp_dir)

            # Check divergence
            diverged = check_divergence(metrics)

            # Analyze convergence
            stats = analyze_convergence_rate(metrics) if not diverged else None

            results[lr] = {
                'exp_name': exp_name,
                'exp_dir': exp_dir,
                'metrics': metrics,
                'diverged': diverged,
                'stats': stats,
            }

            status = "DIVERGED" if diverged else "converged"
            print(f"  Status: {status}")

        except Exception as e:
            print(f"  Error: {e}")

    print(f"\nSuccessfully loaded {len(results)} experiments\n")

    # Generate plots
    print("Generating plots...")

    plot_lr_comparison(
        results,
        output_dir / "lr_sweep_comparison.png",
    )

    plot_final_loss_vs_lr(
        results,
        output_dir / "lr_vs_final_loss.png",
    )

    # Edge of stability analysis
    print("\nPerforming edge of stability analysis...")
    edge_analysis = analyze_edge_of_stability(results)

    print(f"\nResults:")
    print(f"  Best LR: {edge_analysis['best_lr']:.2e}")
    print(f"  Best loss: {edge_analysis['best_loss']:.4f}")
    print(f"  Edge LR: {edge_analysis['edge_lr']:.2e}")
    if edge_analysis['ratio']:
        print(f"  Ratio (edge/best): {edge_analysis['ratio']:.2f}")

    # Generate report
    print("\nGenerating analysis report...")
    generate_analysis_report(
        results,
        edge_analysis,
        output_dir / "lr_sweep_analysis.md",
    )

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  - {output_dir / 'lr_sweep_comparison.png'}")
    print(f"  - {output_dir / 'lr_vs_final_loss.png'}")
    print(f"  - {output_dir / 'lr_sweep_analysis.md'}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
