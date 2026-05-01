#!/usr/bin/env python3
"""
Analyze and visualize ablation experiment results.

This script compares the performance of different ablations and generates
comparative plots and analysis.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')


def load_ablation_results(base_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load results from all ablation experiments."""
    ablations = ["baseline", "no_layer_norm", "post_norm", "no_position_emb", "silu_ffn"]
    results = {}

    for ablation in ablations:
        summary_path = base_dir / ablation / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                results[ablation] = json.load(f)
            print(f"✓ Loaded {ablation}")
        else:
            print(f"✗ Missing {ablation} (expected at {summary_path})")

    return results


def create_comparison_plots(results: Dict[str, Dict], output_path: Path):
    """Generate comparative visualizations for ablations."""
    fig = plt.figure(figsize=(16, 10))

    # Extract metrics
    ablation_names = list(results.keys())
    friendly_names = {
        "baseline": "Baseline\n(Pre-norm + SwiGLU + RoPE)",
        "no_layer_norm": "No Layer Norm",
        "post_norm": "Post-norm",
        "no_position_emb": "NoPE\n(No Position Emb)",
        "silu_ffn": "SiLU FFN\n(No Gating)",
    }

    best_val_losses = [results[abl]["statistics"]["val_loss"]["best"] for abl in ablation_names]
    final_train_losses = [results[abl]["statistics"]["train_loss"]["final"] for abl in ablation_names]
    training_times = [results[abl]["statistics"]["total_wallclock_hours"] for abl in ablation_names]
    total_steps = [results[abl]["statistics"]["total_steps"] for abl in ablation_names]

    # 1. Best Validation Loss Comparison
    ax1 = plt.subplot(2, 3, 1)
    colors = ['#2E7D32' if abl == 'baseline' else '#1976D2' for abl in ablation_names]
    bars = ax1.bar(range(len(ablation_names)), best_val_losses, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Ablation', fontsize=11)
    ax1.set_ylabel('Best Validation Loss', fontsize=11)
    ax1.set_title('Validation Loss Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(ablation_names)))
    ax1.set_xticklabels([friendly_names.get(abl, abl) for abl in ablation_names],
                         rotation=15, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, best_val_losses)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    # Add baseline reference line
    baseline_val = best_val_losses[0]
    ax1.axhline(y=baseline_val, color='green', linestyle=':', alpha=0.5, label='Baseline')
    ax1.legend(fontsize=9)

    # 2. Training Time Comparison
    ax2 = plt.subplot(2, 3, 2)
    colors_time = plt.cm.viridis(np.linspace(0, 1, len(ablation_names)))
    bars_time = ax2.bar(range(len(ablation_names)), training_times,
                        color=colors_time, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Ablation', fontsize=11)
    ax2.set_ylabel('Training Time (hours)', fontsize=11)
    ax2.set_title('Training Time Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(ablation_names)))
    ax2.set_xticklabels([friendly_names.get(abl, abl) for abl in ablation_names],
                         rotation=15, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, time in zip(bars_time, training_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{time:.2f}h', ha='center', va='bottom', fontsize=9)

    # 3. Performance Degradation (relative to baseline)
    ax3 = plt.subplot(2, 3, 3)
    degradation = [(val - baseline_val) / baseline_val * 100 for val in best_val_losses]
    colors_deg = ['green' if d <= 0 else 'red' if d > 10 else 'orange' for d in degradation]
    bars_deg = ax3.bar(range(len(ablation_names)), degradation,
                       color=colors_deg, alpha=0.7, edgecolor='black')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_xlabel('Ablation', fontsize=11)
    ax3.set_ylabel('Performance Degradation (%)', fontsize=11)
    ax3.set_title('Degradation vs. Baseline', fontsize=13, fontweight='bold')
    ax3.set_xticks(range(len(ablation_names)))
    ax3.set_xticklabels([friendly_names.get(abl, abl) for abl in ablation_names],
                         rotation=15, ha='right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    for bar, deg in zip(bars_deg, degradation):
        height = bar.get_height()
        va = 'bottom' if deg > 0 else 'top'
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{deg:+.1f}%', ha='center', va=va, fontsize=9)

    # 4. Train vs Val Loss
    ax4 = plt.subplot(2, 3, 4)
    x_pos = np.arange(len(ablation_names))
    width = 0.35
    bars1 = ax4.bar(x_pos - width/2, final_train_losses, width, label='Train Loss',
                    color='steelblue', alpha=0.7, edgecolor='black')
    bars2 = ax4.bar(x_pos + width/2, best_val_losses, width, label='Val Loss',
                    color='coral', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Ablation', fontsize=11)
    ax4.set_ylabel('Loss', fontsize=11)
    ax4.set_title('Final Train vs Best Val Loss', fontsize=13, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([friendly_names.get(abl, abl) for abl in ablation_names],
                         rotation=15, ha='right', fontsize=9)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Generalization Gap
    ax5 = plt.subplot(2, 3, 5)
    gaps = [train - val for train, val in zip(final_train_losses, best_val_losses)]
    colors_gap = ['green' if gap < 0 else 'red' if gap > 0.1 else 'orange' for gap in gaps]
    bars_gap = ax5.bar(range(len(ablation_names)), gaps,
                       color=colors_gap, alpha=0.7, edgecolor='black')
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax5.set_xlabel('Ablation', fontsize=11)
    ax5.set_ylabel('Train Loss - Val Loss', fontsize=11)
    ax5.set_title('Generalization Gap', fontsize=13, fontweight='bold')
    ax5.set_xticks(range(len(ablation_names)))
    ax5.set_xticklabels([friendly_names.get(abl, abl) for abl in ablation_names],
                         rotation=15, ha='right', fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')

    for bar, gap in zip(bars_gap, gaps):
        height = bar.get_height()
        va = 'bottom' if gap > 0 else 'top'
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{gap:.3f}', ha='center', va=va, fontsize=9)

    # 6. Efficiency (Val Loss vs Time)
    ax6 = plt.subplot(2, 3, 6)
    scatter = ax6.scatter(training_times, best_val_losses, s=200,
                         c=range(len(ablation_names)), cmap='viridis',
                         alpha=0.7, edgecolors='black', linewidth=2)
    ax6.set_xlabel('Training Time (hours)', fontsize=11)
    ax6.set_ylabel('Best Validation Loss', fontsize=11)
    ax6.set_title('Efficiency: Quality vs Speed', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # Annotate points
    for i, (time, loss, name) in enumerate(zip(training_times, best_val_losses, ablation_names)):
        short_name = name.replace('_', '\n')
        ax6.annotate(short_name, xy=(time, loss), xytext=(5, 5),
                    textcoords='offset points', fontsize=7)

    # Highlight baseline
    baseline_idx = ablation_names.index('baseline')
    ax6.scatter([training_times[baseline_idx]], [best_val_losses[baseline_idx]],
               s=400, facecolors='none', edgecolors='red', linewidth=3)

    plt.suptitle('Transformer Ablation Study Results', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plots saved to: {output_path}")


def print_summary_table(results: Dict[str, Dict]):
    """Print formatted comparison table."""
    print("\n" + "="*100)
    print("ABLATION STUDY SUMMARY")
    print("="*100)

    headers = ["Ablation", "Steps", "Time (h)", "Train Loss", "Val Loss", "Best Val", "Δ vs Baseline"]
    print(f"{headers[0]:<20} {headers[1]:<8} {headers[2]:<10} {headers[3]:<12} {headers[4]:<12} {headers[5]:<12} {headers[6]:<15}")
    print("-"*100)

    baseline_val = results["baseline"]["statistics"]["val_loss"]["best"]

    for name, data in results.items():
        stats = data["statistics"]
        steps = stats["total_steps"]
        time = stats["total_wallclock_hours"]
        train_loss = stats["train_loss"]["final"]
        val_loss = stats["val_loss"]["final"]
        best_val = stats["val_loss"]["best"]
        delta = ((best_val - baseline_val) / baseline_val) * 100

        print(f"{name:<20} {steps:<8,} {time:<10.2f} {train_loss:<12.4f} {val_loss:<12.4f} "
              f"{best_val:<12.4f} {delta:+.2f}%")

    print("="*100)


def generate_analysis_report(results: Dict[str, Dict], output_path: Path):
    """Generate detailed analysis report."""
    baseline_val = results["baseline"]["statistics"]["val_loss"]["best"]

    report = []
    report.append("=" * 80)
    report.append("ABLATION STUDY ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")

    # Summary statistics
    report.append("## Overall Statistics")
    report.append("")
    for name, data in results.items():
        stats = data["statistics"]
        desc = data["config"]["description"]
        report.append(f"### {name}")
        report.append(f"Description: {desc}")
        report.append(f"  - Best validation loss: {stats['val_loss']['best']:.4f}")
        report.append(f"  - Final train loss: {stats['train_loss']['final']:.4f}")
        report.append(f"  - Training time: {stats['total_wallclock_hours']:.2f} hours")
        report.append(f"  - Total steps: {stats['total_steps']:,}")
        delta = ((stats['val_loss']['best'] - baseline_val) / baseline_val) * 100
        report.append(f"  - Δ vs baseline: {delta:+.2f}%")
        report.append("")

    # Key findings
    report.append("## Key Findings")
    report.append("")

    # Find best and worst
    sorted_by_val = sorted(results.items(), key=lambda x: x[1]["statistics"]["val_loss"]["best"])
    best_abl = sorted_by_val[0][0]
    worst_abl = sorted_by_val[-1][0]

    report.append(f"1. **Best performing ablation:** {best_abl}")
    report.append(f"   - Validation loss: {sorted_by_val[0][1]['statistics']['val_loss']['best']:.4f}")
    report.append("")

    report.append(f"2. **Worst performing ablation:** {worst_abl}")
    report.append(f"   - Validation loss: {sorted_by_val[-1][1]['statistics']['val_loss']['best']:.4f}")
    report.append(f"   - Degradation: {((sorted_by_val[-1][1]['statistics']['val_loss']['best'] - baseline_val) / baseline_val) * 100:.1f}%")
    report.append("")

    # Component importance ranking
    report.append("3. **Component Importance Ranking (by validation loss degradation):**")
    degradations = []
    for name, data in results.items():
        if name != "baseline":
            val_loss = data["statistics"]["val_loss"]["best"]
            deg = ((val_loss - baseline_val) / baseline_val) * 100
            degradations.append((name, deg))

    degradations.sort(key=lambda x: abs(x[1]), reverse=True)
    for i, (name, deg) in enumerate(degradations, 1):
        impact = "CRITICAL" if abs(deg) > 10 else "MODERATE" if abs(deg) > 5 else "MINOR"
        report.append(f"   {i}. {name}: {deg:+.2f}% ({impact})")
    report.append("")

    # Save report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"✓ Analysis report saved to: {output_path}")

    # Print to console too
    print('\n'.join(report))


def main():
    """Main analysis function."""
    base_dir = Path("cs336_basics/basics/runs/ablations")

    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist. Run experiments first.")
        return

    # Load results
    print("Loading ablation results...")
    results = load_ablation_results(base_dir)

    if not results:
        print("No results found. Run experiments first.")
        return

    if len(results) < 2:
        print(f"Only {len(results)} experiments found. Need at least 2 for comparison.")
        return

    # Print summary table
    print_summary_table(results)

    # Generate plots
    plots_path = base_dir / "ablation_comparison.png"
    create_comparison_plots(results, plots_path)

    # Generate analysis report
    report_path = base_dir / "ablation_analysis.txt"
    generate_analysis_report(results, report_path)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
