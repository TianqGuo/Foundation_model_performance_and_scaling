"""
Batch Size Sweep Analysis and Visualization
Generates comparative plots from batch size experiments
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# Load data
sweep_dir = Path("cs336_basics/basics/runs/batch_size_sweep")

batch_sizes = [1, 8, 32, 64, 128, 256]
results = {}

for bs in batch_sizes:
    summary_path = sweep_dir / f"batch_{bs}" / "summary.json"
    with open(summary_path) as f:
        results[bs] = json.load(f)

# Extract metrics
batch_sizes_completed = list(results.keys())
final_val_losses = [results[bs]["statistics"]["val_loss"]["final"] for bs in batch_sizes_completed]
best_val_losses = [results[bs]["statistics"]["val_loss"]["best"] for bs in batch_sizes_completed]
final_train_losses = [results[bs]["statistics"]["train_loss"]["final"] for bs in batch_sizes_completed]
training_times = [results[bs]["statistics"]["total_wallclock_hours"] for bs in batch_sizes_completed]
total_steps = [results[bs]["statistics"]["total_steps"] for bs in batch_sizes_completed]

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# 1. Validation Loss vs Batch Size
ax1 = plt.subplot(2, 3, 1)
ax1.plot(batch_sizes_completed, best_val_losses, 'o-', linewidth=2, markersize=8, label='Best Val Loss')
ax1.plot(batch_sizes_completed, final_val_losses, 's--', linewidth=2, markersize=8, label='Final Val Loss', alpha=0.7)
ax1.set_xlabel('Batch Size', fontsize=12)
ax1.set_ylabel('Validation Loss', fontsize=12)
ax1.set_title('Validation Loss vs Batch Size', fontsize=14, fontweight='bold')
ax1.set_xscale('log', base=2)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.axhline(y=min(best_val_losses), color='green', linestyle=':', alpha=0.5, label='Best Overall')

# Add annotations for best
best_idx = np.argmin(best_val_losses)
ax1.annotate(f'Best: {best_val_losses[best_idx]:.4f}\n(batch={batch_sizes_completed[best_idx]})',
             xy=(batch_sizes_completed[best_idx], best_val_losses[best_idx]),
             xytext=(20, 20), textcoords='offset points',
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

# 2. Training Time vs Batch Size
ax2 = plt.subplot(2, 3, 2)
colors = plt.cm.viridis(np.linspace(0, 1, len(batch_sizes_completed)))
bars = ax2.bar(range(len(batch_sizes_completed)), training_times, color=colors, alpha=0.7, edgecolor='black')
ax2.set_xlabel('Batch Size', fontsize=12)
ax2.set_ylabel('Training Time (hours)', fontsize=12)
ax2.set_title('Training Time vs Batch Size', fontsize=14, fontweight='bold')
ax2.set_xticks(range(len(batch_sizes_completed)))
ax2.set_xticklabels(batch_sizes_completed)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, time) in enumerate(zip(bars, training_times)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{time:.2f}h',
             ha='center', va='bottom', fontsize=9)

# 3. Number of Gradient Updates
ax3 = plt.subplot(2, 3, 3)
ax3.plot(batch_sizes_completed, total_steps, 'o-', linewidth=2, markersize=8, color='coral')
ax3.set_xlabel('Batch Size', fontsize=12)
ax3.set_ylabel('Total Gradient Updates', fontsize=12)
ax3.set_title('Gradient Updates vs Batch Size', fontsize=14, fontweight='bold')
ax3.set_xscale('log', base=2)
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3)

# Add annotations
for bs, steps in zip(batch_sizes_completed, total_steps):
    ax3.annotate(f'{steps:,}', xy=(bs, steps), xytext=(0, 10),
                textcoords='offset points', ha='center', fontsize=8)

# 4. Training Loss vs Validation Loss
ax4 = plt.subplot(2, 3, 4)
ax4.plot(batch_sizes_completed, final_train_losses, 'o-', linewidth=2, markersize=8, label='Train Loss', color='blue')
ax4.plot(batch_sizes_completed, final_val_losses, 's-', linewidth=2, markersize=8, label='Val Loss', color='red')
ax4.set_xlabel('Batch Size', fontsize=12)
ax4.set_ylabel('Loss', fontsize=12)
ax4.set_title('Train vs Validation Loss', fontsize=14, fontweight='bold')
ax4.set_xscale('log', base=2)
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10)

# 5. Generalization Gap (Train-Val)
ax5 = plt.subplot(2, 3, 5)
gaps = [train - val for train, val in zip(final_train_losses, final_val_losses)]
colors_gap = ['green' if gap < 0 else 'red' for gap in gaps]
bars_gap = ax5.bar(range(len(batch_sizes_completed)), gaps, color=colors_gap, alpha=0.6, edgecolor='black')
ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax5.set_xlabel('Batch Size', fontsize=12)
ax5.set_ylabel('Train Loss - Val Loss', fontsize=12)
ax5.set_title('Generalization Gap', fontsize=14, fontweight='bold')
ax5.set_xticks(range(len(batch_sizes_completed)))
ax5.set_xticklabels(batch_sizes_completed)
ax5.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, gap in zip(bars_gap, gaps):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
             f'{gap:.3f}',
             ha='center', va='bottom' if gap > 0 else 'top', fontsize=9)

# 6. Efficiency: Val Loss vs Training Time
ax6 = plt.subplot(2, 3, 6)
scatter = ax6.scatter(training_times, best_val_losses, s=200, c=batch_sizes_completed,
                     cmap='viridis', alpha=0.7, edgecolors='black', linewidth=2)
ax6.set_xlabel('Training Time (hours)', fontsize=12)
ax6.set_ylabel('Best Validation Loss', fontsize=12)
ax6.set_title('Efficiency: Quality vs Speed', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax6)
cbar.set_label('Batch Size', fontsize=10)

# Annotate points
for bs, time, loss in zip(batch_sizes_completed, training_times, best_val_losses):
    ax6.annotate(f'BS={bs}', xy=(time, loss), xytext=(5, 5),
                textcoords='offset points', fontsize=8)

# Highlight best performer
best_idx = np.argmin(best_val_losses)
ax6.scatter([training_times[best_idx]], [best_val_losses[best_idx]],
           s=400, facecolors='none', edgecolors='red', linewidth=3)

plt.suptitle('Batch Size Sweep Analysis: 22.7M Parameter Transformer on TinyStories',
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig('batch_size_sweep_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: batch_size_sweep_analysis.png")

# Create a summary table
print("\n" + "="*80)
print("BATCH SIZE SWEEP SUMMARY")
print("="*80)
print(f"{'Batch':<8} {'Steps':<10} {'Time (h)':<10} {'Train Loss':<12} {'Val Loss':<12} {'Best Val':<12}")
print("-"*80)
for bs, steps, time, train_loss, val_loss, best_val in zip(
    batch_sizes_completed, total_steps, training_times,
    final_train_losses, final_val_losses, best_val_losses):
    print(f"{bs:<8} {steps:<10,} {time:<10.2f} {train_loss:<12.4f} {val_loss:<12.4f} {best_val:<12.4f}")
print("="*80)

# Identify best configurations
best_quality_idx = np.argmin(best_val_losses)
fastest_idx = np.argmin(training_times)

print(f"\nBest Quality: Batch Size {batch_sizes_completed[best_quality_idx]} "
      f"(Val Loss: {best_val_losses[best_quality_idx]:.4f}, Time: {training_times[best_quality_idx]:.2f}h)")
print(f"Fastest: Batch Size {batch_sizes_completed[fastest_idx]} "
      f"(Val Loss: {best_val_losses[fastest_idx]:.4f}, Time: {training_times[fastest_idx]:.2f}h)")

# Calculate efficiency score (lower is better): normalized val loss + normalized time
norm_val_loss = (np.array(best_val_losses) - min(best_val_losses)) / (max(best_val_losses) - min(best_val_losses))
norm_time = (np.array(training_times) - min(training_times)) / (max(training_times) - min(training_times))
efficiency_score = norm_val_loss + norm_time
best_efficiency_idx = np.argmin(efficiency_score)

print(f"Best Efficiency (quality + speed): Batch Size {batch_sizes_completed[best_efficiency_idx]} "
      f"(Val Loss: {best_val_losses[best_efficiency_idx]:.4f}, Time: {training_times[best_efficiency_idx]:.2f}h)")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)
