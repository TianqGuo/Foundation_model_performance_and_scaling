"""Plot GRPO training results from eval metrics JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(path: Path) -> list[dict]:
    metrics = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                metrics.append(json.loads(line))
    return metrics


def plot_accuracy_curves(runs: dict[str, list[dict]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, metrics in runs.items():
        steps = [m["grpo_step"] for m in metrics]
        acc = [m["accuracy"] for m in metrics]
        ax.plot(steps, acc, marker="o", markersize=4, label=label)
    ax.set_xlabel("GRPO Step")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("GRPO Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Results saved to {output_path}")


def plot_reward_curves(runs: dict[str, list[dict]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, metrics in runs.items():
        steps = [m["grpo_step"] for m in metrics]
        reward = [m["avg_reward"] for m in metrics]
        ax.plot(steps, reward, marker="o", markersize=4, label=label)
    ax.set_xlabel("GRPO Step")
    ax.set_ylabel("Validation Avg Reward")
    ax.set_title("GRPO Validation Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Results saved to {output_path}")


def plot_entropy_curve(metrics: list[dict], output_path: Path) -> None:
    steps = [m["grpo_step"] for m in metrics]
    entropy = [m.get("avg_token_entropy", 0.0) for m in metrics]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, entropy, marker="o", markersize=4, color="steelblue")
    ax.set_xlabel("GRPO Step")
    ax.set_ylabel("Avg Token Entropy")
    ax.set_title("Token Entropy over GRPO Training")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Results saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GRPO training results")
    parser.add_argument("--results_dir", default="results/section7",
                        help="Directory containing eval_metrics_*.jsonl files")
    parser.add_argument("--output_dir", default="results/section7",
                        help="Directory to save plots")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_files = sorted(results_dir.glob("eval_metrics_*.jsonl"))
    if not metric_files:
        print(f"No eval_metrics_*.jsonl files found in {results_dir}")
        return

    runs: dict[str, list[dict]] = {}
    for f in metric_files:
        label = f.stem.replace("eval_metrics_", "")
        data = load_metrics(f)
        if data:
            runs[label] = data
            print(f"Loaded {len(data)} eval points from {f.name}")

    if not runs:
        print("No data to plot.")
        return

    plot_accuracy_curves(runs, output_dir / "grpo_accuracy.png")
    plot_reward_curves(runs, output_dir / "grpo_reward.png")

    # Entropy for first run only (or all if they have it)
    first_label, first_data = next(iter(runs.items()))
    if any("avg_token_entropy" in m for m in first_data):
        plot_entropy_curve(first_data, output_dir / f"grpo_entropy_{first_label}.png")


if __name__ == "__main__":
    main()
