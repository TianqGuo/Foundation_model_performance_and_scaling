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


def _make_fig(title: str, xlabel: str, ylabel: str) -> tuple:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return fig, ax


def _save(fig, ax, path: Path) -> None:
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Results saved to {path}")


def plot_metric_curves(
    runs: dict[str, list[dict]],
    key: str,
    title: str,
    ylabel: str,
    output_path: Path,
    x_key: str = "grpo_step",
    xlabel: str = "GRPO Step",
) -> None:
    fig, ax = _make_fig(title, xlabel, ylabel)
    for label, metrics in runs.items():
        xs = [m[x_key] for m in metrics if x_key in m and key in m]
        ys = [m[key] for m in metrics if x_key in m and key in m]
        if xs:
            ax.plot(xs, ys, marker="o", markersize=4, label=label)
    _save(fig, ax, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GRPO training results")
    parser.add_argument("--results_dir", default="results/section8",
                        help="Directory containing eval_metrics_*.jsonl files")
    parser.add_argument("--output_dir", default="results/section8",
                        help="Directory to save plots")
    parser.add_argument("--x_axis", default="grpo_step",
                        choices=["grpo_step", "eval_step"],
                        help="X-axis for plots (grpo_step or eval_step)")
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

    x_key = args.x_axis
    xlabel = "GRPO Step" if x_key == "grpo_step" else "Eval Step"

    plot_metric_curves(runs, "accuracy", "Validation Accuracy", "Accuracy",
                       output_dir / "grpo_accuracy.png", x_key, xlabel)

    plot_metric_curves(runs, "avg_reward", "Validation Avg Reward", "Avg Reward",
                       output_dir / "grpo_reward.png", x_key, xlabel)

    plot_metric_curves(runs, "avg_token_entropy", "Token Entropy over Training", "Avg Token Entropy",
                       output_dir / "grpo_entropy.png", x_key, xlabel)

    plot_metric_curves(runs, "avg_response_length", "Response Length over Training", "Avg Response Length (tokens)",
                       output_dir / "grpo_response_length.png", x_key, xlabel)


if __name__ == "__main__":
    main()
