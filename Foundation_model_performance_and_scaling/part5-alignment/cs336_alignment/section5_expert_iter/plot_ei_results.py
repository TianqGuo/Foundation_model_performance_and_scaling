"""Plot Expert Iteration validation accuracy and entropy curves.

Reads all eval_metrics_*.jsonl files in results/section5/ and produces:
  - Accuracy vs EI step for each run (one curve per G value)
  - Token entropy vs EI step for each run

Usage:
    uv run python cs336_alignment/section5_expert_iter/plot_ei_results.py
    uv run python cs336_alignment/section5_expert_iter/plot_ei_results.py --results results/section5
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def load_metrics(results_dir: Path) -> dict[str, list[dict]]:
    runs = {}
    for path in sorted(results_dir.glob("eval_metrics_*.jsonl")):
        run_name = path.stem.replace("eval_metrics_", "")
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if records:
            runs[run_name] = records
    return runs


def plot_accuracy(runs: dict, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for run_name, records in sorted(runs.items()):
        steps = [r["ei_step"] for r in records]
        accuracy = [r["accuracy"] for r in records]
        ax.plot(steps, accuracy, marker="o", markersize=5, label=run_name)

    ax.set_xlabel("EI step")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Expert Iteration — Validation Accuracy")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = output_dir / "ei_accuracy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_entropy(runs: dict, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for run_name, records in sorted(runs.items()):
        steps = [r["ei_step"] for r in records]
        entropy = [r["avg_token_entropy"] for r in records]
        ax.plot(steps, entropy, marker="o", markersize=5, label=run_name)

    ax.set_xlabel("EI step")
    ax.set_ylabel("Avg token entropy (nats)")
    ax.set_title("Expert Iteration — Token Entropy")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = output_dir / "ei_entropy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_rollout_size(runs: dict, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for run_name, records in sorted(runs.items()):
        if "n_rollout" not in records[0]:
            continue
        steps = [r["ei_step"] for r in records]
        n_rollout = [r["n_rollout"] for r in records]
        ax.plot(steps, n_rollout, marker="o", markersize=5, label=run_name)

    ax.set_xlabel("EI step")
    ax.set_ylabel("Filtered rollout examples")
    ax.set_title("Expert Iteration — Rollout Dataset Size per Step")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = output_dir / "ei_rollout_size.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def print_summary(runs: dict) -> None:
    print("\nFinal validation accuracy per run:")
    print(f"  {'Run':<20} {'Steps':>6}  {'Final acc':>10}  {'Peak acc':>10}")
    print("  " + "-" * 52)
    for run_name, records in sorted(runs.items()):
        accuracies = [r["accuracy"] for r in records]
        print(
            f"  {run_name:<20} {records[-1]['ei_step']:>6}  "
            f"{accuracies[-1]:>9.1%}  {max(accuracies):>9.1%}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot EI accuracy and entropy curves")
    parser.add_argument("--results", default="results/section5")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results_dir = Path(args.results)
    output_dir = Path(args.output) if args.output else results_dir

    runs = load_metrics(results_dir)
    if not runs:
        print(f"No eval_metrics_*.jsonl files found in {results_dir}")
        print("Run training first: ./part_5_5.sh --train-only")
        return

    print(f"Found {len(runs)} run(s): {', '.join(sorted(runs))}")
    print_summary(runs)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_accuracy(runs, output_dir)
    plot_entropy(runs, output_dir)
    plot_rollout_size(runs, output_dir)


if __name__ == "__main__":
    main()