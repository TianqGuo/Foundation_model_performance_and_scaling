"""Plot SFT validation accuracy curves from eval_metrics_*.jsonl files.

Reads all eval_metrics_*.jsonl files in results/section4/ and produces:
  - Accuracy vs training step for each dataset size (ablation plot)
  - Accuracy comparison: full dataset vs filtered dataset

Usage:
    uv run python cs336_alignment/section4_sft/plot_sft_results.py
    uv run python cs336_alignment/section4_sft/plot_sft_results.py --output results/section4
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# Display order and labels for the ablation runs
ABLATION_ORDER = ["sft_n128", "sft_n256", "sft_n512", "sft_n1024", "sft_full"]
ABLATION_LABELS = {
    "sft_n128": "128",
    "sft_n256": "256",
    "sft_n512": "512",
    "sft_n1024": "1024",
    "sft_full": "Full dataset",
}
FILTER_RUNS = {"sft_full", "sft_filtered"}
FILTER_LABELS = {
    "sft_full": "Full (unfiltered)",
    "sft_filtered": "Correct-answer filtered",
}


def load_metrics(results_dir: Path) -> dict[str, list[dict]]:
    """Load all eval_metrics_*.jsonl files. Returns {run_name: [step_record, ...]}."""
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


def plot_ablation(runs: dict, output_dir: Path) -> None:
    """Accuracy vs train_step for each dataset size."""
    fig, ax = plt.subplots(figsize=(8, 5))

    plotted = False
    for run_name in ABLATION_ORDER:
        if run_name not in runs:
            continue
        records = runs[run_name]
        steps = [r["train_step"] for r in records]
        accuracy = [r["accuracy"] for r in records]
        label = ABLATION_LABELS.get(run_name, run_name)
        ax.plot(steps, accuracy, marker="o", markersize=3, label=label)
        plotted = True

    if not plotted:
        print("No ablation runs found — skipping ablation plot.")
        plt.close(fig)
        return

    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("SFT — Validation Accuracy by Dataset Size")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(title="Training examples", loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = output_dir / "sft_ablation_accuracy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_filtered_comparison(runs: dict, output_dir: Path) -> None:
    """Accuracy vs train_step: full dataset vs filtered dataset."""
    fig, ax = plt.subplots(figsize=(7, 4))

    plotted = False
    for run_name in ["sft_full", "sft_filtered"]:
        if run_name not in runs:
            continue
        records = runs[run_name]
        steps = [r["train_step"] for r in records]
        accuracy = [r["accuracy"] for r in records]
        label = FILTER_LABELS.get(run_name, run_name)
        ax.plot(steps, accuracy, marker="o", markersize=3, label=label)
        plotted = True

    if not plotted:
        print("No filtered-vs-full runs found — skipping filter comparison plot.")
        plt.close(fig)
        return

    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("SFT — Full Dataset vs Correct-Answer Filtered")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = output_dir / "sft_filtered_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def print_summary(runs: dict) -> None:
    """Print final accuracy for each run."""
    print("\nFinal validation accuracy per run:")
    print(f"  {'Run':<25} {'Steps':>6}  {'Final acc':>10}  {'Peak acc':>10}")
    print("  " + "-" * 57)
    for run_name in ABLATION_ORDER + ["sft_filtered"]:
        if run_name not in runs:
            continue
        records = runs[run_name]
        accuracies = [r["accuracy"] for r in records]
        print(
            f"  {run_name:<25} {records[-1]['train_step']:>6}  "
            f"{accuracies[-1]:>9.1%}  {max(accuracies):>9.1%}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SFT accuracy curves")
    parser.add_argument("--results", default="results/section4",
                        help="Directory containing eval_metrics_*.jsonl files")
    parser.add_argument("--output", default=None,
                        help="Output directory for plots (defaults to --results)")
    args = parser.parse_args()

    results_dir = Path(args.results)
    output_dir = Path(args.output) if args.output else results_dir

    runs = load_metrics(results_dir)
    if not runs:
        print(f"No eval_metrics_*.jsonl files found in {results_dir}")
        print("Run training first: ./part_5_4.sh --train-only")
        return

    print(f"Found {len(runs)} run(s): {', '.join(sorted(runs))}")
    print_summary(runs)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_ablation(runs, output_dir)
    plot_filtered_comparison(runs, output_dir)


if __name__ == "__main__":
    main()