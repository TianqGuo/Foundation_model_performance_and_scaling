"""Plot SFT training curves from results/section3/.

Reads train_metrics_<run_name>.jsonl and produces:
  sft_loss_curve.png     — train loss (and val loss if present) over optimizer steps
  sft_lr_schedule.png    — learning rate schedule over optimizer steps

Usage:
    uv run python cs336_alignment/section3_sft/plot_sft_training.py
    uv run python cs336_alignment/section3_sft/plot_sft_training.py \\
        --results results/section3 \\
        --run-name sft_ultrachat \\
        --output results/section3
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(results_dir: Path, run_name: str) -> list[dict]:
    path = results_dir / f"train_metrics_{run_name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def plot_loss_curve(rows: list[dict], output_dir: Path, run_name: str) -> None:
    steps       = [r["train_step"] for r in rows]
    train_loss  = [r["train_loss"] for r in rows]
    val_steps   = [r["train_step"] for r in rows if "val_loss" in r]
    val_loss    = [r["val_loss"]   for r in rows if "val_loss" in r]

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(steps, train_loss, color="#4C72B0", linewidth=1.2, alpha=0.8, label="Train loss")
    if val_steps:
        ax.plot(val_steps, val_loss, color="#DD8452", linewidth=1.8,
                marker="o", markersize=4, label="Val loss")

    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"SFT Training Loss — {run_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = output_dir / "sft_loss_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_lr_schedule(rows: list[dict], output_dir: Path, run_name: str) -> None:
    steps = [r["train_step"] for r in rows]
    lrs   = [r["lr"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(steps, lrs, color="#55A868", linewidth=1.5)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Learning rate")
    ax.set_title(f"LR Schedule (cosine decay + warmup) — {run_name}")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = output_dir / "sft_lr_schedule.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def print_summary(rows: list[dict], results_dir: Path, run_name: str) -> None:
    if not rows:
        print("No training rows found.")
        return

    total_steps = rows[-1]["train_step"]
    final_train_loss = rows[-1]["train_loss"]
    val_rows = [r for r in rows if "val_loss" in r]
    final_val_loss = val_rows[-1]["val_loss"] if val_rows else None

    # Total elapsed time from final_val json if present, else from last metrics row
    elapsed_human = None
    final_json = results_dir / f"final_val_{run_name}.json"
    if final_json.exists():
        with open(final_json) as f:
            data = json.load(f)
        elapsed_human = data.get("total_elapsed_human")
        final_val_loss = data.get("val_loss", final_val_loss)
    if elapsed_human is None and "elapsed_sec" in rows[-1]:
        from cs336_alignment.section3_sft.train_sft import _fmt_time
        elapsed_human = _fmt_time(rows[-1]["elapsed_sec"])

    print("\n" + "=" * 50)
    print(f"  Run:              {run_name}")
    print(f"  Total steps:      {total_steps}")
    print(f"  Final train loss: {final_train_loss:.4f}")
    if final_val_loss is not None:
        print(f"  Final val loss:   {final_val_loss:.4f}")
    if elapsed_human:
        print(f"  Total time:       {elapsed_human}")
    print("=" * 50 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SFT training curves")
    parser.add_argument("--results", type=Path, default=Path("results/section3"),
                        help="Directory containing train_metrics_<run_name>.jsonl")
    parser.add_argument("--run-name", default="sft_ultrachat",
                        help="Run name used when training (matches --run_name in train_sft.py)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory for plots (defaults to --results)")
    args = parser.parse_args()

    output_dir = args.output or args.results
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_metrics(args.results, args.run_name)
    print(f"Loaded {len(rows)} metric rows from {args.results}")

    print_summary(rows, args.results, args.run_name)
    plot_loss_curve(rows, output_dir, args.run_name)
    plot_lr_schedule(rows, output_dir, args.run_name)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
