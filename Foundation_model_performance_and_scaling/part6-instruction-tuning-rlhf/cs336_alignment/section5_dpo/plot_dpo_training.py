"""Plot DPO training curves from results/section5/.

Reads train_metrics_<run_name>.jsonl and produces:
  dpo_loss_curve.png          — training loss over optimizer steps
  dpo_reward_accuracy.png     — implicit reward accuracy on validation set
  dpo_eval_summary.png        — all 4 benchmarks: baseline vs SFT vs DPO
  baseline_accuracy_bar.png   — MMLU + GSM8K comparison (all three models)
  alpaca_eval_winrate.png     — AlpacaEval winrate comparison
  sst_safety_by_category.png  — SST safety rate comparison

Usage:
    # After part_6_5.sh completes:
    uv run python cs336_alignment/section5_dpo/plot_dpo_training.py

    # Custom paths:
    uv run python cs336_alignment/section5_dpo/plot_dpo_training.py \\
        --results results/section5 \\
        --run-name dpo_hh
"""

import argparse
import json
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"baseline": "#4C72B0", "sft": "#DD8452", "dpo": "#55A868"}
LABELS = {"baseline": "Zero-shot baseline", "sft": "SFT", "dpo": "SFT + DPO"}


# ── loaders ──────────────────────────────────────────────────────────────────

def load_metrics(results_dir: Path, run_name: str) -> list[dict]:
    path = results_dir / f"train_metrics_{run_name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    return [json.loads(l) for l in open(path) if l.strip()]


def load_summary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_mmlu_summary(d: Path, suffix: str) -> dict | None:
    p = d / f"eval_mmlu_{suffix}.summary.json"
    return load_summary(p) if p.exists() else None


def load_gsm8k_summary(d: Path, suffix: str) -> dict | None:
    p = d / f"eval_gsm8k_{suffix}.summary.json"
    return load_summary(p) if p.exists() else None


def load_sst_annotated(d: Path, suffix: str) -> list[dict] | None:
    p = d / f"sst_{suffix}_annotated.jsonl"
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else None


def load_alpaca_leaderboard(d: Path) -> dict | None:
    p = d / "leaderboard.csv"
    if not p.exists():
        return None
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            if "win_rate" in row:
                wr = float(row["win_rate"])
                lc = float(row.get("length_controlled_winrate", 0) or 0)
                return {
                    "win_rate": wr / 100.0 if wr > 1 else wr,
                    "length_controlled_winrate": lc / 100.0 if lc > 1 else lc,
                }
    return None


def sst_safe_rate(records: list[dict]) -> float:
    n = sum(1 for r in records if r.get("metrics", {}).get("safe", 0) == 1.0)
    return n / len(records) if records else 0.0


# ── training curve plots ──────────────────────────────────────────────────────

def plot_loss_curve(rows: list[dict], output_dir: Path) -> None:
    steps = [r["train_step"] for r in rows]
    losses = [r["train_loss"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, losses, color=COLORS["dpo"], linewidth=1.2, alpha=0.8)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("DPO loss")
    ax.set_title("DPO Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = output_dir / "dpo_loss_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_reward_accuracy(rows: list[dict], output_dir: Path) -> None:
    val_rows = [r for r in rows if "val_reward_acc" in r]
    if not val_rows:
        print("No validation reward accuracy data — skipping.")
        return

    steps = [r["train_step"] for r in val_rows]
    accs = [r["val_reward_acc"] * 100 for r in val_rows]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, accs, color=COLORS["dpo"], linewidth=1.8,
            marker="o", markersize=4)
    ax.axhline(y=50, color="gray", linestyle="--", linewidth=1, label="50% (random)")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Reward accuracy (%)")
    ax.set_title("DPO Implicit Reward Accuracy (validation)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = output_dir / "dpo_reward_accuracy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ── benchmark comparison ──────────────────────────────────────────────────────

def plot_eval_summary(metrics: dict[str, dict], output_dir: Path) -> None:
    benchmarks = ["MMLU\nAccuracy", "GSM8K\nAccuracy",
                  "AlpacaEval\nWin Rate", "AlpacaEval\nLC Win Rate", "SST\n% Safe"]
    keys = ["mmlu_accuracy", "gsm8k_accuracy", "ae_winrate", "ae_lc_winrate", "sst_safe_rate"]
    models = [m for m in ["baseline", "sft", "dpo"] if m in metrics]

    present = [i for i in range(len(keys))
               if any(keys[i] in metrics.get(m, {}) for m in models)]
    if not present:
        return

    benchmarks = [benchmarks[i] for i in present]
    x = np.arange(len(benchmarks))
    width = 0.25
    offsets = np.linspace(-(len(models)-1)/2, (len(models)-1)/2, len(models)) * width

    fig, ax = plt.subplots(figsize=(max(8, len(benchmarks)*1.5), 4))
    for model, offset in zip(models, offsets):
        vals = [metrics[model].get(keys[i], 0) * 100 for i in present]
        bars = ax.bar(x + offset, vals, width, label=LABELS[model],
                      color=COLORS[model], alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{v:.1f}%", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks)
    ax.set_ylabel("Score (%)")
    ax.set_title("Benchmark Results — Baseline vs SFT vs DPO")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out = output_dir / "dpo_eval_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def print_summary(rows: list[dict], metrics: dict[str, dict]) -> None:
    last = rows[-1] if rows else {}
    val_rows = [r for r in rows if "val_reward_acc" in r]
    best_acc = max((r["val_reward_acc"] for r in val_rows), default=None)

    print("\n" + "=" * 60)
    print(f"  DPO training: {last.get('train_step', '?')} steps  "
          f"final loss={last.get('train_loss', '?'):.4f}")
    if best_acc is not None:
        print(f"  Best val reward accuracy: {best_acc:.1%}")
    print()
    print(f"  {'Benchmark':<28} {'Baseline':>10} {'SFT':>10} {'DPO':>10}")
    print("  " + "-" * 58)
    for label, key in [
        ("MMLU accuracy", "mmlu_accuracy"),
        ("GSM8K accuracy", "gsm8k_accuracy"),
        ("AlpacaEval win rate", "ae_winrate"),
        ("AlpacaEval LC win rate", "ae_lc_winrate"),
        ("SST % safe", "sst_safe_rate"),
    ]:
        row = []
        for m in ["baseline", "sft", "dpo"]:
            v = metrics.get(m, {}).get(key)
            row.append(f"{v:.1%}" if v is not None else "—")
        print(f"  {label:<28} {row[0]:>10} {row[1]:>10} {row[2]:>10}")
    print("=" * 60 + "\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DPO training and evaluation results")
    parser.add_argument("--results", type=Path, default=Path("results/section5"))
    parser.add_argument("--results-section2", type=Path, default=Path("results/section2"))
    parser.add_argument("--results-section4", type=Path, default=Path("results/section4"))
    parser.add_argument("--run-name", default="dpo_hh")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    s5 = args.results
    s2 = args.results_section2
    s4 = args.results_section4
    output_dir = args.output or s5
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training curves
    rows = load_metrics(s5, args.run_name)
    print(f"Loaded {len(rows)} training rows")
    plot_loss_curve(rows, output_dir)
    plot_reward_accuracy(rows, output_dir)

    # Benchmark metrics for all three model variants
    metrics: dict[str, dict] = {}
    for model, d, suffix in [("baseline", s2, "baseline"), ("sft", s4, "sft"), ("dpo", s5, "dpo")]:
        m: dict = {}
        mmlu = load_mmlu_summary(d, suffix)
        gsm = load_gsm8k_summary(d, suffix)
        ae = load_alpaca_leaderboard(d)
        sst = load_sst_annotated(d, suffix)
        if mmlu:
            m["mmlu_accuracy"] = mmlu["accuracy"]
        if gsm:
            m["gsm8k_accuracy"] = gsm["accuracy"]
        if ae:
            m["ae_winrate"] = ae["win_rate"]
            m["ae_lc_winrate"] = ae.get("length_controlled_winrate", 0) or 0
        if sst:
            m["sst_safe_rate"] = sst_safe_rate(sst)
        if m:
            metrics[model] = m

    print_summary(rows, metrics)
    plot_eval_summary(metrics, output_dir)

    # Full three-model comparison plots via plot_zero_shot_results
    print("Generating full comparison plots ...")
    import subprocess, sys
    subprocess.run([
        sys.executable, "-m", "cs336_alignment.section2_zero_shot.plot_zero_shot_results",
        "--results-section2", str(s2),
        "--results-section4", str(s4),
        "--results-section5", str(s5),
        "--output", str(output_dir),
    ], check=True)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
