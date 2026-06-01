"""Plot SFT evaluation results from results/section4/.

Produces section4-specific plots and triggers the full baseline vs SFT
comparison plots via plot_zero_shot_results.py.

Section4-specific outputs (saved to results/section4/):
  mmlu_subject_accuracy_sft.png   — per-subject MMLU accuracy for the SFT model
  sft_eval_summary.png            — bar chart of all four benchmark metrics

Comparison outputs (saved to results/section2/ alongside baseline plots):
  baseline_accuracy_bar.png       — MMLU + GSM8K: baseline vs SFT
  alpaca_eval_winrate.png         — AlpacaEval win rate: baseline vs SFT
  sst_safety_by_category.png      — SST safety rate: baseline vs SFT

Usage:
    # After part_6_4.sh completes:
    uv run python cs336_alignment/section4_eval/plot_sft_eval.py

    # Custom paths:
    uv run python cs336_alignment/section4_eval/plot_sft_eval.py \\
        --results-section4 results/section4 \\
        --results-section2 results/section2
"""

import argparse
import json
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"baseline": "#4C72B0", "sft": "#DD8452"}


# ── loaders (mirrors plot_zero_shot_results.py) ───────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_summary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_mmlu_records(results_dir: Path, suffix: str) -> list[dict] | None:
    path = results_dir / f"eval_mmlu_{suffix}.jsonl"
    return load_jsonl(path) if path.exists() else None


def load_mmlu_summary(results_dir: Path, suffix: str) -> dict | None:
    path = results_dir / f"eval_mmlu_{suffix}.summary.json"
    if path.exists():
        return load_summary(path)
    records = load_mmlu_records(results_dir, suffix)
    if records is None:
        return None
    n = sum(1 for r in records if r.get("correct"))
    return {"accuracy": n / len(records), "n_correct": n, "n_total": len(records)}


def load_gsm8k_summary(results_dir: Path, suffix: str) -> dict | None:
    path = results_dir / f"eval_gsm8k_{suffix}.summary.json"
    if path.exists():
        return load_summary(path)
    jsonl = results_dir / f"eval_gsm8k_{suffix}.jsonl"
    if not jsonl.exists():
        return None
    records = load_jsonl(jsonl)
    n = sum(1 for r in records if r.get("correct"))
    return {"accuracy": n / len(records), "n_correct": n, "n_total": len(records)}


def load_sst_annotated(results_dir: Path, suffix: str) -> list[dict] | None:
    path = results_dir / f"sst_{suffix}_annotated.jsonl"
    return load_jsonl(path) if path.exists() else None


def load_alpaca_leaderboard(results_dir: Path) -> dict | None:
    csv_path = results_dir / "leaderboard.csv"
    if not csv_path.exists():
        return None
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if "win_rate" in row:
                wr = float(row["win_rate"])
                lc = float(row.get("length_controlled_winrate", 0) or 0)
                # normalise from percent to fraction if needed
                return {
                    "win_rate": wr / 100.0 if wr > 1 else wr,
                    "length_controlled_winrate": lc / 100.0 if lc > 1 else lc,
                }
    return None


def sst_safe_rate(records: list[dict]) -> float:
    n_safe = sum(1 for r in records if r.get("metrics", {}).get("safe", 0) == 1.0)
    return n_safe / len(records) if records else 0.0


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_mmlu_subject_accuracy(records: list[dict], output_dir: Path) -> None:
    """Horizontal bar chart of per-subject MMLU accuracy for SFT model."""
    subject_stats: dict[str, list[bool]] = {}
    for r in records:
        subject_stats.setdefault(r.get("subject", "unknown"), []).append(r.get("correct", False))

    subjects = sorted(subject_stats, key=lambda s: sum(subject_stats[s]) / len(subject_stats[s]))
    accs = [sum(subject_stats[s]) / len(subject_stats[s]) * 100 for s in subjects]
    labels = [s.replace("_", " ").title() for s in subjects]
    mean_acc = sum(accs) / len(accs)

    fig, ax = plt.subplots(figsize=(9, max(6, len(subjects) * 0.28)))
    colors = [COLORS["sft"] if a >= 60 else COLORS["baseline"] for a in accs]
    ax.barh(labels, accs, color=colors, alpha=0.85)
    ax.axvline(x=mean_acc, color="black", linestyle="--", linewidth=1,
               label=f"Mean {mean_acc:.1f}%")
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("MMLU Per-Subject Accuracy — SFT")
    ax.set_xlim(0, 105)
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    out = output_dir / "mmlu_subject_accuracy_sft.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_summary_bar(metrics: dict, baseline_metrics: dict | None,
                     output_dir: Path) -> None:
    """Four-benchmark summary bar chart comparing baseline and SFT."""
    benchmarks = ["MMLU\nAccuracy", "GSM8K\nAccuracy",
                  "AlpacaEval\nWin Rate", "AlpacaEval\nLC Win Rate", "SST\n% Safe"]
    sft_vals = [
        metrics.get("mmlu_accuracy", 0) * 100,
        metrics.get("gsm8k_accuracy", 0) * 100,
        metrics.get("ae_winrate", 0) * 100,
        metrics.get("ae_lc_winrate", 0) * 100,
        metrics.get("sst_safe_rate", 0) * 100,
    ]
    # only include benchmarks where we have SFT data
    present = [i for i, v in enumerate(sft_vals) if v > 0]
    if not present:
        print("No SFT metrics to plot in summary bar.")
        return

    benchmarks = [benchmarks[i] for i in present]
    sft_vals = [sft_vals[i] for i in present]

    x = np.arange(len(benchmarks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(7, len(benchmarks) * 1.5), 4))

    if baseline_metrics:
        base_vals = [
            [baseline_metrics.get("mmlu_accuracy", 0) * 100,
             baseline_metrics.get("gsm8k_accuracy", 0) * 100,
             baseline_metrics.get("ae_winrate", 0) * 100,
             baseline_metrics.get("ae_lc_winrate", 0) * 100,
             baseline_metrics.get("sst_safe_rate", 0) * 100][i]
            for i in present
        ]
        bars_b = ax.bar(x - width / 2, base_vals, width, label="Zero-shot baseline",
                        color=COLORS["baseline"], alpha=0.85)
        bars_s = ax.bar(x + width / 2, sft_vals, width, label="SFT",
                        color=COLORS["sft"], alpha=0.85)
        for bar, v in list(zip(bars_b, base_vals)) + list(zip(bars_s, sft_vals)):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    else:
        bars = ax.bar(x, sft_vals, width * 1.5, label="SFT", color=COLORS["sft"], alpha=0.85)
        for bar, v in zip(bars, sft_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks)
    ax.set_ylabel("Score (%)")
    ax.set_title("SFT Model — All Benchmark Results")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out = output_dir / "sft_eval_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def print_summary(sft: dict, baseline: dict | None) -> None:
    print("\n" + "=" * 60)
    print(f"  {'Benchmark':<30} {'Baseline':>10} {'SFT':>10} {'Δ':>8}")
    print("  " + "-" * 56)
    metrics = [
        ("MMLU accuracy",         "mmlu_accuracy"),
        ("GSM8K accuracy",        "gsm8k_accuracy"),
        ("AlpacaEval win rate",   "ae_winrate"),
        ("AlpacaEval LC win rate","ae_lc_winrate"),
        ("SST % safe",            "sst_safe_rate"),
    ]
    for label, key in metrics:
        s = sft.get(key)
        b = baseline.get(key) if baseline else None
        if s is None and b is None:
            continue
        s_str = f"{s:.1%}" if s is not None else "—"
        b_str = f"{b:.1%}" if b is not None else "—"
        d_str = f"{(s-b):+.1%}" if (s is not None and b is not None) else "—"
        print(f"  {label:<30} {b_str:>10} {s_str:>10} {d_str:>8}")
    print("=" * 60 + "\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SFT evaluation results")
    parser.add_argument("--results-section4", type=Path, default=Path("results/section4"))
    parser.add_argument("--results-section2", type=Path, default=Path("results/section2"))
    parser.add_argument("--output", type=Path, default=None,
                        help="Output dir for section4 plots (defaults to --results-section4)")
    args = parser.parse_args()

    s4 = args.results_section4
    s2 = args.results_section2
    output_dir = args.output or s4
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load SFT metrics ─────────────────────────────────────────────────────
    sft: dict = {}
    mmlu_sft = load_mmlu_summary(s4, "sft")
    gsm_sft  = load_gsm8k_summary(s4, "sft")
    ae_sft   = load_alpaca_leaderboard(s4)
    sst_sft  = load_sst_annotated(s4, "sft")

    if mmlu_sft:
        sft["mmlu_accuracy"] = mmlu_sft["accuracy"]
    if gsm_sft:
        sft["gsm8k_accuracy"] = gsm_sft["accuracy"]
    if ae_sft:
        sft["ae_winrate"] = ae_sft["win_rate"]
        sft["ae_lc_winrate"] = ae_sft.get("length_controlled_winrate", 0) or 0
    if sst_sft:
        sft["sst_safe_rate"] = sst_safe_rate(sst_sft)

    if not sft:
        print("No section4 results found. Run part_6_4.sh first.")
        return

    # ── Load baseline metrics for comparison ─────────────────────────────────
    baseline: dict | None = {}
    mmlu_base = load_mmlu_summary(s2, "baseline")
    gsm_base  = load_gsm8k_summary(s2, "baseline")
    ae_base   = load_alpaca_leaderboard(s2)
    sst_base  = load_sst_annotated(s2, "baseline")

    if mmlu_base:
        baseline["mmlu_accuracy"] = mmlu_base["accuracy"]
    if gsm_base:
        baseline["gsm8k_accuracy"] = gsm_base["accuracy"]
    if ae_base:
        baseline["ae_winrate"] = ae_base["win_rate"]
        baseline["ae_lc_winrate"] = ae_base.get("length_controlled_winrate", 0) or 0
    if sst_base:
        baseline["sst_safe_rate"] = sst_safe_rate(sst_base)
    if not baseline:
        baseline = None

    # ── Print summary table ───────────────────────────────────────────────────
    print_summary(sft, baseline)

    # ── Section4-specific plots ───────────────────────────────────────────────
    mmlu_records = load_mmlu_records(s4, "sft")
    if mmlu_records:
        plot_mmlu_subject_accuracy(mmlu_records, output_dir)

    plot_summary_bar(sft, baseline, output_dir)

    # ── Comparison plots (baseline vs SFT) → results/section4/, not section2/
    # section2/ plots stay as baseline-only (referenced by the §2 README).
    # The side-by-side comparison belongs in section4/.
    print("Generating baseline vs SFT comparison plots -> results/section4/ ...")
    import subprocess, sys
    subprocess.run([
        sys.executable, "-m", "cs336_alignment.section2_zero_shot.plot_zero_shot_results",
        "--results-section2", str(s2),
        "--results-section4", str(s4),
        "--output", str(output_dir),
    ], check=True)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
