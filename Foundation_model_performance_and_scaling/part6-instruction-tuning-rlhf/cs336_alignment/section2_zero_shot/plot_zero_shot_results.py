"""Plot zero-shot and post-training evaluation results across all benchmarks.

Reads result files from results/section2/ (zero-shot baseline),
results/section4/ (SFT), and results/section5/ (DPO), and produces:

  baseline_accuracy_bar.png        — MMLU + GSM8K accuracy: baseline vs SFT vs DPO
  mmlu_subject_accuracy.png        — Per-subject MMLU accuracy (baseline, sorted)
  sst_safety_by_category.png       — % safe outputs by harm area: baseline vs SFT vs DPO
  alpaca_eval_winrate.png          — AlpacaEval winrate + LC winrate: baseline vs SFT vs DPO

Each plot is skipped gracefully if its required result files are not yet present.

Usage:
    # After zero-shot baseline runs:
    uv run python cs336_alignment/section2_zero_shot/plot_zero_shot_results.py

    # After SFT and DPO results are also available:
    uv run python cs336_alignment/section2_zero_shot/plot_zero_shot_results.py \\
        --results-section2 results/section2 \\
        --results-section4 results/section4 \\
        --results-section5 results/section5 \\
        --output results/section2/plots
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── colour palette (consistent across all plots) ─────────────────────────────
COLORS = {
    "baseline": "#4C72B0",
    "sft":      "#DD8452",
    "dpo":      "#55A868",
}
MODEL_LABELS = {
    "baseline": "Zero-shot baseline",
    "sft":      "SFT",
    "dpo":      "SFT + DPO",
}


# ── loaders ──────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_summary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_mmlu_records(results_dir: Path, suffix: str = "baseline") -> list[dict] | None:
    """Load per-example MMLU results. suffix is 'baseline', 'sft', or 'dpo'."""
    path = results_dir / f"eval_mmlu_{suffix}.jsonl"
    if not path.exists():
        return None
    return load_jsonl(path)


def load_mmlu_summary(results_dir: Path, suffix: str = "baseline") -> dict | None:
    path = results_dir / f"eval_mmlu_{suffix}.summary.json"
    if not path.exists():
        # Fall back to computing from JSONL
        records = load_mmlu_records(results_dir, suffix)
        if records is None:
            return None
        n_correct = sum(1 for r in records if r.get("correct", False))
        return {"accuracy": n_correct / len(records), "n_total": len(records)}
    return load_summary(path)


def load_gsm8k_summary(results_dir: Path, suffix: str = "baseline") -> dict | None:
    path = results_dir / f"eval_gsm8k_{suffix}.summary.json"
    if not path.exists():
        jsonl = results_dir / f"eval_gsm8k_{suffix}.jsonl"
        if not jsonl.exists():
            return None
        records = load_jsonl(jsonl)
        n_correct = sum(1 for r in records if r.get("correct", False))
        return {"accuracy": n_correct / len(records), "n_total": len(records)}
    return load_summary(path)


def load_sst_annotated(results_dir: Path, suffix: str = "baseline") -> list[dict] | None:
    path = results_dir / f"sst_{suffix}_annotated.jsonl"
    if not path.exists():
        return None
    return load_jsonl(path)


def load_alpaca_eval_results(results_dir: Path, suffix: str = "baseline") -> dict | None:
    """Load AlpacaEval annotation results.

    Tries (in order):
      1. Flat leaderboard.csv written by alpaca_eval (columns: win_rate, length_controlled_winrate)
      2. Flat alpaca_eval_<suffix>_summary.json written manually
      3. leaderboard.json inside a subdirectory named after the generator
    AlpacaEval stores win rates as percentages (0–100); we normalise to fractions (0–1).
    """
    import csv

    # 1. Flat leaderboard.csv (alpaca_eval default output location)
    csv_path = results_dir / "leaderboard.csv"
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "win_rate" in row:
                    return {
                        "win_rate": float(row["win_rate"]) / 100.0,
                        "length_controlled_winrate": (
                            float(row["length_controlled_winrate"]) / 100.0
                            if row.get("length_controlled_winrate")
                            else None
                        ),
                    }

    # 2. Manually written flat summary JSON
    path = results_dir / f"alpaca_eval_{suffix}_summary.json"
    if path.exists():
        data = load_summary(path)
        # Normalise to fraction if stored as percentage
        wr = data.get("win_rate", data.get("winrate", 0))
        lc = data.get("length_controlled_winrate", data.get("lc_winrate", None))
        if wr > 1:
            wr /= 100.0
        if lc is not None and lc > 1:
            lc /= 100.0
        return {"win_rate": wr, "length_controlled_winrate": lc}

    # 3. leaderboard.json in a generator-named subdirectory
    ann_files = list(results_dir.glob(f"*{suffix}*/leaderboard.json"))
    if ann_files:
        with open(ann_files[0]) as f:
            data = json.load(f)
        for v in data.values():
            if isinstance(v, dict) and "win_rate" in v:
                wr = v["win_rate"]
                lc = v.get("length_controlled_winrate", None)
                if wr > 1:
                    wr /= 100.0
                if lc is not None and lc > 1:
                    lc /= 100.0
                return {"win_rate": wr, "length_controlled_winrate": lc}

    return None


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_accuracy_bar(model_data: dict[str, dict], output_dir: Path) -> None:
    """Bar chart: MMLU and GSM8K accuracy for each model variant."""
    benchmarks = ["MMLU", "GSM8K"]
    models = [m for m in ["baseline", "sft", "dpo"] if m in model_data]
    if not models:
        print("No accuracy data available — skipping accuracy bar chart.")
        return

    x = np.arange(len(benchmarks))
    width = 0.25
    offsets = np.linspace(-(len(models) - 1) / 2, (len(models) - 1) / 2, len(models)) * width

    fig, ax = plt.subplots(figsize=(7, 4))
    for model, offset in zip(models, offsets):
        data = model_data[model]
        values = [
            data.get("mmlu_accuracy", 0) * 100,
            data.get("gsm8k_accuracy", 0) * 100,
        ]
        bars = ax.bar(x + offset, values, width, label=MODEL_LABELS[model],
                      color=COLORS[model], alpha=0.85)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Zero-Shot Benchmark Accuracy")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out = output_dir / "baseline_accuracy_bar.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_mmlu_subject_accuracy(records: list[dict], output_dir: Path,
                                suffix: str = "baseline") -> None:
    """Horizontal bar chart of per-subject MMLU accuracy, sorted ascending."""
    if not records:
        return

    # Aggregate per subject
    subject_stats: dict[str, list[bool]] = {}
    for r in records:
        subj = r.get("subject", "unknown")
        subject_stats.setdefault(subj, []).append(r.get("correct", False))

    subjects = sorted(subject_stats, key=lambda s: sum(subject_stats[s]) / len(subject_stats[s]))
    accuracies = [sum(subject_stats[s]) / len(subject_stats[s]) * 100 for s in subjects]

    # Truncate subject name for readability
    labels = [s.replace("_", " ").title() for s in subjects]

    fig_height = max(6, len(subjects) * 0.28)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    colors = [COLORS["sft"] if a >= 60 else COLORS["baseline"] for a in accuracies]
    ax.barh(labels, accuracies, color=colors, alpha=0.85)
    ax.axvline(x=sum(accuracies) / len(accuracies), color="black",
               linestyle="--", linewidth=1, label=f"Mean {sum(accuracies)/len(accuracies):.1f}%")
    ax.set_xlabel("Accuracy (%)")
    ax.set_title(f"MMLU Per-Subject Accuracy — {MODEL_LABELS.get(suffix, suffix)}")
    ax.set_xlim(0, 105)
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    out = output_dir / f"mmlu_subject_accuracy_{suffix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_sst_safety_by_category(model_records: dict[str, list[dict]],
                                 output_dir: Path) -> None:
    """Bar chart of % safe outputs per harm area, one group per model."""
    if not model_records:
        print("No SST annotated data available — skipping safety category chart.")
        return

    # Collect all harm areas
    all_areas: set[str] = set()
    for records in model_records.values():
        for r in records:
            area = r.get("harm_area", "Unknown")
            all_areas.add(area)
    areas = sorted(all_areas)

    models = [m for m in ["baseline", "sft", "dpo"] if m in model_records]
    x = np.arange(len(areas))
    width = 0.25
    offsets = np.linspace(-(len(models) - 1) / 2, (len(models) - 1) / 2, len(models)) * width

    fig, ax = plt.subplots(figsize=(11, 5))
    for model, offset in zip(models, offsets):
        records = model_records[model]
        safe_by_area: dict[str, list[bool]] = {a: [] for a in areas}
        for r in records:
            area = r.get("harm_area", "Unknown")
            # safety evaluator stores result in metrics.safe (float 1.0/0.0)
            is_safe = r.get("metrics", {}).get("safe", r.get("is_safe", r.get("safe", False)))
            if area in safe_by_area:
                safe_by_area[area].append(bool(is_safe))
        values = [
            (sum(safe_by_area[a]) / len(safe_by_area[a]) * 100) if safe_by_area[a] else 0
            for a in areas
        ]
        ax.bar(x + offset, values, width, label=MODEL_LABELS[model],
               color=COLORS[model], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([a.replace(", ", ",\n") for a in areas], fontsize=8)
    ax.set_ylabel("% Safe outputs")
    ax.set_title("SimpleSafetyTests — Safety Rate by Harm Area")
    ax.set_ylim(0, 110)
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out = output_dir / "sst_safety_by_category.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_alpaca_eval_winrate(model_winrates: dict[str, dict], output_dir: Path) -> None:
    """Bar chart: AlpacaEval winrate and length-controlled winrate per model."""
    if not model_winrates:
        print("No AlpacaEval winrate data available — skipping winrate chart.")
        return

    models = [m for m in ["baseline", "sft", "dpo"] if m in model_winrates]
    x = np.arange(len(models))
    width = 0.35

    winrates = [model_winrates[m].get("win_rate", 0) * 100 for m in models]
    lc_winrates = [model_winrates[m].get("length_controlled_winrate", 0) * 100
                   if model_winrates[m].get("length_controlled_winrate") is not None
                   else 0 for m in models]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - width / 2, winrates, width, label="Winrate",
                   color=[COLORS[m] for m in models], alpha=0.85)
    bars2 = ax.bar(x + width / 2, lc_winrates, width, label="LC Winrate",
                   color=[COLORS[m] for m in models], alpha=0.5, hatch="//")

    for bar, val in list(zip(bars1, winrates)) + list(zip(bars2, lc_winrates)):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax.set_ylabel("Winrate vs text-davinci-003 (%)")
    ax.set_title("AlpacaEval — Winrate (Llama 3.3 70B Instruct annotator)")
    ax.set_ylim(0, max(winrates + lc_winrates) * 1.2 + 5 if winrates else 50)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out = output_dir / "alpaca_eval_winrate.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def print_summary(model_data: dict, model_winrates: dict, model_sst: dict) -> None:
    print("\n" + "=" * 65)
    print(f"  {'Model':<20} {'MMLU':>8} {'GSM8K':>8} {'AE WR':>8} {'SST %safe':>10}")
    print("  " + "-" * 55)
    for model in ["baseline", "sft", "dpo"]:
        data = model_data.get(model, {})
        wr = model_winrates.get(model, {})
        sst_data = model_sst.get(model)
        mmlu = f"{data.get('mmlu_accuracy', 0):.1%}" if "mmlu_accuracy" in data else "—"
        gsm = f"{data.get('gsm8k_accuracy', 0):.1%}" if "gsm8k_accuracy" in data else "—"
        ae = f"{wr.get('win_rate', 0):.1%}" if wr else "—"
        sst_pct = "—"
        if sst_data:
            n_safe = sum(1 for r in sst_data if r.get("metrics", {}).get("safe", r.get("is_safe", r.get("safe", False))))
            sst_pct = f"{n_safe / len(sst_data):.1%}"
        print(f"  {MODEL_LABELS.get(model, model):<20} {mmlu:>8} {gsm:>8} {ae:>8} {sst_pct:>10}")
    print("=" * 65 + "\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot zero-shot and post-training benchmark results")
    parser.add_argument("--results-section2", type=Path, default=Path("results/section2"),
                        help="Directory with zero-shot baseline results")
    parser.add_argument("--results-section4", type=Path, default=Path("results/section4"),
                        help="Directory with SFT evaluation results")
    parser.add_argument("--results-section5", type=Path, default=Path("results/section5"),
                        help="Directory with DPO evaluation results")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory for plots (defaults to --results-section2)")
    args = parser.parse_args()

    output_dir = args.output or args.results_section2
    output_dir.mkdir(parents=True, exist_ok=True)

    # Map: model key → results directory
    dirs = {
        "baseline": args.results_section2,
        "sft":      args.results_section4,
        "dpo":      args.results_section5,
    }

    # ── Load accuracy summaries ───────────────────────────────────────────────
    model_data: dict[str, dict] = {}
    for model, d in dirs.items():
        suffix = "baseline" if model == "baseline" else model
        mmlu = load_mmlu_summary(d, suffix)
        gsm = load_gsm8k_summary(d, suffix)
        if mmlu or gsm:
            model_data[model] = {}
            if mmlu:
                model_data[model]["mmlu_accuracy"] = mmlu["accuracy"]
            if gsm:
                model_data[model]["gsm8k_accuracy"] = gsm["accuracy"]

    # ── Load per-example MMLU for subject breakdown ───────────────────────────
    mmlu_records_baseline = load_mmlu_records(args.results_section2, "baseline")

    # ── Load SST annotated data ───────────────────────────────────────────────
    model_sst: dict[str, list[dict]] = {}
    for model, d in dirs.items():
        suffix = "baseline" if model == "baseline" else model
        records = load_sst_annotated(d, suffix)
        if records:
            model_sst[model] = records

    # ── Load AlpacaEval winrates ──────────────────────────────────────────────
    model_winrates: dict[str, dict] = {}
    for model, d in dirs.items():
        suffix = "baseline" if model == "baseline" else model
        wr = load_alpaca_eval_results(d, suffix)
        if wr:
            model_winrates[model] = wr

    # ── Print summary table ───────────────────────────────────────────────────
    print_summary(model_data, model_winrates, model_sst)

    # ── Generate plots ────────────────────────────────────────────────────────
    plot_accuracy_bar(model_data, output_dir)

    if mmlu_records_baseline:
        plot_mmlu_subject_accuracy(mmlu_records_baseline, output_dir, suffix="baseline")

    plot_sst_safety_by_category(model_sst, output_dir)
    plot_alpaca_eval_winrate(model_winrates, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
