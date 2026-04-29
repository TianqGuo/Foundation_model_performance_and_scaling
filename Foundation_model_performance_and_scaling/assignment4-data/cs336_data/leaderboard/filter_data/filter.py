"""
Part 4: Filter CC WET files for language modeling (Problem: filter_data).

Pipeline (applied in order, first rejection wins):
  1. Minimum length     — skip records with < 100 chars
  2. Language ID        — keep English (score >= lang_threshold, default 0.65)
  3. Gopher rules       — word count, mean word length, ellipsis ratio, alpha ratio
  4. Quality classifier — keep pages with wiki-probability >= quality_threshold (default 0.3)
  5. NSFW filter        — discard NSFW pages (confidence >= nsfw_threshold, default 0.8)
  6. PII masking        — mask emails, phones, IPs (always applied, not a filter)

Usage:
    python -m cs336_data.leaderboard.filter_data.filter \\
        --input-dir /data/CC \\
        --output-dir /data/filtered \\
        --workers 8

    # Dry-run on 5 files to check pipeline
    python -m cs336_data.leaderboard.filter_data.filter \\
        --input-dir /data/CC \\
        --output-dir /tmp/filtered_test \\
        --limit 5 --workers 4

Output:
    One .txt per WET file in --output-dir (one document per line, newlines collapsed).
    filter_stats.json with per-stage rejection counts, timing, and config.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import os
import time
from pathlib import Path

from fastwarc.warc import ArchiveIterator, WarcRecordType

# Ordered rejection stages tracked in stats
_STAGES = ["total", "empty", "non_english", "gopher_fail", "low_quality", "nsfw", "kept"]


def _zero() -> dict[str, int]:
    return {k: 0 for k in _STAGES}


def process_wet_file(
    input_path: str,
    output_path: str,
    lang_threshold: float,
    quality_threshold: float,
    nsfw_threshold: float,
) -> dict[str, int]:
    """Filter one WET file; write kept documents (one per line) to output_path."""
    # Imports are cached in sys.modules after first call per worker process
    from cs336_data.filtering_cc.language_id.identify import identify_language
    from cs336_data.filtering_cc.quality_rules.gopher import gopher_quality_filter
    from cs336_data.filtering_cc.quality_classifier.classify import classify_quality
    from cs336_data.filtering_cc.pii.mask import mask_emails, mask_phone_numbers, mask_ips
    from cs336_data.filtering_cc.harmful_content.classify import classify_nsfw

    counts = _zero()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(input_path, "rb") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for record in ArchiveIterator(f_in, record_types=WarcRecordType.conversion):
            counts["total"] += 1
            text = record.reader.read().decode("utf-8", errors="replace").strip()

            # 1. Minimum length
            if len(text) < 100:
                counts["empty"] += 1
                continue

            # 2. Language ID — keep English only
            lang, lang_score = identify_language(text)
            if lang != "en" or lang_score < lang_threshold:
                counts["non_english"] += 1
                continue

            # 3. Gopher quality rules
            if not gopher_quality_filter(text):
                counts["gopher_fail"] += 1
                continue

            # 4. Quality classifier — keep pages with sufficient wiki-probability
            label, score = classify_quality(text)
            wiki_prob = score if label == "wiki" else (1.0 - score)
            if wiki_prob < quality_threshold:
                counts["low_quality"] += 1
                continue

            # 5. NSFW filter (skip if nsfw_threshold == 1.0 to disable)
            if nsfw_threshold < 1.0:
                nsfw_label, nsfw_score = classify_nsfw(text)
                if nsfw_label == "nsfw" and nsfw_score >= nsfw_threshold:
                    counts["nsfw"] += 1
                    continue

            # 6. PII masking — always applied, not a rejection filter
            text, _ = mask_emails(text)
            text, _ = mask_phone_numbers(text)
            text, _ = mask_ips(text)

            # Write as a single line (internal newlines collapsed to space)
            f_out.write(text.replace("\n", " ").strip() + "\n")
            counts["kept"] += 1

    return counts


def _print_stats(totals: dict[str, int], elapsed: float, num_files: int) -> None:
    total = totals["total"]
    print(f"\n{'Stage':<25} {'Removed':>10}   {'% of total':>10}")
    print("-" * 52)
    for stage in _STAGES[1:-1]:
        n = totals[stage]
        pct = 100.0 * n / total if total else 0.0
        print(f"  {stage:<23} {n:>10,}   {pct:>9.1f}%")
    print("-" * 52)
    kept = totals["kept"]
    print(f"  {'kept':<23} {kept:>10,}   {100.0*kept/total if total else 0:>9.1f}%")
    print(f"  {'total':<23} {total:>10,}")

    per_file = elapsed / num_files if num_files else 0
    full_cc_h = per_file * 100_000 / 3600
    print(f"\nElapsed: {elapsed:.1f}s  ({per_file:.1f}s/file)")
    print(f"Estimated time for 100,000 WETs at this rate: {full_cc_h:.1f}h")


def main() -> None:
    # Cap at 16 workers: each worker imports numpy/fasttext which triggers OpenBLAS
    # thread creation. Too many workers × OpenBLAS threads exhausts RLIMIT_NPROC.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    try:
        default_workers = min(16, len(os.sched_getaffinity(0)))
    except AttributeError:
        default_workers = min(16, os.cpu_count() or 4)

    parser = argparse.ArgumentParser(description="Filter CC WET files for LM training (Part 4)")
    parser.add_argument("--input-dir",  required=True,  help="Directory with CC*.warc.wet.gz files")
    parser.add_argument("--output-dir", required=True,  help="Output directory for filtered .txt files")
    parser.add_argument("--workers",    type=int, default=default_workers)
    parser.add_argument("--lang-threshold",    type=float, default=0.65,
                        help="Min language-ID confidence to keep (default: 0.65)")
    parser.add_argument("--quality-threshold", type=float, default=0.3,
                        help="Min wiki-probability to keep (0=keep all, 1=only confident wiki, default: 0.3)")
    parser.add_argument("--nsfw-threshold",    type=float, default=0.8,
                        help="NSFW confidence threshold; set to 1.0 to disable (default: 0.8)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N files (for testing)")
    args = parser.parse_args()

    wet_files = sorted(Path(args.input_dir).glob("*.warc.wet.gz"))
    if args.limit:
        wet_files = wet_files[: args.limit]
    if not wet_files:
        raise SystemExit(f"No *.warc.wet.gz files found in {args.input_dir}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Filtering {len(wet_files)} WET file(s) with {args.workers} worker(s) …")
    print(f"  lang>={args.lang_threshold}  quality>={args.quality_threshold}  "
          f"nsfw<={args.nsfw_threshold}")

    totals = _zero()
    t0 = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_wet_file,
                str(p),
                str(Path(args.output_dir) / p.name.replace(".warc.wet.gz", ".txt")),
                args.lang_threshold,
                args.quality_threshold,
                args.nsfw_threshold,
            ): p
            for p in wet_files
        }
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            counts = fut.result()
            for k in _STAGES:
                totals[k] += counts[k]
            done += 1
            if done % max(1, len(wet_files) // 20) == 0 or done == len(wet_files):
                print(f"  [{done}/{len(wet_files)}]  kept so far: {totals['kept']:,}")

    elapsed = time.time() - t0
    _print_stats(totals, elapsed, len(wet_files))

    stats_path = Path(args.output_dir) / "filter_stats.json"
    with open(stats_path, "w") as f:
        json.dump(
            {
                "counts": totals,
                "elapsed_s": round(elapsed, 2),
                "num_files": len(wet_files),
                "per_file_s": round(elapsed / len(wet_files), 2) if wet_files else 0,
                "estimated_100k_files_hours": round(elapsed / len(wet_files) * 100_000 / 3600, 1)
                if wet_files else 0,
                "config": vars(args),
            },
            f,
            indent=2,
        )
    print(f"\nStats saved to {stats_path}")


if __name__ == "__main__":
    main()