"""
Explore Common Crawl WARC/WET files for Problem look_at_cc (Section 2.1).
Produces observations that help answer written questions (a)-(d).

USAGE:
    python explore_cc.py --warc <path/to/example.warc.gz> \
                         --wet  <path/to/example.warc.wet.gz> \
                         --output ../../results/filtering_cc/look_at_cc_observations.txt
"""
import argparse
import gzip
import io
from pathlib import Path

import tldextract
from fastwarc.warc import ArchiveIterator, WarcRecordType


# ── helpers ──────────────────────────────────────────────────────────────────

def _open(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return open(path, "rb")


def _get_records(path: Path, record_type, limit: int | None = None):
    with _open(path) as fh:
        n = 0
        for record in ArchiveIterator(fh, record_types=record_type):
            yield record
            n += 1
            if limit is not None and n >= limit:
                break


def _record_url(record) -> str:
    return record.headers.get("WARC-Target-URI", "<unknown>")


def _record_body_preview(record, max_bytes: int = 2000) -> str:
    raw = record.reader.read(max_bytes)
    try:
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return repr(raw[:200])


# ── part (a): first WARC response record ─────────────────────────────────────

def analyze_part_a(warc_path: Path) -> str:
    lines = ["=" * 70, "PART (a): First WARC response record", "=" * 70]
    for record in _get_records(warc_path, WarcRecordType.response, limit=1):
        url = _record_url(record)
        preview = _record_body_preview(record, max_bytes=3000)
        lines += [f"URL: {url}", "", "--- Raw content preview (first 3000 bytes) ---", preview]
    return "\n".join(lines)


# ── part (b): first WET record ───────────────────────────────────────────────

def analyze_part_b(wet_path: Path) -> str:
    lines = ["=" * 70, "PART (b): First WET extracted-text record", "=" * 70]
    # WET files use WarcRecordType.conversion for extracted text records
    for record in _get_records(wet_path, WarcRecordType.conversion, limit=1):
        url = _record_url(record)
        text = _record_body_preview(record, max_bytes=4000)
        lines += [f"URL: {url}", "", "--- Extracted text (first 4000 bytes) ---", text]
    return "\n".join(lines)


# ── part (d): 25-record WET survey ───────────────────────────────────────────

def _guess_language_hint(text: str) -> str:
    """Very rough heuristic language hint (no model needed for this helper)."""
    ascii_ratio = sum(1 for c in text[:500] if ord(c) < 128) / max(len(text[:500]), 1)
    if ascii_ratio > 0.92:
        return "likely English/Latin"
    elif ascii_ratio > 0.70:
        return "possibly mixed/European"
    else:
        return "possibly non-Latin script"


def analyze_part_d(wet_path: Path, n: int = 26) -> str:
    lines = ["=" * 70, f"PART (d): Survey of first {n} WET records", "=" * 70,
             f"{'#':<4} {'URL / Domain':<55} {'Lang hint':<25} {'Text preview'}",
             "-" * 120]

    high_quality_index = None
    for i, record in enumerate(_get_records(wet_path, WarcRecordType.conversion, limit=n), start=1):
        url = _record_url(record)
        text = _record_body_preview(record, max_bytes=600)
        extracted = tldextract.extract(url)
        domain = f"{extracted.domain}.{extracted.suffix}" if extracted.domain else url[:40]
        lang_hint = _guess_language_hint(text)
        preview = text[:80].replace("\n", " ").strip()

        # Rough quality signal: length, mostly words, not repetitive
        word_count = len(text.split())
        looks_quality = word_count > 100 and ascii_ratio_ok(text)
        marker = " ★" if looks_quality and high_quality_index is None else ""
        if looks_quality and high_quality_index is None:
            high_quality_index = i

        lines.append(f"{i:<4} {domain:<55} {lang_hint:<25} {preview[:60]}{marker}")

    lines += ["", f"First 'high-quality' candidate: record #{high_quality_index}" if high_quality_index else "No clear high-quality record found in first {n}."]
    return "\n".join(lines)


def ascii_ratio_ok(text: str) -> bool:
    sample = text[:1000]
    return sum(1 for c in sample if ord(c) < 128) / max(len(sample), 1) > 0.85


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Explore CC WARC/WET files (Section 2.1)")
    parser.add_argument("--warc", type=Path, required=True, help="Path to .warc.gz file")
    parser.add_argument("--wet", type=Path, required=True, help="Path to .warc.wet.gz file")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../../results/filtering_cc/look_at_cc_observations.txt"),
        help="Output text file for observations",
    )
    args = parser.parse_args()

    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    sections = []

    print("Analyzing part (a): first WARC record...")
    sections.append(analyze_part_a(args.warc))

    print("Analyzing part (b): first WET record...")
    sections.append(analyze_part_b(args.wet))

    print("Analyzing part (d): 25-record WET survey...")
    sections.append(analyze_part_d(args.wet, n=26))

    report = "\n\n".join(sections)
    output.write_text(report, encoding="utf-8")
    print(f"Observations saved to {output}")


if __name__ == "__main__":
    main()