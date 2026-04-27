"""
Compare our HTML extraction output against CC's pre-built WET file (Section 2.2b).

For each WARC response record, extracts text using our function and finds the
corresponding WET conversion record by URL, then compares side-by-side.

USAGE:
    cd cs336_data/filtering_cc/html_extraction
    python compare_wet.py --warc ../../../data/CC/<file>.warc.gz \
                          --wet  ../../../data/CC/<file>.warc.wet.gz \
                          --output ../../../results/filtering_cc/wet_comparison.txt \
                          --n 20
"""
import argparse
import gzip
from pathlib import Path

from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.filtering_cc.html_extraction.extract import extract_text_from_html_bytes


def _open(path: Path):
    return gzip.open(path, "rb") if path.suffix == ".gz" else open(path, "rb")


def _load_wet_index(wet_path: Path) -> dict[str, str]:
    """Build URL → extracted_text map from WET file."""
    index = {}
    with _open(wet_path) as fh:
        for record in ArchiveIterator(fh, record_types=WarcRecordType.conversion):
            url = record.headers.get("WARC-Target-URI", "")
            text = record.reader.read().decode("utf-8", errors="replace")
            index[url] = text
    return index


def _preview(text: str, chars: int = 600) -> str:
    snippet = text[:chars].replace("\n", "↵ ")
    return snippet + ("…" if len(text) > chars else "")


def compare(warc_path: Path, wet_path: Path, n: int, output: Path) -> None:
    print("Loading WET index…")
    wet_index = _load_wet_index(wet_path)
    print(f"  {len(wet_index)} WET records indexed.")

    lines = [
        "=" * 80,
        "Section 2.2b: Our extraction vs. CC WET file",
        "=" * 80,
    ]

    compared = 0
    with _open(warc_path) as fh:
        for record in ArchiveIterator(fh, record_types=WarcRecordType.response):
            if compared >= n:
                break

            url = record.headers.get("WARC-Target-URI", "")
            content_type = record.headers.get("WARC-Identified-Payload-Type", "")
            if "html" not in content_type.lower() and "html" not in url.lower():
                # Skip non-HTML records
                raw_peek = record.reader.read(200)
                if b"<html" not in raw_peek.lower() and b"<!doctype" not in raw_peek.lower():
                    continue
                raw_bytes = raw_peek + record.reader.read()
            else:
                raw_bytes = record.reader.read()

            our_text = extract_text_from_html_bytes(raw_bytes) or ""
            wet_text = wet_index.get(url, "<not found in WET>")

            lines += [
                "",
                f"Record {compared + 1}: {url}",
                "-" * 80,
                f"[OURS]  ({len(our_text)} chars)",
                _preview(our_text),
                "",
                f"[WET]   ({len(wet_text)} chars)",
                _preview(wet_text),
            ]
            compared += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Comparison saved to {output}")


def main():
    parser = argparse.ArgumentParser(description="Compare our extraction vs WET (Section 2.2b)")
    parser.add_argument("--warc", type=Path, required=True)
    parser.add_argument("--wet", type=Path, required=True)
    parser.add_argument("--n", type=int, default=20, help="Number of records to compare")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../../results/filtering_cc/wet_comparison.txt"),
    )
    args = parser.parse_args()
    compare(args.warc, args.wet, args.n, args.output)


if __name__ == "__main__":
    main()