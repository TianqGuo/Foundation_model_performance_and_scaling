"""
Train a fastText quality classifier (Section 2.7).

Positives: pages linked from English Wikipedia (high quality)
Negatives: random CC pages

USAGE:
    python -m cs336_data.filtering_cc.quality_classifier.train \
        --wiki-urls  data/wiki/enwiki-20240420-extracted_urls.txt.gz \
        --cc-warc    data/CC/CC-MAIN-20250417135010-20250417165010-00065.warc.gz \
        --output     cs336_data/assets/quality_classifier.bin \
        --n-wiki 1000 --n-cc 1000 --n-try-wiki 8000 --workers 30
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import random
import tempfile
from pathlib import Path

import fasttext
import requests
from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.filtering_cc.html_extraction.extract import extract_text_from_html_bytes
from cs336_data.filtering_cc.language_id.identify import identify_language
from cs336_data.filtering_cc.quality_rules.gopher import gopher_quality_filter

WIKI_DOWNLOAD_URL = (
    "https://nlp.stanford.edu/data/nfliu/cs336-spring-2024/assignment4/"
    "enwiki-20240420-extracted_urls.txt.gz"
)
CLUSTER_WIKI = Path("/data/wiki/enwiki-20240420-extracted_urls.txt.gz")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_text(url: str) -> str | None:
    try:
        resp = requests.get(
            url, timeout=5,
            headers={"User-Agent": "Mozilla/5.0 (compatible; cs336-data/1.0)"},
        )
        if resp.status_code != 200:
            return None
        return extract_text_from_html_bytes(resp.content)
    except Exception:
        return None


def _is_good_english(text: str) -> bool:
    if not text or len(text.strip()) < 200:
        return False
    lang, score = identify_language(text)
    return lang == "en" and score >= 0.7


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_wiki_positives(
    wiki_gz: Path, n_target: int, n_try: int, workers: int
) -> list[str]:
    print(f"Reading wiki URLs from {wiki_gz} …")
    with gzip.open(wiki_gz, "rt", encoding="utf-8", errors="replace") as fh:
        urls = [line.strip() for line in fh if line.strip().startswith("http")]
    random.shuffle(urls)
    urls = urls[:n_try]
    print(f"  Trying {len(urls)} URLs to reach {n_target} positives …")

    positives: list[str] = []
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    try:
        fut_to_url = {pool.submit(_fetch_text, u): u for u in urls}
        done = 0
        for fut in concurrent.futures.as_completed(fut_to_url):
            done += 1
            text = fut.result()
            if text and _is_good_english(text) and gopher_quality_filter(text):
                positives.append(text)
            if done % 200 == 0:
                print(f"  Tried {done}/{len(urls)}, kept {len(positives)} …")
            if len(positives) >= n_target:
                break
    finally:
        # cancel_futures=True drops queued (not yet running) tasks immediately
        pool.shutdown(wait=False, cancel_futures=True)

    print(f"  Collected {len(positives)} wiki positives.")
    return positives


def collect_cc_negatives(warc_path: Path, n_target: int) -> list[str]:
    """Collect raw CC pages as negatives — no language/quality filtering.

    Supports both WARC files (WarcRecordType.response, extracts HTML) and
    WET files (WarcRecordType.conversion, pre-extracted text).
    """
    is_wet = "warc.wet" in warc_path.name or warc_path.name.endswith(".wet.gz")
    record_type = WarcRecordType.conversion if is_wet else WarcRecordType.response
    print(f"Extracting CC negatives from {warc_path} ({'WET' if is_wet else 'WARC'}) …")
    negatives: list[str] = []
    with gzip.open(warc_path, "rb") as fh:
        for record in ArchiveIterator(fh, record_types=record_type):
            if len(negatives) >= n_target:
                break
            raw = record.reader.read()
            if is_wet:
                text = raw.decode("utf-8", errors="replace").strip()
            else:
                if b"<html" not in raw[:500].lower() and b"<!doctype" not in raw[:500].lower():
                    continue
                text = extract_text_from_html_bytes(raw)
            if text and len(text.strip()) > 100:
                negatives.append(text)
    print(f"  Collected {len(negatives)} CC negatives.")
    return negatives


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    return text.replace("\n", " ").replace("\r", " ").strip()


def train(
    wiki_gz: Path,
    cc_warc: Path,
    output: Path,
    n_wiki: int = 1000,
    n_cc: int = 1000,
    n_try_wiki: int = 8000,
    workers: int = 30,
    seed: int = 42,
) -> None:
    random.seed(seed)

    positives = collect_wiki_positives(wiki_gz, n_wiki, n_try_wiki, workers)
    negatives = collect_cc_negatives(cc_warc, n_cc)

    if not positives:
        raise RuntimeError("No wiki positives collected — check network or wiki URL file.")
    if not negatives:
        raise RuntimeError("No CC negatives collected — check WARC file path.")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
        train_path = tmp.name
        for text in positives:
            tmp.write(f"__label__wiki {_clean(text)}\n")
        for text in negatives:
            tmp.write(f"__label__cc {_clean(text)}\n")

    print(f"Training fastText on {len(positives)} wiki + {len(negatives)} CC examples …")
    model = fasttext.train_supervised(
        train_path,
        wordNgrams=2,
        epoch=10,
        lr=0.5,
        dim=100,
        loss="softmax",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output))
    print(f"Model saved to {output}")

    Path(train_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train quality classifier (Section 2.7)")
    parser.add_argument("--wiki-urls", type=Path, help="Path to enwiki extracted URLs .gz")
    parser.add_argument("--cc-warc", type=Path, required=True, help="Path to CC WARC .gz")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).parents[2] / "assets" / "quality_classifier.bin",
    )
    parser.add_argument("--n-wiki", type=int, default=1000)
    parser.add_argument("--n-cc", type=int, default=1000)
    parser.add_argument("--n-try-wiki", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Resolve wiki URL file
    wiki_gz = args.wiki_urls
    if wiki_gz is None:
        if CLUSTER_WIKI.exists():
            wiki_gz = CLUSTER_WIKI
            print(f"Using cluster wiki file: {wiki_gz}")
        else:
            raise SystemExit(
                "ERROR: --wiki-urls not specified and cluster file not found.\n"
                f"Download from: {WIKI_DOWNLOAD_URL}\n"
                "  wget -P data/wiki/ " + WIKI_DOWNLOAD_URL
            )

    train(
        wiki_gz=wiki_gz,
        cc_warc=args.cc_warc,
        output=args.output,
        n_wiki=args.n_wiki,
        n_cc=args.n_cc,
        n_try_wiki=args.n_try_wiki,
        workers=args.workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()