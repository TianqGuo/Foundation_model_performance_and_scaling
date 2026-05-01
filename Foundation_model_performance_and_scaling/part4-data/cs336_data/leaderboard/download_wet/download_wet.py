"""
Download WET files from Common Crawl CC-MAIN-2025-18.

Fetches the WET paths manifest, selects N files, and downloads them
to output_dir using concurrent HTTP requests.

Usage:
    python -m cs336_data.leaderboard.download_wet.download_wet \
        --n 100 \
        --output-dir data/CC \
        --workers 8
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import io
import os
import urllib.request
from pathlib import Path

_CRAWL = "CC-MAIN-2025-18"
_MANIFEST_URL = f"https://data.commoncrawl.org/crawl-data/{_CRAWL}/wet.paths.gz"
_BASE_URL = "https://data.commoncrawl.org/"


def _fetch_paths(n: int) -> list[str]:
    """Download and parse the WET manifest, returning the first n paths."""
    print(f"Fetching manifest from {_MANIFEST_URL} …")
    with urllib.request.urlopen(_MANIFEST_URL) as resp:
        raw = resp.read()
    lines = gzip.decompress(raw).decode().splitlines()
    paths = [l.strip() for l in lines if l.strip()]
    print(f"Manifest has {len(paths):,} WET files. Selecting first {n}.")
    return paths[:n]


def _download_one(args: tuple[str, Path]) -> tuple[str, bool, str]:
    rel_path, out_dir = args
    filename = rel_path.split("/")[-1]
    dest = out_dir / filename
    if dest.exists():
        return filename, True, "already exists"
    url = _BASE_URL + rel_path
    tmp = dest.with_suffix(".tmp")
    try:
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
            while chunk := resp.read(1 << 20):  # 1 MB chunks
                f.write(chunk)
        tmp.rename(dest)
        size_mb = dest.stat().st_size / 1e6
        return filename, True, f"{size_mb:.0f} MB"
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        return filename, False, str(e)


def download_wet_files(n: int, output_dir: Path, workers: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _fetch_paths(n)
    args = [(p, output_dir) for p in paths]

    done = 0
    failed: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for filename, ok, msg in pool.map(_download_one, args):
            done += 1
            status = "OK" if ok else "FAIL"
            print(f"[{done:>4}/{n}] {status}  {filename}  ({msg})")
            if not ok:
                failed.append(filename)

    print(f"\nDownloaded {done - len(failed)}/{n} files to {output_dir}")
    if failed:
        print(f"Failed ({len(failed)}): {failed[:5]}{'…' if len(failed) > 5 else ''}")


def main() -> None:
    try:
        default_workers = min(8, len(os.sched_getaffinity(0)))
    except AttributeError:
        default_workers = 4

    parser = argparse.ArgumentParser(description="Download CC WET files")
    parser.add_argument("--n",          type=int,  default=100,
                        help="Number of WET files to download (default: 100)")
    parser.add_argument("--output-dir", default="data/CC",
                        help="Destination directory (default: data/CC)")
    parser.add_argument("--workers",    type=int, default=default_workers,
                        help=f"Parallel downloads (default: {default_workers})")
    args = parser.parse_args()

    download_wet_files(
        n=args.n,
        output_dir=Path(args.output_dir),
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
