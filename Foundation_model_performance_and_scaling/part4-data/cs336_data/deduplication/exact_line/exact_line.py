"""
Exact line deduplication (Section 3.1).

Two-pass algorithm:
  1. Count line occurrences across all files using an MD5 hash as key.
  2. Rewrite each file keeping only lines whose hash count is exactly 1.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path


def _line_hash(line: str) -> bytes:
    return hashlib.md5(line.encode("utf-8")).digest()


def exact_line_deduplication(
    input_files: list[os.PathLike],
    output_directory: os.PathLike,
) -> None:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Pass 1: count occurrences of each line (keyed by hash)
    counts: dict[bytes, int] = {}
    for path in input_files:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                h = _line_hash(line)
                counts[h] = counts.get(h, 0) + 1

    # Pass 2: keep only lines that appear exactly once
    for path in input_files:
        path = Path(path)
        out_path = output_directory / path.name
        with open(path, encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
            for line in f_in:
                if counts[_line_hash(line)] == 1:
                    f_out.write(line)
