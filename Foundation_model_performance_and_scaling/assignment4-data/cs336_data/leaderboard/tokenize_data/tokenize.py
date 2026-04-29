"""
Part 4: Tokenize filtered data (Problem: tokenize_data).

Reads a filtered .txt file (one document per line), tokenizes each document
with the GPT-2 tokenizer, appends <|endoftext|> after each document, and
serializes the full token sequence as a np.uint16 array.

Usage:
    python -m cs336_data.leaderboard.tokenize_data.tokenize \\
        --input-dir  data/filtered \\
        --output     data/tokenized/train.bin

    # Tokenize a single file
    python -m cs336_data.leaderboard.tokenize_data.tokenize \\
        --input-file data/filtered/CC-MAIN-*.txt \\
        --output     data/tokenized/train.bin

Output:
    A single .bin file containing all token IDs as np.uint16.
    Compatible with the cs336-basics training script (np.fromfile dtype=np.uint16).
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, logging as hf_logging

# Documents longer than GPT-2's context window (1024) are expected and fine —
# the training script slices them into context windows. Suppress the noise.
hf_logging.set_verbosity_error()

_tokenizer: AutoTokenizer | None = None


def _init_worker() -> None:
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained("gpt2")


def _tokenize_doc(line: str) -> list[int]:
    return _tokenizer.encode(line) + [_tokenizer.eos_token_id]


def tokenize_files(
    input_files: list[Path],
    output_path: Path,
    workers: int | None = None,
    chunksize: int = 100,
) -> int:
    """
    Tokenize all documents from input_files and write to output_path.
    Returns total number of tokens.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all lines from all input files
    lines: list[str] = []
    for path in input_files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line.strip():
                    lines.append(line)

    print(f"Tokenizing {len(lines):,} documents from {len(input_files)} file(s) …")

    n_workers = workers or multiprocessing.cpu_count()
    all_ids: list[int] = []

    with multiprocessing.Pool(processes=n_workers, initializer=_init_worker) as pool:
        for token_ids in tqdm(
            pool.imap(_tokenize_doc, lines, chunksize=chunksize),
            total=len(lines),
            desc="Tokenizing",
        ):
            all_ids.extend(token_ids)

    print(f"Total tokens: {len(all_ids):,}")

    ids_array = np.array(all_ids, dtype=np.uint16)
    ids_array.tofile(str(output_path))
    print(f"Saved to {output_path}  ({output_path.stat().st_size / 1e6:.1f} MB)")

    return len(all_ids)


def main() -> None:
    try:
        default_workers = len(os.sched_getaffinity(0))
    except AttributeError:
        default_workers = multiprocessing.cpu_count()

    parser = argparse.ArgumentParser(description="Tokenize filtered data (Part 4)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-dir",  help="Directory of filtered .txt files (tokenizes all *.txt)")
    group.add_argument("--input-file", help="Single filtered .txt file")
    parser.add_argument("--output",   required=True, help="Output .bin file path")
    parser.add_argument("--workers",  type=int, default=default_workers)
    parser.add_argument("--chunksize", type=int, default=100)
    args = parser.parse_args()

    if args.input_dir:
        input_files = sorted(Path(args.input_dir).glob("*.txt"))
        if not input_files:
            raise SystemExit(f"No *.txt files found in {args.input_dir}")
    else:
        input_files = [Path(args.input_file)]

    total = tokenize_files(
        input_files=input_files,
        output_path=Path(args.output),
        workers=args.workers,
        chunksize=args.chunksize,
    )
    print(f"\nDataset size: {total:,} tokens  ({total / 1e9:.3f}B tokens)")


if __name__ == "__main__":
    main()
