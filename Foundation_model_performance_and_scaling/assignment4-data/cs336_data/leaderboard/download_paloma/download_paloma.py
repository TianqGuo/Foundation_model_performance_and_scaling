"""
Download and tokenize the Paloma C4-100-domains validation split.

Reproduces the pre-tokenized file that lives at
/data/paloma/tokenized_paloma_c4_100_domains_validation.bin on the Together cluster.

Usage:
    python -m cs336_data.leaderboard.download_paloma.download_paloma \
        --output data/paloma/tokenized_paloma_c4_100_domains_validation.bin
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer, logging as hf_logging

hf_logging.set_verbosity_error()


def download_and_tokenize(output_path: Path) -> int:
    from datasets import load_dataset

    print("Downloading Paloma C4-100-domains validation split from HuggingFace …")
    ds = load_dataset("allenai/paloma", "c4_100_domains", split="val")
    print(f"  {len(ds):,} examples loaded.")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    eos = tokenizer.eos_token_id

    all_ids: list[int] = []
    for example in ds:
        text = example.get("text") or example.get("content") or ""
        if text.strip():
            all_ids.extend(tokenizer.encode(text))
            all_ids.append(eos)

    print(f"  Total tokens: {len(all_ids):,}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.array(all_ids, dtype=np.uint16).tofile(str(output_path))
    print(f"  Saved to {output_path}  ({output_path.stat().st_size / 1e6:.1f} MB)")
    return len(all_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and tokenize Paloma validation set")
    parser.add_argument(
        "--output",
        default="data/paloma/tokenized_paloma_c4_100_domains_validation.bin",
        help="Output path for tokenized validation bin",
    )
    args = parser.parse_args()
    download_and_tokenize(Path(args.output))


if __name__ == "__main__":
    main()
