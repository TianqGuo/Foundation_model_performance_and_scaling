from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

_ALPACA_TEMPLATE = (Path(__file__).parent.parent / "prompts" / "alpaca_sft.prompt").read_text().rstrip("\n")


class PackedSFTDataset(Dataset):
    """Packed sequence dataset for instruction fine-tuning.

    All formatted documents are concatenated (with EOS as delimiter) into one
    long token sequence, then split into non-overlapping chunks. Each chunk
    yields input_ids = chunk[:-1] and labels = chunk[1:], both of length seq_length.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: str | Path,
        seq_length: int,
        shuffle: bool,
    ):
        dataset_path = Path(dataset_path)
        opener = gzip.open if dataset_path.suffix == ".gz" else open
        examples = []
        with opener(dataset_path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))

        if shuffle:
            random.shuffle(examples)

        eos_id = tokenizer.eos_token_id
        all_ids: list[int] = []
        for ex in examples:
            text = _ALPACA_TEMPLATE.format(
                instruction=ex["prompt"],
                response=ex["response"],
            )
            ids = tokenizer.encode(text, add_special_tokens=True)
            all_ids.extend(ids)
            all_ids.append(eos_id)

        # Build (seq_length+1)-token windows; input_ids = window[:-1], labels = window[1:]
        tokens = torch.tensor(all_ids, dtype=torch.long)
        n = (len(tokens) - 1) // seq_length
        self._data = [
            {
                "input_ids": tokens[i * seq_length : i * seq_length + seq_length].clone(),
                "labels": tokens[i * seq_length + 1 : i * seq_length + seq_length + 1].clone(),
            }
            for i in range(n)
        ]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return self._data[i]


def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Return a DataLoader that constitutes one epoch over dataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)