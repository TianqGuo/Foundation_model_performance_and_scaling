"""
Response parsing utilities for zero-shot evaluation.

parse_mmlu_response  — extract predicted letter (A/B/C/D) from model output
parse_gsm8k_response — extract final numeric answer from model output
"""

import re
from typing import Any


def parse_mmlu_response(mmlu_example: dict[str, Any], model_output: str) -> str | None:
    """Parse a model output into the predicted MMLU answer letter.

    Looks for the pattern "The correct answer is X" where X is A, B, C, or D.
    Returns None if the output cannot be parsed into a valid letter.

    Args:
        mmlu_example: dict with keys "subject", "question", "options", "answer".
        model_output: raw string output from the model.

    Returns:
        One of "A", "B", "C", "D", or None.
    """
    match = re.search(r'[Tt]he correct answer is\s+([A-D])\b', model_output)
    if match:
        return match.group(1).upper()
    return None


def parse_gsm8k_response(model_output: str) -> str | None:
    """Parse a model output into the final numeric answer.

    Takes the last number that appears in the output (stripping commas).
    Returns None if no numeric digits are found.

    Args:
        model_output: raw string output from the model.

    Returns:
        String representation of the numeric answer (e.g. "72"), or None.
    """
    numbers = re.findall(r'\b\d[\d,]*(?:\.\d+)?\b', model_output)
    if numbers:
        return numbers[-1].replace(',', '')
    return None
