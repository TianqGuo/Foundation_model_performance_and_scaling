from __future__ import annotations


def gopher_quality_filter(text: str) -> bool:
    """Return True if the document passes all Gopher heuristic quality rules."""
    words = text.split()
    n = len(words)

    # Rule 1: word count 50–100,000
    if n < 50 or n > 100_000:
        return False

    # Rule 2: mean word length 3–10 characters
    mean_len = sum(len(w) for w in words) / n
    if mean_len < 3 or mean_len > 10:
        return False

    # Rule 3: lines ending with ellipsis ≤ 30%
    lines = text.splitlines()
    if lines:
        ellipsis_count = sum(1 for line in lines if line.rstrip().endswith("..."))
        if ellipsis_count / len(lines) > 0.3:
            return False

    # Rule 4: ≥ 80% of words contain at least one alphabetic character
    alpha_words = sum(1 for w in words if any(c.isalpha() for c in w))
    if alpha_words / n < 0.8:
        return False

    return True