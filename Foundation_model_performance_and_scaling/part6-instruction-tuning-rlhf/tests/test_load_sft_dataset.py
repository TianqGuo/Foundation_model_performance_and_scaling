"""Tests for load_sft_dataset filtering logic in train_sft.py.

Run: uv run pytest tests/test_load_sft_dataset.py -v
"""

import json
import re
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers that mirror the implementation (for unit-level testing without import side effects)
# ---------------------------------------------------------------------------

R1_ZERO_PROMPT = (
    "A conversation between User and Assistant. The User asks a question, and the Assistant "
    "solves it. The Assistant first thinks about the reasoning process in the mind and then "
    "provides the User with the answer. The reasoning process is enclosed within <think> </think> "
    "and answer is enclosed within <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think> <answer> answer here </answer>.\n"
    "User: {question}\n"
    "Assistant: <think>"
)


def make_prompt(question: str) -> str:
    return R1_ZERO_PROMPT.format(question=question)


def make_response(reasoning: str, answer: str) -> str:
    return f"{reasoning}</think> <answer>{answer}</answer>"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PROBLEMS = [
    ("What is 2 + 2?",    "4"),
    ("What is 3 * 5?",    "15"),
    ("What is 10 - 7?",   "3"),
    ("What is 100 / 4?",  "25"),
    ("What is 2 ^ 3?",    "8"),
]


@pytest.fixture
def tmp_data(tmp_path):
    """Write a minimal sft.jsonl and train.jsonl with 5 examples.

    Examples 0, 1, 2 have correct answers; 3 has a wrong answer; 4 has no
    matching ground-truth (simulates a question absent from train.jsonl) but
    has a valid <answer> tag.
    """
    sft_path = tmp_path / "sft.jsonl"
    train_path = tmp_path / "train.jsonl"

    sft_rows = [
        {"prompt": make_prompt(PROBLEMS[0][0]), "response": make_response("step", PROBLEMS[0][1])},   # correct
        {"prompt": make_prompt(PROBLEMS[1][0]), "response": make_response("step", PROBLEMS[1][1])},   # correct
        {"prompt": make_prompt(PROBLEMS[2][0]), "response": make_response("step", PROBLEMS[2][1])},   # correct
        {"prompt": make_prompt(PROBLEMS[3][0]), "response": make_response("step", "99")},              # wrong answer
        {"prompt": make_prompt("What is the meaning of life?"), "response": make_response("hmm", "42")},  # not in train.jsonl
    ]
    sft_path.write_text("\n".join(json.dumps(r) for r in sft_rows))

    train_rows = [
        {"problem": PROBLEMS[i][0], "solution": PROBLEMS[i][1]}
        for i in range(4)  # intentionally exclude the 5th question
    ]
    train_path.write_text("\n".join(json.dumps(r) for r in train_rows))

    return sft_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_no_filter_returns_all(tmp_data):
    from cs336_alignment.section4_sft.train_sft import load_sft_dataset
    result = load_sft_dataset(tmp_data, filter_correct=False)
    assert len(result) == 5


def test_filter_correct_removes_wrong_answer(tmp_data):
    """filter_correct=True should keep examples 0,1,2 (correct) and 4 (no GT → format fallback)."""
    from cs336_alignment.section4_sft.train_sft import load_sft_dataset
    result = load_sft_dataset(tmp_data, filter_correct=True)
    # 3 correct-answer examples + 1 fallback (no GT, has <answer> tag) = 4
    assert len(result) == 4
    # Confirm the wrong-answer example (PROBLEMS[3]) is gone
    questions_kept = {
        re.search(r"User: (.+?)\nAssistant:", ex["prompt"], re.DOTALL).group(1).strip()
        for ex in result
    }
    assert PROBLEMS[3][0] not in questions_kept


def test_filter_correct_keeps_correct_answers(tmp_data):
    """All three examples with correct verified answers should be retained."""
    from cs336_alignment.section4_sft.train_sft import load_sft_dataset
    result = load_sft_dataset(tmp_data, filter_correct=True)
    questions_kept = {
        re.search(r"User: (.+?)\nAssistant:", ex["prompt"], re.DOTALL).group(1).strip()
        for ex in result
    }
    for i in range(3):
        assert PROBLEMS[i][0] in questions_kept


def test_filter_fallback_without_train_jsonl(tmp_path):
    """When train.jsonl is absent, filter falls back to format check (<answer> tag)."""
    from cs336_alignment.section4_sft.train_sft import load_sft_dataset
    sft_path = tmp_path / "sft.jsonl"
    rows = [
        {"prompt": make_prompt("Q1"), "response": make_response("r", "42")},   # has <answer>
        {"prompt": make_prompt("Q2"), "response": "no tags here"},              # no <answer>
    ]
    sft_path.write_text("\n".join(json.dumps(r) for r in rows))
    result = load_sft_dataset(sft_path, filter_correct=True)
    assert len(result) == 1
    assert "<answer>" in result[0]["response"]


def test_filter_with_explicit_solution_field(tmp_path):
    """Examples that already have a 'solution' field should use it directly."""
    from cs336_alignment.section4_sft.train_sft import load_sft_dataset
    sft_path = tmp_path / "sft.jsonl"
    rows = [
        {"prompt": "Q", "response": make_response("r", "7"), "solution": "7"},   # correct
        {"prompt": "Q", "response": make_response("r", "99"), "solution": "7"},  # wrong
    ]
    sft_path.write_text("\n".join(json.dumps(r) for r in rows))
    result = load_sft_dataset(sft_path, filter_correct=True)
    assert len(result) == 1
    assert "7" in result[0]["response"]


def test_max_examples_applied_after_filter(tmp_data):
    """max_examples should limit the result AFTER filtering."""
    from cs336_alignment.section4_sft.train_sft import load_sft_dataset
    result = load_sft_dataset(tmp_data, filter_correct=True, max_examples=2)
    assert len(result) == 2


def test_max_examples_no_filter(tmp_data):
    from cs336_alignment.section4_sft.train_sft import load_sft_dataset
    result = load_sft_dataset(tmp_data, filter_correct=False, max_examples=3)
    assert len(result) == 3


def test_extract_question_from_prompt():
    """_extract_question_from_prompt should handle single- and multi-line questions."""
    from cs336_alignment.section4_sft.train_sft import _extract_question_from_prompt
    q1 = "What is 2 + 2?"
    assert _extract_question_from_prompt(make_prompt(q1)) == q1

    q2 = "Let f(x) = x^2.\nCompute f(3)."
    assert _extract_question_from_prompt(make_prompt(q2)) == q2

    assert _extract_question_from_prompt("no match here") == ""


# ---------------------------------------------------------------------------
# Optional: verify against the real MATH dataset (skipped if data absent)
# ---------------------------------------------------------------------------

REAL_SFT = Path(__file__).parent.parent / "data" / "math" / "sft.jsonl"

@pytest.mark.skipif(not REAL_SFT.exists(), reason="MATH sft.jsonl not available")
def test_real_data_filter_count():
    """Filtering real MATH sft.jsonl should yield 4542 correct examples (pre-computed)."""
    from cs336_alignment.section4_sft.train_sft import load_sft_dataset
    result = load_sft_dataset(REAL_SFT, filter_correct=True)
    # Allow ±5 tolerance for reward-function edge cases (sympy non-determinism)
    assert abs(len(result) - 4542) <= 5, f"Expected ~4542 correct examples, got {len(result)}"