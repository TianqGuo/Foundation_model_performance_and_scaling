"""CPU data/grader checks: python -m unittest discover -s tests -v.

Requires pyarrow, jinja2 and the Part 5 CPU math-grader dependencies; no models.
"""

import json
from pathlib import Path
import tempfile
import unittest

from jinja2 import Environment, StrictUndefined
import pyarrow.parquet as pq

from workloads.math_workload import (
    DATA_SOURCE, PART5_ROOT, RAW_CHAT_TEMPLATE, compute_score, fingerprint,
    load_reference, make_row, prepare_data, r1_sampling_params,
)


class MathWorkloadTests(unittest.TestCase):
    def test_raw_prompt_matches_reference_including_braces(self):
        template = (PART5_ROOT / "cs336_alignment/prompts/r1_zero.prompt").read_text()
        question = r"Find $\frac{1}{2}$ + {x}."
        row = make_row({"problem": question, "solution": "2"}, 7, "train", template)
        rendered = Environment(undefined=StrictUndefined).from_string(RAW_CHAT_TEMPLATE).render(
            messages=row["prompt"], add_generation_prompt=True)
        self.assertEqual(rendered, template.format(question=question))
        self.assertTrue(rendered.endswith("Assistant: <think>"))

    def test_ground_truth_precedence_and_answer_fallback(self):
        for example, expected in [
            ({"problem": "Q", "solution": r"Text \boxed{2}", "answer": "wrong"}, r"Text \boxed{2}"),
            ({"question": "Q", "answer": "Work #### 2"}, "2"),
            ({"problem": "Q", "ground_truth": 0}, "0"),
        ]:
            self.assertEqual(make_row(example, 0, "train", "{question}")["reward_model"]["ground_truth"], expected)

    def test_invalid_records_fail_instead_of_becoming_zero_reward(self):
        for example in [{"problem": "Q"}, {"solution": "2"}, {"problem": " ", "solution": "2"},
                        {"problem": "Q", "answer": "#### "}]:
            with self.assertRaises(ValueError):
                make_row(example, 0, "train", "{question}")

    def test_reward_matches_real_reference_for_format_and_math(self):
        grader = load_reference(str((PART5_ROOT / "cs336_alignment/drgrpo_grader.py").resolve()))
        for response, truth in [
            ("Reason </think> <answer>2</answer>", "2"),
            ("Reason </think> <answer>3</answer>", "2"),
            ("2", "2"),
            ("Reason </think><answer>2</answer>", "2"),
            (r"Reason </think> <answer>\boxed{\frac{1}{2}}</answer>", r"\frac{1}{2}"),
        ]:
            expected = grader.r1_zero_reward_fn(response, truth, fast=True)
            actual = compute_score(DATA_SOURCE, response, truth)
            self.assertEqual(actual["score"], expected["reward"])
            self.assertEqual(actual["acc"], expected["answer_reward"])
            self.assertEqual(actual["format_reward"], expected["format_reward"])
        self.assertEqual(compute_score(DATA_SOURCE, "Reason </think> <answer>2</answer>", "2")["score"], 1)

    def test_unknown_data_source_is_rejected(self):
        with self.assertRaises(ValueError):
            compute_score("unrelated", "2", "2")

    def test_sampling_preserves_stops_without_collapsing_training_groups(self):
        original = {"temperature": 1.0, "logprobs": False}
        params = r1_sampling_params(original, {"split": "train", "index": 7})
        self.assertNotIn("stop", original)
        self.assertNotIn("seed", params)
        self.assertEqual(params["min_tokens"], 4)
        self.assertTrue(params["include_stop_str_in_output"])
        self.assertEqual(params["stop"], ["</answer>"])
        evaluation = r1_sampling_params({"temperature": 0.0}, {"split": "validation", "index": 7})
        self.assertEqual(evaluation["seed"], 12352)
        self.assertEqual(evaluation["min_tokens"], 0)

    def test_parquet_selection_provenance_and_portable_reward(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            examples = [{"problem": f"Q{i}", "solution": str(i)} for i in range(12)]
            for name in ["train", "validation"]:
                (root / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in examples))
            manifest = prepare_data(root / "train.jsonl", root / "validation.jsonl", root / "out",
                                    max_train_examples=5, eval_count=4)
            train = pq.read_table(root / "out/train.parquet").to_pylist()
            val = pq.read_table(root / "out/validation.parquet").to_pylist()
            helper = load_reference(str((PART5_ROOT / "cs336_alignment/section7_grpo/run_utils.py").resolve()))
            expected = helper.evaluation_indices(12, 4, 12345)
            self.assertEqual(len(train), 5)
            self.assertEqual([x["extra_info"]["index"] for x in val], expected)
            self.assertEqual(len({x["extra_info"]["example_id"] for x in train + val}), 9)
            for name, info in manifest["outputs"].items():
                self.assertEqual(fingerprint(root / "out" / name)["sha256"], info["sha256"])
            self.assertEqual(manifest["inputs"]["grader"]["sha256"],
                             manifest["outputs"]["reference/drgrpo_grader.py"]["sha256"])
            self.assertEqual(compute_score(DATA_SOURCE, "x </think> <answer>2</answer>", "2",
                                          grader_path=root / "out/reference/drgrpo_grader.py")["score"], 1)
            with self.assertRaises(FileExistsError):
                prepare_data(root / "train.jsonl", root / "validation.jsonl", root / "out")

    def test_invalid_data_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "train.jsonl").write_text('{"problem":"Q"}\n')
            (root / "validation.jsonl").write_text('{"problem":"Q", "solution":"2"}\n')
            with self.assertRaises(ValueError):
                prepare_data(root / "train.jsonl", root / "validation.jsonl", root / "out")
            self.assertFalse((root / "out").exists())

    def test_empty_solution_exclusion_is_opt_in_and_keeps_source_indices(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = [{"problem": "Q", "solution": ""}, {"problem": "Q2", "solution": "2"}]
            (root / "train.jsonl").write_text("".join(json.dumps(x) + "\n" for x in records))
            (root / "validation.jsonl").write_text(json.dumps(records[1]) + "\n")
            with self.assertRaises(ValueError):
                prepare_data(root / "train.jsonl", root / "validation.jsonl", root / "strict")
            manifest = prepare_data(root / "train.jsonl", root / "validation.jsonl", root / "out",
                                    drop_empty_solutions=True)
            self.assertEqual(manifest["exclusions"], [{"split": "train", "index": 0, "reason": "empty_solution"}])
            self.assertEqual(pq.read_table(root / "out/train.parquet").to_pylist()[0]["extra_info"]["index"], 1)


if __name__ == "__main__":
    unittest.main()
