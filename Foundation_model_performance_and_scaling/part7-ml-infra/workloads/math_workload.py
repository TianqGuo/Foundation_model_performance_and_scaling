"""MATH data/prompt/reward adapters for verl; no model or trainer imports.

CPU preparation from the Part 7 root:
    python workloads/math_workload.py --output results/7a/prepared_math --drop-empty-solutions
Optional --train, --validation and --part5-root override discovered references.
Transfer the complete output directory to cloud: reference/ contains the exact
Part 5 grader, prompt and selection helper, with their attribution intact.
verl loads compute_score from this file; pass grader_path through reward_kwargs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PART5_ROOT = PROJECT_ROOT.parent / "part5-alignment"
DATA_SOURCE = "part7/math"
# The row already contains the fully rendered Part 5 prompt. No model-specific
# role markers, implicit system prompt, BOS/EOS or generation suffix are added.
RAW_CHAT_TEMPLATE = "{% for message in messages %}{{ message['content'] }}{% endfor %}"


def r1_sampling_params(sampling_params: dict, extra_info: dict, eval_seed=12345) -> dict:
    """Preserve the stop string and keep evaluation randomness separate.

    Training uses verl's engine RNG streams; never assign the same request seed
    to every response in a group, which would collapse its reward variance.
    """
    params = dict(sampling_params)
    validation = extra_info.get("split") == "validation"
    params.update(stop=["</answer>"], include_stop_str_in_output=True,
                  min_tokens=0 if validation else 4)
    if validation:
        params["seed"] = eval_seed + int(extra_info["index"])
    return params


def fingerprint(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@lru_cache(maxsize=8)
def load_reference(path: str) -> ModuleType:
    """Load a standalone reference without importing Part 5's training package.

    Compile directly to avoid writing bytecode into the preserved Part 5 tree.
    External CPU grading libraries are still imported normally; failures propagate.
    """
    source = Path(path).resolve()
    module = ModuleType("part7_reference_" + hashlib.sha256(str(source).encode()).hexdigest()[:16])
    module.__file__ = str(source)
    sys.modules[module.__name__] = module
    try:
        exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
    except Exception:
        sys.modules.pop(module.__name__, None)
        raise
    return module


def compute_score(data_source, solution_str, ground_truth, extra_info=None, *, grader_path=None, **kwargs):
    """verl's custom reward API: preserve the Part 5 binary, format-gated reward.

    grader_path should point to the prepared reference snapshot on cloud. The
    sibling Part 5 default is convenient for local CPU validation only.
    """
    if data_source != DATA_SOURCE:
        raise ValueError(f"Unexpected reward data source: {data_source!r}")
    path = Path(grader_path) if grader_path else PART5_ROOT / "cs336_alignment/drgrpo_grader.py"
    grader = load_reference(str(path.resolve()))
    result = grader.r1_zero_reward_fn(solution_str, ground_truth, fast=True)
    return {
        "score": float(result["reward"]),
        "acc": float(result["answer_reward"]),
        "format_reward": float(result["format_reward"]),
        "answer_reward": float(result["answer_reward"]),
    }


def read_examples(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{path}:{line_number}: blank record")
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        records.append(record)
    if not records:
        raise ValueError(f"Empty dataset: {path}")
    return records


def make_row(example: dict, index: int, split: str, template: str) -> dict:
    question = example.get("problem", example.get("question"))
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{split}[{index}]: nonempty problem/question required")
    # Match Part 5 train_grpo.get_ground_truth without importing its GPU stack.
    if "solution" in example:
        truth = example["solution"]
    else:
        truth = example.get("answer", example.get("ground_truth"))
    if truth is None or not str(truth).strip():
        raise ValueError(f"{split}[{index}]: nonempty ground truth required")
    truth = str(truth)
    if "solution" not in example and "####" in truth:
        truth = truth.split("####")[-1].strip()
    if not truth:
        raise ValueError(f"{split}[{index}]: empty extracted ground truth")
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": template.format(question=question)}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": truth},
        "extra_info": {"split": split, "index": index, "example_id": f"math/{split}/{index}"},
    }


def prepare_data(train: Path, validation: Path, output: Path, *, part5_root=PART5_ROOT,
                 max_train_examples=1024, eval_count=32, eval_seed=12345,
                 drop_empty_solutions=False) -> dict:
    """Write framework Parquet plus immutable reference snapshots and provenance.

    Does not tokenize, download datasets or load checkpoints. Refuses overwrite.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if max_train_examples <= 0 or eval_count <= 0:
        raise ValueError("Training and evaluation counts must be positive")
    part5_root = Path(part5_root).resolve()
    train, validation, output = (Path(p).resolve() for p in (train, validation, output))
    sources = {
        "train": train,
        "validation": validation,
        "grader": part5_root / "cs336_alignment/drgrpo_grader.py",
        "prompt": part5_root / "cs336_alignment/prompts/r1_zero.prompt",
        "selection_helper": part5_root / "cs336_alignment/section7_grpo/run_utils.py",
        "adapter": Path(__file__).resolve(),
        "agent": Path(__file__).with_name("r1_agent.py").resolve(),
    }
    inputs = {name: fingerprint(path) for name, path in sources.items()}
    template = sources["prompt"].read_text()
    train_records = read_examples(train)
    val_records = read_examples(validation)
    helper = load_reference(str(sources["selection_helper"]))
    indices = helper.evaluation_indices(len(val_records), eval_count, eval_seed)
    train_rows, exclusions = [], []
    for i, ex in enumerate(train_records[:max_train_examples]):
        if drop_empty_solutions and "solution" in ex and (
            ex["solution"] is None or not str(ex["solution"]).strip()
        ):
            exclusions.append({"split": "train", "index": i, "reason": "empty_solution"})
            continue
        train_rows.append(make_row(ex, i, "train", template))
    if not train_rows:
        raise ValueError("No usable training records in selected prefix")
    val_rows = [make_row(val_records[i], i, "validation", template) for i in indices]
    # Validate rows and create Arrow tables before claiming the output directory.
    train_table, val_table = (pa.Table.from_pylist(rows) for rows in (train_rows, val_rows))
    output.mkdir(parents=True, exist_ok=False)
    reference_dir = output / "reference"
    reference_dir.mkdir()
    for key in ("grader", "prompt", "selection_helper"):
        source = sources[key]
        snapshot = reference_dir / source.name
        snapshot.write_bytes(source.read_bytes())
        if fingerprint(snapshot)["sha256"] != inputs[key]["sha256"]:
            raise RuntimeError(f"Reference changed during preparation: {source}")
    pq.write_table(train_table, output / "train.parquet")
    pq.write_table(val_table, output / "validation.parquet")
    (output / "raw_prompt.jinja").write_text(RAW_CHAT_TEMPLATE)
    selection = {
        "protocol": "fixed_subset_v1", "seed": eval_seed, "indices": indices,
        "source_size": len(val_records), "source_sha256": inputs["validation"]["sha256"],
    }
    (output / "evaluation_subset.json").write_text(json.dumps(selection, indent=2) + "\n")
    outputs = {str(p.relative_to(output)): fingerprint(p)
               for p in sorted(output.rglob("*")) if p.is_file()}
    manifest = {
        "schema_version": 1, "data_source": DATA_SOURCE, "inputs": inputs,
        "train_source_size": len(train_records), "train_count": len(train_rows),
        "training_selection": "prefix", "validation_count": len(val_rows),
        "training_prefix_count": min(len(train_records), max_train_examples),
        "drop_empty_solutions": drop_empty_solutions, "exclusions": exclusions,
        "evaluation": selection, "chat_template": RAW_CHAT_TEMPLATE, "outputs": outputs,
        "reward_function": "r1_zero_reward_fn", "reward_fast": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def discover_data(explicit: Path | None, name: str, part5_root: Path) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
        return explicit.resolve()
    for directory in (part5_root / "data/math", Path("/data/a5-alignment/MATH")):
        candidate = directory / name
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not discover {name}; pass an explicit input path")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part5-root", type=Path, default=PART5_ROOT)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--validation", type=Path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/7a" / f"prepared_{stamp}")
    parser.add_argument("--max-train-examples", type=int, default=1024)
    parser.add_argument("--eval-count", type=int, default=32)
    parser.add_argument("--eval-seed", type=int, default=12345)
    parser.add_argument("--drop-empty-solutions", action="store_true",
                        help="Explicitly exclude empty training solutions and record their source indices")
    args = parser.parse_args()
    part5 = args.part5_root.resolve()
    manifest = prepare_data(
        discover_data(args.train, "train.jsonl", part5),
        discover_data(args.validation, "validation.jsonl", part5), args.output,
        part5_root=part5, max_train_examples=args.max_train_examples,
        eval_count=args.eval_count, eval_seed=args.eval_seed,
        drop_empty_solutions=args.drop_empty_solutions,
    )
    print(f"Prepared {manifest['train_count']} training prompts and {manifest['validation_count']} evaluation prompts")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
