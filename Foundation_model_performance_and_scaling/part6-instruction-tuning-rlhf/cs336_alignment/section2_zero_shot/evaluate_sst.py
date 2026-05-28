"""
Zero-shot SimpleSafetyTests prediction collection for Llama 3.1 8B.

Loads data/simple_safety_tests/simple_safety_tests.csv, generates model
outputs for each prompt, and saves predictions in the JSONL format required
by the safety evaluator.

Usage:
    uv run python cs336_alignment/section2_zero_shot/evaluate_sst.py \
        --model-path /data/a5-alignment/models/Llama-3.1-8B \
        --data-path data/simple_safety_tests/simple_safety_tests.csv \
        --output-path results/section2/sst_baseline.jsonl

Then evaluate with:
    uv run python scripts/evaluate_safety.py \\
        --input-path results/section2/sst_baseline.jsonl \\
        --model-name-or-path /data/a5-alignment/models/Llama-3.3-70B-Instruct \\
        --num-gpus 2 \\
        --output-path results/section2/sst_baseline_annotated.jsonl
"""

import argparse
import csv
import json
import time
from pathlib import Path

from vllm import LLM, SamplingParams


def _check_model_path(model_path: str) -> None:
    """Fail early with a helpful message if the model path does not exist."""
    p = Path(model_path)
    if p.exists():
        return
    hints: list[str] = []
    for base in ["/data/a5-alignment/models", "/data/models", "/workspace/models", str(p.parent)]:
        bp = Path(base)
        if bp.exists():
            children = sorted(x.name for x in bp.iterdir() if x.is_dir())[:12]
            hints.append(f"  {base}/: {', '.join(children) or '(empty)'}")
    hint_block = "\n".join(hints) if hints else "  (no candidate directories found)"
    raise FileNotFoundError(
        f"\nModel path not found: {model_path}\n"
        f"Pass the correct path with --model-path. Directories found nearby:\n"
        f"{hint_block}"
    )

SYSTEM_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "zero_shot_system_prompt.prompt"


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text()


def format_prompt(system_prompt: str, instruction: str) -> str:
    return system_prompt.format(instruction=instruction)


def load_sst(data_path: Path, max_examples: int | None = None) -> list[dict]:
    """Load SimpleSafetyTests CSV. Expected columns include 'prompts_final'."""
    examples = []
    with open(data_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            examples.append(dict(row))
            if max_examples is not None and len(examples) >= max_examples:
                break
    return examples


def main():
    parser = argparse.ArgumentParser(description="Zero-shot SimpleSafetyTests prediction collection")
    parser.add_argument(
        "--model-path",
        default="/data/a5-alignment/models/Llama-3.1-8B",
        help="Path to the model",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/simple_safety_tests/simple_safety_tests.csv"),
        help="Path to SimpleSafetyTests CSV",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results/section2/sst_baseline.jsonl"),
        help="Path to write predictions (JSONL for safety evaluator)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens to generate per example",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run on 10 examples only",
    )
    args = parser.parse_args()

    _check_model_path(args.model_path)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    system_prompt = load_system_prompt()
    print(f"Loading SimpleSafetyTests data from {args.data_path} ...")
    examples = load_sst(args.data_path, max_examples=10 if args.smoke_test else None)
    print(f"  Loaded {len(examples)} examples")

    # Build prompts — instruction is the raw SST prompt
    prompts = [format_prompt(system_prompt, ex["prompts_final"]) for ex in examples]

    # Run inference
    print(f"Loading model from {args.model_path} ...")
    llm = LLM(model=args.model_path, dtype="bfloat16", gpu_memory_utilization=0.85)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop=["# Query:"],
    )

    print("Running inference ...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    throughput = len(prompts) / elapsed
    print(f"  {len(prompts)} examples in {elapsed:.1f}s ({throughput:.2f} examples/s)")

    # Save in JSONL format required by the safety evaluator
    with open(args.output_path, "w") as f:
        for ex, output in zip(examples, outputs):
            record = {
                **ex,  # preserves id, harm_area, counter, category, prompts_final
                "output": output.outputs[0].text.strip(),
            }
            f.write(json.dumps(record) + "\n")
    print(f"\nPredictions saved to {args.output_path}")
    print(f"Throughput: {throughput:.2f} examples/s")
    print(f"\nTo evaluate safety, run:")
    print(f"  uv run python scripts/evaluate_safety.py \\")
    print(f"      --input-path {args.output_path} \\")
    print(f"      --model-name-or-path /data/a5-alignment/models/Llama-3.3-70B-Instruct \\")
    print(f"      --num-gpus 2 \\")
    print(f"      --output-path {args.output_path.with_suffix('.annotated.jsonl')}")


if __name__ == "__main__":
    main()
