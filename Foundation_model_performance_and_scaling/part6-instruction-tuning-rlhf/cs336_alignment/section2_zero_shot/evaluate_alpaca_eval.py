"""
Zero-shot AlpacaEval prediction collection for Llama 3.1 8B.

Loads data/alpaca_eval/alpaca_eval.jsonl, generates model outputs for each
instruction using vLLM (greedy decoding), and saves predictions in the JSON
array format required by the AlpacaEval evaluator.

Usage:
    uv run python cs336_alignment/section2_zero_shot/evaluate_alpaca_eval.py \
        --model-path /data/a5-alignment/models/Llama-3.1-8B \
        --data-path data/alpaca_eval/alpaca_eval.jsonl \
        --output-path results/section2/alpaca_eval_baseline.json \
        --generator llama-3.1-8b-base

Then evaluate with:
    uv run alpaca_eval \\
        --model_outputs results/section2/alpaca_eval_baseline.json \\
        --annotators_config 'scripts/alpaca_eval_vllm_llama3_3_70b_fn' \\
        --base-dir '.'
"""

import argparse
import json
import time
from pathlib import Path

from vllm import LLM, SamplingParams

SYSTEM_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "zero_shot_system_prompt.prompt"


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text()


def format_prompt(system_prompt: str, instruction: str) -> str:
    return system_prompt.format(instruction=instruction)


def load_alpaca_eval(data_path: Path, max_examples: int | None = None) -> list[dict]:
    examples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
            if max_examples is not None and len(examples) >= max_examples:
                break
    return examples


def main():
    parser = argparse.ArgumentParser(description="Zero-shot AlpacaEval prediction collection")
    parser.add_argument(
        "--model-path",
        default="/data/a5-alignment/models/Llama-3.1-8B",
        help="Path to the model",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/alpaca_eval/alpaca_eval.jsonl"),
        help="Path to AlpacaEval JSONL",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results/section2/alpaca_eval_baseline.json"),
        help="Path to write predictions (JSON array for AlpacaEval evaluator)",
    )
    parser.add_argument(
        "--generator",
        default="llama-3.1-8b-base",
        help="Generator identifier used in the output JSON",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max tokens to generate per example",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run on 10 examples only",
    )
    args = parser.parse_args()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    system_prompt = load_system_prompt()
    print(f"Loading AlpacaEval data from {args.data_path} ...")
    examples = load_alpaca_eval(args.data_path, max_examples=10 if args.smoke_test else None)
    print(f"  Loaded {len(examples)} examples")

    # Build prompts — instruction is the raw AlpacaEval instruction
    prompts = [format_prompt(system_prompt, ex["instruction"]) for ex in examples]

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

    # Build output in AlpacaEval-required format
    predictions = []
    for ex, output in zip(examples, outputs):
        predictions.append({
            "instruction": ex["instruction"],
            "output": output.outputs[0].text.strip(),
            "generator": args.generator,
            "dataset": ex.get("dataset", ""),
        })

    with open(args.output_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"\nPredictions saved to {args.output_path}")
    print(f"Throughput: {throughput:.2f} examples/s")
    print(f"\nTo evaluate, run:")
    print(f"  uv run alpaca_eval \\")
    print(f"      --model_outputs {args.output_path} \\")
    print(f"      --annotators_config 'scripts/alpaca_eval_vllm_llama3_3_70b_fn' \\")
    print(f"      --base-dir '.'")


if __name__ == "__main__":
    main()
