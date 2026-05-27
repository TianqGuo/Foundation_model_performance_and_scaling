"""
Zero-shot GSM8K evaluation for Llama 3.1 8B.

Loads data/gsm8k/test.jsonl, formats prompts with the zero-shot system
prompt, runs vLLM inference (greedy decoding), and computes accuracy.

Usage:
    uv run python cs336_alignment/section2_zero_shot/evaluate_gsm8k.py \
        --model-path /data/a5-alignment/models/Llama-3.1-8B \
        --data-path data/gsm8k/test.jsonl \
        --output-path results/section2/eval_gsm8k_baseline.jsonl
"""

import argparse
import json
import time
from pathlib import Path

from vllm import LLM, SamplingParams

from cs336_alignment.section2_zero_shot.parse_responses import parse_gsm8k_response

SYSTEM_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "zero_shot_system_prompt.prompt"


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text()


def format_prompt(system_prompt: str, instruction: str) -> str:
    return system_prompt.format(instruction=instruction)


def extract_gold_answer(answer_str: str) -> str:
    """Extract the numeric answer after '####' in GSM8K answer strings."""
    if "####" in answer_str:
        return answer_str.split("####")[-1].strip().replace(",", "")
    # Fallback: last number in the string
    import re
    numbers = re.findall(r'\b\d[\d,]*(?:\.\d+)?\b', answer_str)
    return numbers[-1].replace(",", "") if numbers else answer_str.strip()


def load_gsm8k(data_path: Path, max_examples: int | None = None) -> list[dict]:
    examples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            examples.append({
                "question": ex["question"],
                "gold_answer": extract_gold_answer(ex["answer"]),
                "raw_answer": ex["answer"],
            })
            if max_examples is not None and len(examples) >= max_examples:
                break
    return examples


def main():
    parser = argparse.ArgumentParser(description="Zero-shot GSM8K evaluation")
    parser.add_argument(
        "--model-path",
        default="/data/a5-alignment/models/Llama-3.1-8B",
        help="Path to the model",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/gsm8k/test.jsonl"),
        help="Path to GSM8K test JSONL",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results/section2/eval_gsm8k_baseline.jsonl"),
        help="Path to write evaluation results",
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
        help="Run on 20 examples only",
    )
    args = parser.parse_args()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    system_prompt = load_system_prompt()
    print(f"Loading GSM8K data from {args.data_path} ...")
    examples = load_gsm8k(args.data_path, max_examples=20 if args.smoke_test else None)
    print(f"  Loaded {len(examples)} examples")

    # Build prompts — instruction is "{question}\nAnswer:"
    prompts = []
    for ex in examples:
        instruction = f"{ex['question']}\nAnswer:"
        prompts.append(format_prompt(system_prompt, instruction))

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

    # Parse and score
    results = []
    n_correct = n_parse_fail = 0
    for ex, output in zip(examples, outputs):
        model_output = output.outputs[0].text
        predicted = parse_gsm8k_response(model_output)
        correct = predicted == ex["gold_answer"] if predicted is not None else False
        if predicted is None:
            n_parse_fail += 1
        if correct:
            n_correct += 1
        results.append({
            **ex,
            "model_output": model_output,
            "predicted": predicted,
            "correct": correct,
        })

    accuracy = n_correct / len(results)
    print(f"\nResults:")
    print(f"  Accuracy:       {accuracy:.3f} ({n_correct}/{len(results)})")
    print(f"  Parse failures: {n_parse_fail} ({n_parse_fail/len(results):.3f})")
    print(f"  Throughput:     {throughput:.2f} examples/s")

    # Save
    with open(args.output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to {args.output_path}")

    summary = {
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n_total": len(results),
        "n_parse_fail": n_parse_fail,
        "throughput_examples_per_sec": throughput,
        "elapsed_sec": elapsed,
    }
    summary_path = args.output_path.with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
