"""GSM8K evaluation for the SFT-tuned Llama 3.1 8B model.

Uses the Alpaca prompt format (same as training). Results go to
results/section4/ for direct comparison against the zero-shot baseline.

Usage:
    uv run python cs336_alignment/section4_eval/evaluate_gsm8k_sft.py \
        --model-path assets/sft_ultrachat \
        --data-path data/gsm8k/test.jsonl \
        --output-path results/section4/eval_gsm8k_sft.jsonl
"""

import argparse
import json
import re
import time
from pathlib import Path

from vllm import LLM, SamplingParams

from cs336_alignment.section2_zero_shot.parse_responses import parse_gsm8k_response

ALPACA_INFERENCE_TEMPLATE = """\
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


def extract_gold_answer(answer_str: str) -> str:
    if "####" in answer_str:
        return answer_str.split("####")[-1].strip().replace(",", "")
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
    parser = argparse.ArgumentParser(description="GSM8K evaluation — SFT model")
    parser.add_argument("--model-path", default="assets/sft_ultrachat")
    parser.add_argument("--data-path", type=Path, default=Path("data/gsm8k/test.jsonl"))
    parser.add_argument("--output-path", type=Path,
                        default=Path("results/section4/eval_gsm8k_sft.jsonl"))
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--smoke-test", action="store_true", help="20 examples only")
    args = parser.parse_args()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading GSM8K data from {args.data_path} ...")
    examples = load_gsm8k(args.data_path, max_examples=20 if args.smoke_test else None)
    print(f"  {len(examples)} examples")

    prompts = [
        ALPACA_INFERENCE_TEMPLATE.format(instruction=f"{ex['question']}\nAnswer:")
        for ex in examples
    ]

    print(f"Loading model from {args.model_path} ...")
    llm = LLM(model=args.model_path, dtype="bfloat16", gpu_memory_utilization=0.85)
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0,
        max_tokens=args.max_tokens,
        stop=["### Instruction:"],
    )

    print("Running inference ...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    throughput = len(prompts) / elapsed
    print(f"  {len(prompts)} examples in {elapsed:.1f}s ({throughput:.2f} ex/s)")

    results = []
    n_correct = n_parse_fail = 0
    for ex, out in zip(examples, outputs):
        model_output = out.outputs[0].text
        predicted = parse_gsm8k_response(model_output)
        correct = predicted == ex["gold_answer"] if predicted is not None else False
        if predicted is None:
            n_parse_fail += 1
        if correct:
            n_correct += 1
        results.append({**ex, "model_output": model_output,
                        "predicted": predicted, "correct": correct})

    accuracy = n_correct / len(results)
    print(f"\nAccuracy:       {accuracy:.3f} ({n_correct}/{len(results)})")
    print(f"Parse failures: {n_parse_fail}")
    print(f"Throughput:     {throughput:.2f} ex/s")

    with open(args.output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Results saved to {args.output_path}")

    summary = {
        "accuracy": accuracy, "n_correct": n_correct, "n_total": len(results),
        "n_parse_fail": n_parse_fail,
        "throughput_examples_per_sec": throughput, "elapsed_sec": elapsed,
    }
    summary_path = args.output_path.with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()