"""SimpleSafetyTests prediction collection for the SFT-tuned Llama 3.1 8B model.

Uses the Alpaca prompt format (same as training). Output is JSONL in the
format required by scripts/evaluate_safety.py.

Usage:
    uv run python cs336_alignment/section4_eval/evaluate_sst_sft.py \
        --model-path assets/sft_ultrachat \
        --data-path data/simple_safety_tests/simple_safety_tests.csv \
        --output-path results/section4/sst_sft.jsonl

Then annotate safety (requires 2× 80 GB GPUs):
    uv run python scripts/evaluate_safety.py \
        --input-path results/section4/sst_sft.jsonl \
        --model-name-or-path assets/Llama-3.3-70B-Instruct \
        --num-gpus 2 \
        --output-path results/section4/sst_sft_annotated.jsonl
"""

import argparse
import csv
import json
import time
from pathlib import Path

from vllm import LLM, SamplingParams

ALPACA_INFERENCE_TEMPLATE = """\
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


def load_sst(data_path: Path, max_examples: int | None = None) -> list[dict]:
    examples = []
    with open(data_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            examples.append(dict(row))
            if max_examples is not None and len(examples) >= max_examples:
                break
    return examples


def main():
    parser = argparse.ArgumentParser(description="SimpleSafetyTests prediction collection — SFT model")
    parser.add_argument("--model-path", default="assets/sft_ultrachat")
    parser.add_argument("--data-path", type=Path,
                        default=Path("data/simple_safety_tests/simple_safety_tests.csv"))
    parser.add_argument("--output-path", type=Path,
                        default=Path("results/section4/sst_sft.jsonl"))
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--smoke-test", action="store_true", help="10 examples only")
    args = parser.parse_args()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading SimpleSafetyTests data from {args.data_path} ...")
    examples = load_sst(args.data_path, max_examples=10 if args.smoke_test else None)
    print(f"  {len(examples)} examples")

    prompts = [
        ALPACA_INFERENCE_TEMPLATE.format(instruction=ex["prompts_final"])
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

    with open(args.output_path, "w") as f:
        for ex, out in zip(examples, outputs):
            record = {**ex, "output": out.outputs[0].text.strip()}
            f.write(json.dumps(record) + "\n")
    print(f"\nPredictions saved to {args.output_path}")
    print(f"Throughput: {throughput:.2f} ex/s")
    print(f"\nTo annotate safety (requires 2× 80 GB GPUs), run:")
    print(f"  uv run python scripts/evaluate_safety.py \\")
    print(f"      --input-path {args.output_path} \\")
    print(f"      --model-name-or-path assets/Llama-3.3-70B-Instruct \\")
    print(f"      --num-gpus 2 \\")
    print(f"      --output-path {args.output_path.with_name('sst_sft_annotated.jsonl')}")


if __name__ == "__main__":
    main()
