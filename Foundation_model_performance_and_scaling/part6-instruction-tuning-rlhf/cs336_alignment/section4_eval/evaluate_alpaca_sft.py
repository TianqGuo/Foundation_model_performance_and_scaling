"""AlpacaEval prediction collection for the SFT-tuned Llama 3.1 8B model.

Uses the Alpaca prompt format (same as training). Output is a JSON array
in the format required by the alpaca_eval evaluator.

Usage:
    uv run python cs336_alignment/section4_eval/evaluate_alpaca_sft.py \
        --model-path assets/sft_ultrachat \
        --data-path data/alpaca_eval/alpaca_eval.jsonl \
        --output-path results/section4/alpaca_eval_sft.json

Then evaluate (requires 2× 80 GB GPUs):
    cd <project-root>
    uv run alpaca_eval \
        --model_outputs results/section4/alpaca_eval_sft.json \
        --annotators_config scripts/alpaca_eval_vllm_llama3_3_70b_fn \
        --base-dir .
"""

import argparse
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
    parser = argparse.ArgumentParser(description="AlpacaEval prediction collection — SFT model")
    parser.add_argument("--model-path", default="assets/sft_ultrachat")
    parser.add_argument("--data-path", type=Path,
                        default=Path("data/alpaca_eval/alpaca_eval.jsonl"))
    parser.add_argument("--output-path", type=Path,
                        default=Path("results/section4/alpaca_eval_sft.json"))
    parser.add_argument("--generator", default="llama-3.1-8b-sft")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--smoke-test", action="store_true", help="10 examples only")
    args = parser.parse_args()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading AlpacaEval data from {args.data_path} ...")
    examples = load_alpaca_eval(args.data_path, max_examples=10 if args.smoke_test else None)
    print(f"  {len(examples)} examples")

    prompts = [
        ALPACA_INFERENCE_TEMPLATE.format(instruction=ex["instruction"])
        for ex in examples
    ]

    print(f"Loading model from {args.model_path} ...")
    if Path(args.model_path).exists():
        import os; os.environ.setdefault("HF_HUB_OFFLINE", "1")
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

    predictions = [
        {
            "instruction": ex["instruction"],
            "output": out.outputs[0].text.strip(),
            "generator": args.generator,
            "dataset": ex.get("dataset", ""),
        }
        for ex, out in zip(examples, outputs)
    ]

    with open(args.output_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"\nPredictions saved to {args.output_path}")
    print(f"Throughput: {throughput:.2f} ex/s")
    print(f"\nTo evaluate (requires 2× 80 GB GPUs), run from project root:")
    print(f"  uv run alpaca_eval \\")
    print(f"      --model_outputs {args.output_path} \\")
    print(f"      --annotators_config scripts/alpaca_eval_vllm_llama3_3_70b_fn \\")
    print(f"      --base-dir .")


if __name__ == "__main__":
    main()