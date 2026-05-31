"""MMLU evaluation for the SFT-tuned Llama 3.1 8B model.

Uses the Alpaca prompt format (same as training) so the model sees the
format it was fine-tuned on. Results go to results/section4/ for direct
comparison against the zero-shot baseline in results/section2/.

Usage:
    uv run python cs336_alignment/section4_eval/evaluate_mmlu_sft.py \
        --model-path assets/sft_ultrachat \
        --data-dir data/mmlu/test \
        --output-path results/section4/eval_mmlu_sft.jsonl
"""

import argparse
import csv
import json
import time
from pathlib import Path

from vllm import LLM, SamplingParams

from cs336_alignment.section2_zero_shot.parse_responses import parse_mmlu_response

# vLLM calls huggingface_hub.file_exists() which validates the model path as a
# HF repo ID (rejects absolute paths with >1 slash). Patch it to return True
# immediately for paths that exist on disk, bypassing the repo ID validation.
def _patch_vllm_local_path():
    try:
        import vllm.transformers_utils.config as _vc
        _orig = _vc.file_exists
        def _patched(path_or_repo, filename, *args, **kwargs):
            if (Path(path_or_repo) / filename).exists():
                return True
            return _orig(path_or_repo, filename, *args, **kwargs)
        _vc.file_exists = _patched
    except Exception:
        pass
_patch_vllm_local_path()

# Inference prompt: Alpaca header + instruction slot + "### Response:\n"
# The model generates the response; "### Instruction:" stops it from hallucinating a second turn.
ALPACA_INFERENCE_TEMPLATE = """\
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

MMLU_INSTRUCTION_TEMPLATE = (
    "Answer the following multiple choice question about {subject}. "
    'Respond with a single sentence of the form "The correct answer is _", '
    "filling the blank with the letter corresponding to the correct answer "
    "(i.e., A, B, C or D).\n\n"
    "Question: {question}\n"
    "A. {opt_a}\n"
    "B. {opt_b}\n"
    "C. {opt_c}\n"
    "D. {opt_d}\n\n"
    "Answer:"
)


def load_mmlu(data_dir: Path, max_subjects: int | None = None) -> list[dict]:
    examples = []
    csv_files = sorted(data_dir.glob("*_test.csv"))
    if max_subjects is not None:
        csv_files = csv_files[:max_subjects]
    for csv_path in csv_files:
        subject = csv_path.stem.replace("_test", "").replace("_", " ")
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 6:
                    continue
                examples.append({
                    "subject": subject,
                    "question": row[0],
                    "options": [row[1], row[2], row[3], row[4]],
                    "answer": row[5].strip().upper(),
                })
    return examples


def main():
    parser = argparse.ArgumentParser(description="MMLU evaluation — SFT model")
    parser.add_argument("--model-path", default="assets/sft_ultrachat")
    parser.add_argument("--data-dir", type=Path, default=Path("data/mmlu/test"))
    parser.add_argument("--output-path", type=Path,
                        default=Path("results/section4/eval_mmlu_sft.jsonl"))
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--smoke-test", action="store_true", help="3 subjects only")
    args = parser.parse_args()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading MMLU data from {args.data_dir} ...")
    examples = load_mmlu(args.data_dir, max_subjects=3 if args.smoke_test else None)
    print(f"  {len(examples)} examples")

    prompts = [
        ALPACA_INFERENCE_TEMPLATE.format(
            instruction=MMLU_INSTRUCTION_TEMPLATE.format(
                subject=ex["subject"],
                question=ex["question"],
                opt_a=ex["options"][0], opt_b=ex["options"][1],
                opt_c=ex["options"][2], opt_d=ex["options"][3],
            )
        )
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
        predicted = parse_mmlu_response(ex, model_output)
        correct = predicted == ex["answer"] if predicted is not None else False
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