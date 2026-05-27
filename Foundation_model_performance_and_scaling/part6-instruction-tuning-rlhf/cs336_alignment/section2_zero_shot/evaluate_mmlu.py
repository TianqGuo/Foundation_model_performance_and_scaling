"""
Zero-shot MMLU evaluation for Llama 3.1 8B.

Loads all subject CSVs from data/mmlu/test/, formats prompts with the
zero-shot system prompt, runs vLLM inference (greedy decoding), and
computes accuracy.

Usage:
    uv run python cs336_alignment/section2_zero_shot/evaluate_mmlu.py \
        --model-path /data/a5-alignment/models/Llama-3.1-8B \
        --data-dir data/mmlu/test \
        --output-path results/section2/eval_mmlu_baseline.jsonl
"""

import argparse
import csv
import json
import time
from pathlib import Path

from vllm import LLM, SamplingParams

from cs336_alignment.section2_zero_shot.parse_responses import parse_mmlu_response

SYSTEM_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "zero_shot_system_prompt.prompt"

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


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text()


def format_prompt(system_prompt: str, instruction: str) -> str:
    return system_prompt.format(instruction=instruction)


def load_mmlu(data_dir: Path, max_subjects: int | None = None) -> list[dict]:
    """Load all MMLU test CSVs. Each CSV has no header:
       question, A, B, C, D, answer
    """
    examples = []
    csv_files = sorted(data_dir.glob("*_test.csv"))
    if max_subjects is not None:
        csv_files = csv_files[:max_subjects]
    for csv_path in csv_files:
        subject = csv_path.stem.replace("_test", "").replace("_", " ")
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6:
                    continue
                question, opt_a, opt_b, opt_c, opt_d, answer = (
                    row[0], row[1], row[2], row[3], row[4], row[5]
                )
                examples.append({
                    "subject": subject,
                    "question": question,
                    "options": [opt_a, opt_b, opt_c, opt_d],
                    "answer": answer.strip().upper(),
                })
    return examples


def main():
    parser = argparse.ArgumentParser(description="Zero-shot MMLU evaluation")
    parser.add_argument(
        "--model-path",
        default="/data/a5-alignment/models/Llama-3.1-8B",
        help="Path to the model",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/mmlu/test"),
        help="Directory containing MMLU test CSVs",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results/section2/eval_mmlu_baseline.jsonl"),
        help="Path to write evaluation results",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Max tokens to generate per example",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run on 3 subjects only",
    )
    args = parser.parse_args()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    system_prompt = load_system_prompt()
    print(f"Loading MMLU data from {args.data_dir} ...")
    examples = load_mmlu(args.data_dir, max_subjects=3 if args.smoke_test else None)
    print(f"  Loaded {len(examples)} examples")

    # Build prompts
    prompts = []
    for ex in examples:
        instruction = MMLU_INSTRUCTION_TEMPLATE.format(
            subject=ex["subject"],
            question=ex["question"],
            opt_a=ex["options"][0],
            opt_b=ex["options"][1],
            opt_c=ex["options"][2],
            opt_d=ex["options"][3],
        )
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
        predicted = parse_mmlu_response(ex, model_output)
        correct = predicted == ex["answer"] if predicted is not None else False
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

    # Summary
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
