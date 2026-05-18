import argparse
import json
from pathlib import Path
from typing import Callable

from vllm import LLM, SamplingParams


def load_examples(data_path: Path) -> list[dict]:
    examples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def get_question(example: dict) -> str:
    # MATH uses "problem"; GSM8K uses "question"
    return example.get("problem") or example.get("question", "")


def get_ground_truth(example: dict) -> str:
    # MATH: full solution string containing \boxed{answer} — grader extracts it
    if "solution" in example:
        return str(example["solution"])
    # GSM8K: answer field ends with "#### <number>"
    raw = str(example.get("answer", ""))
    if "####" in raw:
        return raw.split("####")[-1].strip()
    return raw


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    eval_sampling_params: SamplingParams,
) -> list[dict]:
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    results = []
    for prompt, gt, output in zip(prompts, ground_truths, outputs):
        response = output.outputs[0].text
        rewards = reward_fn(response, gt)
        results.append({
            "prompt": prompt,
            "response": response,
            "ground_truth": gt,
            **rewards,
        })
    return results


def print_summary(results: list[dict]) -> None:
    n = len(results)
    correct     = [r for r in results if r["format_reward"] == 1.0 and r["answer_reward"] == 1.0]
    fmt_wrong   = [r for r in results if r["format_reward"] == 1.0 and r["answer_reward"] == 0.0]
    no_fmt      = [r for r in results if r["format_reward"] == 0.0]

    print(f"\n{'='*60}")
    print(f"Results on {n} examples")
    print(f"{'='*60}")
    print(f"  Correct (format=1, answer=1): {len(correct):5d}  ({100*len(correct)/n:.1f}%)")
    print(f"  Format ok, wrong answer:      {len(fmt_wrong):5d}  ({100*len(fmt_wrong)/n:.1f}%)")
    print(f"  No format:                    {len(no_fmt):5d}  ({100*len(no_fmt)/n:.1f}%)")
    print(f"{'='*60}")

    def show_examples(label: str, bucket: list[dict], n_show: int = 3) -> None:
        print(f"\n--- {label} (showing {min(n_show, len(bucket))}) ---")
        for r in bucket[:n_show]:
            print(f"  GT:       {r['ground_truth'][:120]}")
            print(f"  Response: {r['response'][:200]}")
            print()

    show_examples("Correct", correct)
    show_examples("Format ok, wrong answer", fmt_wrong)
    show_examples("No format", no_fmt)


def check_cuda() -> None:
    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available in this environment.")
        print("  vLLM requires a GPU. Options:")
        print("  1. Run on the cluster (ssh into the H100 node and re-run this script).")
        print("  2. Set up WSL2 CUDA drivers: https://docs.nvidia.com/cuda/wsl-user-guide/")
        raise SystemExit(1)
    print(f"CUDA: {torch.cuda.device_count()} device(s) found — {torch.cuda.get_device_name(0)}")


def main():
    parser = argparse.ArgumentParser(description="Zero-shot MATH baseline evaluation")
    parser.add_argument("--model", default="/data/a5-alignment/models/Qwen2.5-Math-1.5B")
    parser.add_argument("--data", default="/data/a5-alignment/MATH/validation.jsonl")
    parser.add_argument("--output", default="results/section3/zero_shot_eval.jsonl")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Cap number of examples (useful for local smoke tests)")
    args = parser.parse_args()

    check_cuda()

    prompt_path = Path(__file__).parent.parent / "prompts" / "r1_zero.prompt"
    template = prompt_path.read_text()

    examples = load_examples(Path(args.data))
    if args.max_examples:
        examples = examples[:args.max_examples]
    print(f"Loaded {len(examples)} examples from {args.data}")

    prompts = [template.format(question=get_question(ex)) for ex in examples]
    ground_truths = [get_ground_truth(ex) for ex in examples]

    llm = LLM(model=args.model)
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
    results = evaluate_vllm(llm, r1_zero_reward_fn, prompts, ground_truths, sampling_params)

    print_summary(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()