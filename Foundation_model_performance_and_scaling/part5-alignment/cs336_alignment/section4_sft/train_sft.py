"""SFT training on MATH reasoning traces.

Run via part_5_4.sh or directly:
    uv run python cs336_alignment/section4_sft/train_sft.py [args]
"""

import argparse
import json
import random
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.section4_sft.helpers import (
    get_response_log_probs,
    log_generations,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)


# ---------------------------------------------------------------------------
# vLLM helpers (from requirements)
# ---------------------------------------------------------------------------

def init_vllm(
    model_id: str,
    device: str,
    seed: int,
    gpu_memory_utilization: float = 0.85,
) -> LLM:
    from vllm.model_executor import set_random_seed as vllm_set_random_seed
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy, llm: LLM) -> None:
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _get_ground_truth(example: dict) -> str:
    """Extract ground-truth answer string from a MATH or GSM8K example."""
    if "solution" in example:
        return str(example["solution"])
    raw = str(example.get("answer", example.get("ground_truth", "")))
    if "####" in raw:
        return raw.split("####")[-1].strip()
    return raw


def load_sft_dataset(
    data_path: Path,
    max_examples: Optional[int] = None,
    filter_correct: bool = False,
) -> list[dict]:
    examples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    if filter_correct:
        filtered = []
        for ex in examples:
            gt = _get_ground_truth(ex)
            if gt:
                rewards = r1_zero_reward_fn(ex.get("response", ""), gt)
                if rewards["answer_reward"] == 1.0:
                    filtered.append(ex)
            else:
                # No ground truth available — keep if response has valid format
                if "<answer>" in ex.get("response", ""):
                    filtered.append(ex)
        print(f"Filtered to correct-answer examples: {len(filtered)} / {len(examples)}")
        examples = filtered

    if max_examples is not None:
        examples = examples[:max_examples]

    print(f"Training examples: {len(examples)}")
    return examples


def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_eval(
    policy,
    vllm_model: LLM,
    tokenizer,
    val_examples: list[dict],
    prompt_template: str,
    sampling_params: SamplingParams,
    train_device: str,
    n_eval: int = 200,
) -> dict:
    """Evaluate policy on a random subset of the MATH validation set."""
    subset = random.sample(val_examples, min(n_eval, len(val_examples)))
    prompts = [
        prompt_template.format(question=ex.get("problem", ex.get("question", "")))
        for ex in subset
    ]
    ground_truths = [_get_ground_truth(ex) for ex in subset]

    load_policy_into_vllm_instance(policy, vllm_model)

    metrics = log_generations(
        vllm_model=vllm_model,
        policy_model=policy,
        tokenizer=tokenizer,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        sampling_params=sampling_params,
        device=train_device,
    )
    return metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- GPU check ---
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA not available. SFT training requires GPU(s).")
    n_gpus = torch.cuda.device_count()
    print(f"GPUs available: {n_gpus}")
    use_vllm_eval = n_gpus >= 2 and not args.skip_eval
    if not use_vllm_eval:
        print("INFO: vLLM evaluation disabled (need ≥2 GPUs or --skip_eval set).")

    # --- wandb ---
    if not args.no_wandb:
        run_name = args.run_name or f"sft_n{args.max_train_examples or 'full'}"
        if args.filter_correct and not run_name.endswith("_filtered"):
            run_name += "_filtered"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        wandb.define_metric("train_step")
        wandb.define_metric("eval_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("eval/*", step_metric="eval_step")

    # --- Model and tokenizer ---
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    policy = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(args.train_device)
    policy.train()
    print(f"Policy on {args.train_device}")

    # --- vLLM ---
    vllm_model = None
    if use_vllm_eval:
        print(f"Initializing vLLM on {args.vllm_device} ...")
        vllm_model = init_vllm(args.model, args.vllm_device, args.seed)
        print("vLLM ready")

    # --- Prompt template ---
    prompt_path = Path(__file__).parent.parent / "prompts" / "r1_zero.prompt"
    prompt_template = prompt_path.read_text()

    # --- Datasets ---
    dataset = load_sft_dataset(Path(args.data), args.max_train_examples, args.filter_correct)
    val_examples = []
    if vllm_model is not None and Path(args.val_data).exists():
        val_examples = load_jsonl(Path(args.val_data))
        print(f"Validation set: {len(val_examples)} examples")

    # Save dataset metadata
    (output_path / "dataset_info.json").write_text(
        json.dumps({"n_train": len(dataset), "filter_correct": args.filter_correct}, indent=2)
    )

    # Per-step eval metrics — appended at every eval so results survive crashes
    run_name = args.run_name or f"sft_n{args.max_train_examples or 'full'}"
    if args.filter_correct and not run_name.endswith("_filtered"):
        run_name += "_filtered"
    metrics_path = output_path / f"eval_metrics_{run_name}.jsonl"
    metrics_file = open(metrics_path, "w")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)
    optimizer.zero_grad()

    eval_sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_response_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    ) if vllm_model is not None else None

    train_step = 0
    eval_step = 0
    microbatch_count = 0

    eff_batch = args.micro_batch_size * args.gradient_accumulation_steps
    print(
        f"\nStarting SFT: {args.n_epochs} epochs | lr={args.lr} | "
        f"micro_bs={args.micro_batch_size} | grad_accum={args.gradient_accumulation_steps} | "
        f"eff_batch={eff_batch}\n"
    )

    for epoch in range(args.n_epochs):
        random.shuffle(dataset)

        for batch_start in range(0, len(dataset), args.micro_batch_size):
            batch = dataset[batch_start : batch_start + args.micro_batch_size]
            if not batch:
                continue

            prompts = [ex["prompt"] for ex in batch]
            responses = [ex["response"] for ex in batch]

            tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)
            input_ids = tokenized["input_ids"].to(args.train_device)
            labels = tokenized["labels"].to(args.train_device)
            response_mask = tokenized["response_mask"].to(args.train_device)

            try:
                log_probs_out = get_response_log_probs(policy, input_ids, labels)
                loss, _ = sft_microbatch_train_step(
                    log_probs_out["log_probs"],
                    response_mask,
                    args.gradient_accumulation_steps,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"WARNING: OOM at batch_start={batch_start}, skipping microbatch")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    microbatch_count = 0
                    continue
                raise

            microbatch_count += 1

            if microbatch_count % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                train_step += 1

                print(f"  epoch={epoch+1} train_step={train_step} loss={loss.item():.4f}")
                if not args.no_wandb:
                    wandb.log({"train/loss": loss.item(), "train_step": train_step})

                # Periodic eval
                if vllm_model is not None and train_step % args.eval_interval == 0:
                    policy.eval()
                    with torch.no_grad():
                        metrics = run_eval(
                            policy, vllm_model, tokenizer,
                            val_examples, prompt_template, eval_sampling_params,
                            args.train_device, args.n_eval_examples,
                        )
                    policy.train()
                    eval_step += 1
                    print(
                        f"  [eval] eval_step={eval_step} "
                        f"accuracy={metrics['accuracy']:.3f} "
                        f"format={metrics['format_rate']:.3f} "
                        f"entropy={metrics['avg_token_entropy']:.3f} "
                        f"avg_len={metrics['avg_response_length']:.0f}"
                    )
                    # Save to disk immediately so results survive crashes
                    metrics_file.write(json.dumps({
                        "train_step": train_step,
                        "eval_step": eval_step,
                        "accuracy": metrics["accuracy"],
                        "format_rate": metrics["format_rate"],
                        "avg_reward": metrics["avg_reward"],
                        "avg_token_entropy": metrics["avg_token_entropy"],
                        "avg_response_length": metrics["avg_response_length"],
                        "avg_response_length_correct": metrics["avg_response_length_correct"],
                        "avg_response_length_incorrect": metrics["avg_response_length_incorrect"],
                    }) + "\n")
                    metrics_file.flush()

                    if not args.no_wandb:
                        wandb.log({
                            "eval/accuracy": metrics["accuracy"],
                            "eval/format_rate": metrics["format_rate"],
                            "eval/avg_reward": metrics["avg_reward"],
                            "eval/avg_token_entropy": metrics["avg_token_entropy"],
                            "eval/avg_response_length": metrics["avg_response_length"],
                            "eval/avg_response_length_correct": metrics["avg_response_length_correct"],
                            "eval/avg_response_length_incorrect": metrics["avg_response_length_incorrect"],
                            "eval_step": eval_step,
                        })

    # --- Final eval ---
    if vllm_model is not None:
        print("\nRunning final evaluation ...")
        policy.eval()
        with torch.no_grad():
            final = run_eval(
                policy, vllm_model, tokenizer,
                val_examples, prompt_template, eval_sampling_params,
                args.train_device, n_eval=min(500, len(val_examples)),
            )
        print(f"Final accuracy: {final['accuracy']:.4f}")
        (output_path / "final_eval.json").write_text(
            json.dumps({k: v for k, v in final.items() if k != "examples"}, indent=2)
        )
        if not args.no_wandb:
            eval_step += 1
            wandb.log({
                "eval/accuracy": final["accuracy"],
                "eval/format_rate": final["format_rate"],
                "eval_step": eval_step,
            })

    # --- Save model ---
    import os
    user = os.environ.get("USER", "user")
    save_name = f"sft_n{args.max_train_examples or 'full'}"
    if args.filter_correct:
        save_name += "_filtered"
    cluster_dir = Path(f"/data/{user}/{save_name}")
    local_dir = Path(__file__).parent.parent.parent / "assets" / save_name
    save_dir = cluster_dir if cluster_dir.parent.exists() else local_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")

    metrics_file.close()
    print(f"Eval metrics saved to {metrics_path}")

    if not args.no_wandb:
        wandb.finish()

    print(f"Results saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT on MATH reasoning traces")
    # Paths
    parser.add_argument("--model", default="/data/a5-alignment/models/Qwen2.5-Math-1.5B")
    parser.add_argument("--data", default="/data/a5-alignment/MATH/sft.jsonl")
    parser.add_argument("--val_data", default="/data/a5-alignment/MATH/validation.jsonl")
    parser.add_argument("--output", default="results/section4")
    # Dataset
    parser.add_argument("--max_train_examples", type=int, default=None,
                        help="Limit training examples for ablation (128/256/512/1024/None=full)")
    parser.add_argument("--filter_correct", action="store_true",
                        help="Keep only SFT examples whose response answers correctly")
    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--max_response_tokens", type=int, default=1024)
    # Evaluation
    parser.add_argument("--eval_interval", type=int, default=50,
                        help="Run vLLM eval every N optimizer steps")
    parser.add_argument("--n_eval_examples", type=int, default=200)
    parser.add_argument("--skip_eval", action="store_true",
                        help="Disable vLLM evaluation (single-GPU or fast local runs)")
    # Devices
    parser.add_argument("--train_device", default="cuda:0")
    parser.add_argument("--vllm_device", default="cuda:1")
    # Logging
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", default="cs336-alignment-sft")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())