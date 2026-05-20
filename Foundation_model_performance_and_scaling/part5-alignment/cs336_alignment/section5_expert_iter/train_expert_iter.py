"""Expert Iteration (EI) training on MATH.

EI algorithm: for each step, rollout G responses per training question with vLLM,
keep those with reward > 0, fine-tune the policy on the filtered set, evaluate.
Starts from the base model — not the SFT checkpoint.

Run via part_5_5.sh or directly:
    uv run python cs336_alignment/section5_expert_iter/train_expert_iter.py [args]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# Rollout helpers
# ---------------------------------------------------------------------------

def generate_ei_rollouts(
    vllm_model: LLM,
    prompts: list[str],
    ground_truths: list[str],
    sampling_params: SamplingParams,
) -> list[dict]:
    """Generate G responses per prompt via vLLM; keep those with reward > 0.

    sampling_params must have n=G set.
    Returns list of {prompt, response} dicts ready for SFT training.
    """
    outputs = vllm_model.generate(prompts, sampling_params)
    filtered = []
    for prompt, gt, output in zip(prompts, ground_truths, outputs):
        for completion in output.outputs:
            response = completion.text
            if r1_zero_reward_fn(response, gt).get("reward", 0) > 0:
                filtered.append({"prompt": prompt, "response": response})
    return filtered


# ---------------------------------------------------------------------------
# SFT inner loop (one EI step)
# ---------------------------------------------------------------------------

def run_sft_on_rollouts(
    policy,
    tokenizer,
    rollout_data: list[dict],
    n_epochs: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    optimizer,
    train_device: str,
    train_step: int,
    no_wandb: bool,
) -> tuple[int, float]:
    """Fine-tune policy on rollout_data for n_epochs. Returns (updated train_step, last loss)."""
    import torch
    import wandb
    from cs336_alignment.section4_sft.helpers import (
        get_response_log_probs,
        sft_microbatch_train_step,
        tokenize_prompt_and_output,
    )

    last_loss = 0.0
    microbatch_count = 0

    for epoch in range(n_epochs):
        random.shuffle(rollout_data)
        for batch_start in range(0, len(rollout_data), micro_batch_size):
            batch = rollout_data[batch_start : batch_start + micro_batch_size]
            if not batch:
                continue

            prompts = [ex["prompt"] for ex in batch]
            responses = [ex["response"] for ex in batch]

            tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)
            input_ids = tokenized["input_ids"].to(train_device)
            labels = tokenized["labels"].to(train_device)
            response_mask = tokenized["response_mask"].to(train_device)

            try:
                log_probs_out = get_response_log_probs(policy, input_ids, labels)
                loss, _ = sft_microbatch_train_step(
                    log_probs_out["log_probs"],
                    response_mask,
                    gradient_accumulation_steps,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  WARNING: OOM at batch_start={batch_start}, skipping microbatch")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    microbatch_count = 0
                    continue
                raise

            microbatch_count += 1

            if microbatch_count % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                train_step += 1
                last_loss = loss.item()
                print(f"    train_step={train_step} loss={last_loss:.4f}")
                if not no_wandb:
                    wandb.log({"train/loss": last_loss, "train_step": train_step})

    # flush any remaining accumulated gradients
    if microbatch_count % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        train_step += 1

    return train_step, last_loss


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    import torch
    import wandb
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import SamplingParams
    from cs336_alignment.section4_sft.helpers import log_generations
    from cs336_alignment.section4_sft.train_sft import init_vllm, load_policy_into_vllm_instance

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA not available. EI training requires GPU(s).")
    n_gpus = torch.cuda.device_count()
    # Single-GPU mode: share cuda:0 for both policy and vLLM (lower memory utilization)
    single_gpu = n_gpus == 1
    if single_gpu:
        args.vllm_device = args.train_device
    use_vllm = not args.skip_eval
    print(f"GPUs: {n_gpus}  |  single_gpu: {single_gpu}  |  vLLM rollout+eval: {use_vllm}")

    run_name = args.run_name or f"ei_g{args.G}"
    if not args.no_wandb:
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        wandb.define_metric("ei_step")
        wandb.define_metric("train_step")
        wandb.define_metric("eval/*", step_metric="ei_step")
        wandb.define_metric("train/*", step_metric="train_step")

    # Model and tokenizer
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

    # vLLM: second GPU for full runs, same GPU for single-GPU smoke tests
    vllm_model = None
    if use_vllm:
        # Single-GPU: reduce vLLM memory to leave room for policy gradients
        gpu_memory_utilization = 0.5 if single_gpu else 0.85
        print(f"Initializing vLLM on {args.vllm_device} (mem_util={gpu_memory_utilization}) ...")
        vllm_model = init_vllm(
            args.model, args.vllm_device, args.seed,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        print("vLLM ready")

    # Prompt template
    prompt_path = Path(__file__).parent.parent / "prompts" / "r1_zero.prompt"
    prompt_template = prompt_path.read_text()

    # Load training data (train.jsonl: {problem, solution})
    train_examples = []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if line:
                train_examples.append(json.loads(line))
    if args.max_train_examples:
        train_examples = train_examples[: args.max_train_examples]
    print(f"Training questions: {len(train_examples)}")

    train_prompts = [prompt_template.format(question=ex["problem"]) for ex in train_examples]
    train_gts = [str(ex["solution"]) for ex in train_examples]

    # Validation data
    val_examples = []
    if vllm_model is not None and Path(args.val_data).exists():
        with open(args.val_data) as f:
            for line in f:
                line = line.strip()
                if line:
                    val_examples.append(json.loads(line))
        print(f"Validation examples: {len(val_examples)}")

    # Sampling params
    rollout_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_response_tokens,
        min_tokens=4,
        n=args.G,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    ) if vllm_model is not None else None

    eval_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_response_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    ) if vllm_model is not None else None

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)
    optimizer.zero_grad()

    metrics_path = output_path / f"eval_metrics_{run_name}.jsonl"
    metrics_file = open(metrics_path, "w")

    train_step = 0

    # ---- Expert Iteration loop ----
    for ei_step in range(1, args.n_ei_steps + 1):
        print(f"\n{'='*60}")
        print(f"EI step {ei_step}/{args.n_ei_steps}  (G={args.G})")
        print(f"{'='*60}")

        # 1. Rollout
        if vllm_model is not None:
            print("Syncing policy → vLLM for rollout...")
            load_policy_into_vllm_instance(policy, vllm_model)
            print(f"Generating rollouts ({len(train_prompts)} questions × G={args.G})...")
            rollout_data = generate_ei_rollouts(
                vllm_model, train_prompts, train_gts, rollout_params
            )
            n_total = len(train_prompts) * args.G
            print(f"Filtered rollouts: {len(rollout_data)} / {n_total} "
                  f"({len(rollout_data)/n_total:.1%} correct)")
        else:
            print("WARNING: vLLM not available — skipping rollout (no-GPU mode)")
            rollout_data = []

        if not rollout_data:
            print("No correct rollouts this step; skipping fine-tune.")
            continue

        # 2. Fine-tune
        print(f"Fine-tuning on {len(rollout_data)} examples for {args.n_sft_epochs} epoch(s)...")
        policy.train()
        train_step, _ = run_sft_on_rollouts(
            policy=policy,
            tokenizer=tokenizer,
            rollout_data=rollout_data,
            n_epochs=args.n_sft_epochs,
            micro_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            optimizer=optimizer,
            train_device=args.train_device,
            train_step=train_step,
            no_wandb=args.no_wandb,
        )

        # 3. Evaluate
        if vllm_model is not None:
            print(f"Evaluating on {args.n_eval_examples} validation examples...")
            load_policy_into_vllm_instance(policy, vllm_model)
            policy.eval()
            with torch.no_grad():
                subset = random.sample(val_examples, min(args.n_eval_examples, len(val_examples)))
                eval_prompts = [
                    prompt_template.format(question=ex.get("problem", ex.get("question", "")))
                    for ex in subset
                ]
                eval_gts = [
                    str(ex.get("solution", ex.get("answer", ""))) for ex in subset
                ]
                metrics = log_generations(
                    vllm_model=vllm_model,
                    policy_model=policy,
                    tokenizer=tokenizer,
                    reward_fn=r1_zero_reward_fn,
                    prompts=eval_prompts,
                    ground_truths=eval_gts,
                    sampling_params=eval_params,
                    device=args.train_device,
                )
            policy.train()

            print(
                f"  [eval] ei_step={ei_step} "
                f"accuracy={metrics['accuracy']:.3f} "
                f"format={metrics['format_rate']:.3f} "
                f"entropy={metrics['avg_token_entropy']:.3f} "
                f"n_rollout={len(rollout_data)}"
            )

            record = {
                "ei_step": ei_step,
                "train_step": train_step,
                "n_rollout": len(rollout_data),
                "accuracy": metrics["accuracy"],
                "format_rate": metrics["format_rate"],
                "avg_reward": metrics["avg_reward"],
                "avg_token_entropy": metrics["avg_token_entropy"],
                "avg_response_length": metrics["avg_response_length"],
                "avg_response_length_correct": metrics["avg_response_length_correct"],
                "avg_response_length_incorrect": metrics["avg_response_length_incorrect"],
            }
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()

            if not args.no_wandb:
                wandb.log({
                    "eval/accuracy": metrics["accuracy"],
                    "eval/format_rate": metrics["format_rate"],
                    "eval/avg_reward": metrics["avg_reward"],
                    "eval/avg_token_entropy": metrics["avg_token_entropy"],
                    "eval/avg_response_length": metrics["avg_response_length"],
                    "eval/n_rollout": len(rollout_data),
                    "ei_step": ei_step,
                })

    metrics_file.close()
    if not args.no_wandb:
        wandb.finish()
    print(f"\nEI training complete. Results at {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Expert Iteration on MATH")
    parser.add_argument("--model", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--val_data", default=None)
    parser.add_argument("--output", default="results/section5")
    parser.add_argument("--run_name", default=None)
    # EI hyperparameters
    parser.add_argument("--n_ei_steps", type=int, default=5)
    parser.add_argument("--G", type=int, default=4, help="Rollouts per question per EI step")
    parser.add_argument("--n_sft_epochs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    # SFT inner-loop hyperparameters
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_response_tokens", type=int, default=1024)
    parser.add_argument("--max_train_examples", type=int, default=None)
    # Eval
    parser.add_argument("--n_eval_examples", type=int, default=200)
    # Infrastructure
    parser.add_argument("--train_device", default="cuda:0")
    parser.add_argument("--vllm_device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="cs336-alignment")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
