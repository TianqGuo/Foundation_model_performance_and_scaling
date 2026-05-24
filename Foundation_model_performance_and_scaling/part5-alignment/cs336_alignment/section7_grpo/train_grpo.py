"""GRPO training on MATH with verified rewards.

Run via part_5_7.sh or directly:
    uv run python cs336_alignment/section7_grpo/train_grpo.py [args]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from unittest.mock import patch

import torch
import torch.nn.functional as F

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# vLLM helpers
# ---------------------------------------------------------------------------

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85) -> LLM:
    from vllm import LLM
    from vllm.model_executor import set_random_seed as vllm_set_seed
    vllm_set_seed(seed)
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


def load_policy_into_vllm(policy: torch.nn.Module, llm: LLM) -> None:
    state_dict = {k: v.cpu() for k, v in policy.state_dict().items()}
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def get_ground_truth(ex: dict) -> str:
    if "solution" in ex:
        return str(ex["solution"])
    raw = str(ex.get("answer", ex.get("ground_truth", "")))
    return raw.split("####")[-1].strip() if "####" in raw else raw


# ---------------------------------------------------------------------------
# Old log-prob precomputation
# ---------------------------------------------------------------------------

def precompute_old_log_probs(
    policy: torch.nn.Module,
    rollout_prompts: list[str],
    rollout_responses: list[str],
    tokenizer,
    micro_batch_size: int,
    device: str,
) -> torch.Tensor:
    """Compute log-probs for all rollout examples under the current (frozen) policy.

    Returns a tensor of shape (rollout_batch_size, max_seq_len) padded with zeros
    at positions beyond each sequence's actual length.
    """
    from cs336_alignment.section4_sft.helpers import get_response_log_probs, tokenize_prompt_and_output

    chunks: list[torch.Tensor] = []
    policy.eval()
    with torch.no_grad():
        for start in range(0, len(rollout_prompts), micro_batch_size):
            end = min(start + micro_batch_size, len(rollout_prompts))
            tok = tokenize_prompt_and_output(
                rollout_prompts[start:end], rollout_responses[start:end], tokenizer
            )
            lp = get_response_log_probs(
                policy, tok["input_ids"].to(device), tok["labels"].to(device)
            )["log_probs"]          # (mb, seq_len)
            chunks.append(lp.cpu())
    policy.train()

    # Pad each chunk to a common max length and concatenate
    max_sl = max(c.shape[1] for c in chunks)
    padded = [F.pad(c, (0, max_sl - c.shape[1])) for c in chunks]
    return torch.cat(padded, dim=0)   # (rollout_batch_size, max_sl)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_eval(
    policy: torch.nn.Module,
    vllm_model: LLM,
    tokenizer,
    val_examples: list[dict],
    prompt_template: str,
    eval_sampling_params: SamplingParams,
    device: str,
    n_eval: int = 1024,
    reward_fn=None,
) -> dict:
    from cs336_alignment.section4_sft.helpers import log_generations

    if reward_fn is None:
        reward_fn = r1_zero_reward_fn

    subset = random.sample(val_examples, min(n_eval, len(val_examples)))
    prompts = [prompt_template.format(question=ex.get("problem", ex.get("question", ""))) for ex in subset]
    ground_truths = [get_ground_truth(ex) for ex in subset]

    load_policy_into_vllm(policy, vllm_model)
    return log_generations(
        vllm_model=vllm_model,
        policy_model=policy,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        sampling_params=eval_sampling_params,
        device=device,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    import wandb
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import SamplingParams

    from cs336_alignment.section4_sft.helpers import get_response_log_probs, tokenize_prompt_and_output
    from cs336_alignment.section7_grpo.helpers import (
        compute_group_normalized_rewards,
        grpo_microbatch_train_step,
        masked_mean,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- GPU setup ---
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA not available.")
    n_gpus = torch.cuda.device_count()
    print(f"GPUs available: {n_gpus}")
    use_two_gpus = n_gpus >= 2 and not args.skip_eval
    train_device = args.train_device
    vllm_device = args.vllm_device if use_two_gpus else args.train_device
    vllm_mem = args.gpu_memory_utilization if use_two_gpus else 0.30
    if not use_two_gpus:
        print("INFO: single-GPU mode — vLLM shares cuda:0 with policy (vllm_mem=0.30).")

    # --- wandb ---
    if not args.no_wandb:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
        wandb.define_metric("grpo_step")
        wandb.define_metric("train_step")
        wandb.define_metric("eval_step")
        wandb.define_metric("grpo/*", step_metric="grpo_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("eval/*", step_metric="eval_step")

    # --- Model ---
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    policy = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(train_device)
    if args.gradient_checkpointing:
        policy.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled.")
    policy.train()

    # --- vLLM ---
    vllm_model = None
    if not args.skip_eval or True:   # always need vLLM for rollouts
        print(f"Initializing vLLM on {vllm_device} ...")
        vllm_model = init_vllm(args.model, vllm_device, args.seed, vllm_mem)
        print("vLLM ready")

    # --- Prompt template and reward function ---
    prompt_file = "question_only.prompt" if args.prompt_type == "question_only" else "r1_zero.prompt"
    prompt_path = Path(__file__).parent.parent / "prompts" / prompt_file
    prompt_template = prompt_path.read_text()
    reward_fn = question_only_reward_fn if args.prompt_type == "question_only" else r1_zero_reward_fn
    if args.prompt_type == "question_only":
        print("Using question_only prompt and reward function.")

    # --- Data ---
    train_examples = load_jsonl(Path(args.data))
    if args.max_train_examples:
        train_examples = train_examples[: args.max_train_examples]
    print(f"Train examples: {len(train_examples)}")

    val_examples = []
    if Path(args.val_data).exists():
        val_examples = load_jsonl(Path(args.val_data))
        print(f"Val examples: {len(val_examples)}")

    # --- Sampling params ---
    stop_tokens = ["</answer>"] if args.prompt_type == "r1_zero" else []
    rollout_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_response_tokens,
        min_tokens=4,
        n=args.group_size,
        stop=stop_tokens,
        include_stop_str_in_output=True,
    )
    eval_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_response_tokens,
        stop=stop_tokens,
        include_stop_str_in_output=True,
    )

    # --- Derived hyperparameters ---
    n_prompts = args.rollout_batch_size // args.group_size
    micro_bs = args.train_batch_size // args.gradient_accumulation_steps
    assert args.rollout_batch_size % args.group_size == 0
    assert args.train_batch_size % args.gradient_accumulation_steps == 0
    assert args.train_batch_size >= args.group_size
    print(
        f"n_prompts/step={n_prompts}, group_size={args.group_size}, "
        f"rollout_bs={args.rollout_batch_size}, micro_bs={micro_bs}, "
        f"grad_accum={args.gradient_accumulation_steps}"
    )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95)
    )
    optimizer.zero_grad()

    # --- Metrics file ---
    metrics_path = output_path / f"eval_metrics_{args.run_name}.jsonl"
    metrics_file = open(metrics_path, "w")

    train_step = 0
    eval_step = 0
    needs_old_lp = (args.loss_type in ("grpo_clip", "grpo_no_clip")) or (args.epochs_per_rollout_batch > 1)

    # -----------------------------------------------------------------------
    # GRPO loop
    # -----------------------------------------------------------------------
    for grpo_step in range(1, args.n_grpo_steps + 1):
        print(f"\n=== GRPO step {grpo_step}/{args.n_grpo_steps} ===")

        # --- Sync policy weights into vLLM ---
        policy.eval()
        load_policy_into_vllm(policy, vllm_model)

        # --- Sample questions and generate rollouts ---
        batch_qs = random.sample(train_examples, min(n_prompts, len(train_examples)))
        prompts = [prompt_template.format(question=ex.get("problem", ex.get("question", ""))) for ex in batch_qs]
        ground_truths = [get_ground_truth(ex) for ex in batch_qs]

        vllm_outputs = vllm_model.generate(prompts, rollout_params)

        # Flatten: [q1_r1, q1_r2, ..., q1_rG, q2_r1, ...]
        rollout_prompts: list[str] = []
        rollout_responses: list[str] = []
        rollout_gts: list[str] = []
        for prompt, gt, out in zip(prompts, ground_truths, vllm_outputs):
            for completion in out.outputs:
                rollout_prompts.append(prompt)
                rollout_responses.append(completion.text)
                rollout_gts.append(gt)
        rollout_bs = len(rollout_responses)

        # --- Compute rewards and advantages ---
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=rollout_gts,
            group_size=args.group_size,
            advantage_eps=args.advantage_eps,
            normalize_by_std=args.use_std_normalization,
        )

        print(
            f"  rewards: mean={reward_meta['mean_reward']:.3f} "
            f"correct={reward_meta['fraction_correct']:.3f} "
            f"format={reward_meta['mean_format_reward']:.3f}"
        )

        if not args.no_wandb:
            wandb.log({
                "grpo/mean_reward": reward_meta["mean_reward"],
                "grpo/fraction_correct": reward_meta["fraction_correct"],
                "grpo/mean_format_reward": reward_meta["mean_format_reward"],
                "grpo/mean_answer_reward": reward_meta["mean_answer_reward"],
                "grpo/rollout_batch_size": rollout_bs,
                "grpo_step": grpo_step,
            })

        # --- Precompute old log-probs (for off-policy or grpo_clip) ---
        all_old_log_probs: torch.Tensor | None = None
        if needs_old_lp:
            all_old_log_probs = precompute_old_log_probs(
                policy, rollout_prompts, rollout_responses, tokenizer, micro_bs, train_device
            )

        policy.train()

        # Accumulators across all epochs in this GRPO step (for JSONL logging)
        step_grad_norms: list[float] = []
        step_clip_fracs: list[float] = []

        # --- Training epochs on this rollout batch ---
        for epoch in range(args.epochs_per_rollout_batch):
            perm = list(range(rollout_bs))
            random.shuffle(perm)

            optimizer.zero_grad()
            microbatch_count = 0
            epoch_loss = 0.0
            epoch_clip_frac = 0.0
            epoch_entropy = 0.0
            n_mb = 0

            for mb_start in range(0, rollout_bs, micro_bs):
                mb_end = min(mb_start + micro_bs, rollout_bs)
                mb_idx = perm[mb_start:mb_end]

                mb_p = [rollout_prompts[i] for i in mb_idx]
                mb_r = [rollout_responses[i] for i in mb_idx]
                mb_adv = advantages[mb_idx].unsqueeze(1).to(train_device)
                mb_raw = raw_rewards[mb_idx].unsqueeze(1).to(train_device)

                tok = tokenize_prompt_and_output(mb_p, mb_r, tokenizer)
                input_ids = tok["input_ids"].to(train_device)
                labels = tok["labels"].to(train_device)
                response_mask = tok["response_mask"].to(train_device)

                try:
                    lp_out = get_response_log_probs(
                        policy, input_ids, labels, return_token_entropy=True
                    )
                    policy_lp = lp_out["log_probs"]       # (mb, seq_len)
                    token_ent = lp_out.get("token_entropy")

                    mb_old_lp: torch.Tensor | None = None
                    if all_old_log_probs is not None:
                        curr_sl = policy_lp.shape[1]
                        mb_old_lp = all_old_log_probs[mb_idx, :curr_sl].to(train_device)

                    loss, meta = grpo_microbatch_train_step(
                        policy_log_probs=policy_lp,
                        response_mask=response_mask,
                        gradient_accumulation_steps=args.gradient_accumulation_steps,
                        loss_type=args.loss_type,
                        raw_rewards=mb_raw if args.loss_type == "no_baseline" else None,
                        advantages=mb_adv if args.loss_type != "no_baseline" else None,
                        old_log_probs=mb_old_lp,
                        cliprange=args.cliprange,
                        length_norm=args.length_norm,
                        max_response_tokens=args.max_response_tokens,
                    )

                except RuntimeError as e:
                    if "out of memory" in str(e).lower() or "cuda error" in str(e).lower():
                        print(f"WARNING: OOM at mb_start={mb_start}, skipping microbatch")
                        try:
                            torch.cuda.empty_cache()
                        except RuntimeError:
                            pass
                        optimizer.zero_grad()
                        microbatch_count = 0
                        continue
                    raise

                epoch_loss += loss.item() * args.gradient_accumulation_steps
                if "is_clipped" in meta and response_mask.any():
                    clip_frac = masked_mean(meta["is_clipped"], response_mask.float()).item()
                    epoch_clip_frac += clip_frac
                if token_ent is not None and response_mask.any():
                    epoch_entropy += masked_mean(token_ent, response_mask.float()).item()
                n_mb += 1
                microbatch_count += 1

                if microbatch_count % args.gradient_accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0).item()
                    optimizer.step()
                    optimizer.zero_grad()
                    train_step += 1
                    microbatch_count = 0
                    step_grad_norms.append(grad_norm)

                    avg_loss = epoch_loss / max(n_mb, 1)
                    avg_clip = epoch_clip_frac / max(n_mb, 1)
                    avg_ent = epoch_entropy / max(n_mb, 1)
                    step_clip_fracs.append(avg_clip)
                    print(
                        f"  train_step={train_step} loss={avg_loss:.4f} "
                        f"grad_norm={grad_norm:.3f} entropy={avg_ent:.3f}"
                        + (f" clip_frac={avg_clip:.3f}" if args.loss_type in ("grpo_clip", "grpo_no_clip") else "")
                    )

                    if not args.no_wandb:
                        log_dict: dict = {
                            "train/loss": avg_loss,
                            "train/grad_norm": grad_norm,
                            "train/token_entropy": avg_ent,
                            "train/mean_reward": reward_meta["mean_reward"],
                            "train/fraction_correct": reward_meta["fraction_correct"],
                            "train_step": train_step,
                        }
                        if args.loss_type in ("grpo_clip", "grpo_no_clip"):
                            log_dict["train/clip_fraction"] = avg_clip
                        wandb.log(log_dict)

            # Flush any remaining accumulated gradients
            if microbatch_count > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0).item()
                optimizer.step()
                optimizer.zero_grad()
                train_step += 1

        # --- Periodic evaluation ---
        if val_examples and grpo_step % args.eval_interval == 0:
            print(f"  Running eval on {args.n_eval_examples} val examples ...")
            policy.eval()
            with torch.no_grad():
                metrics = run_eval(
                    policy, vllm_model, tokenizer,
                    val_examples, prompt_template, eval_params,
                    train_device, args.n_eval_examples,
                    reward_fn=reward_fn,
                )
            policy.train()
            eval_step += 1
            print(
                f"  [eval] step={eval_step} "
                f"accuracy={metrics['accuracy']:.3f} "
                f"format={metrics['format_rate']:.3f} "
                f"reward={metrics['avg_reward']:.3f} "
                f"entropy={metrics['avg_token_entropy']:.3f}"
            )
            metrics_file.write(json.dumps({
                "grpo_step": grpo_step,
                "train_step": train_step,
                "eval_step": eval_step,
                "timestamp": time.time(),
                "accuracy": metrics["accuracy"],
                "format_rate": metrics["format_rate"],
                "avg_reward": metrics["avg_reward"],
                "avg_token_entropy": metrics["avg_token_entropy"],
                "avg_response_length": metrics["avg_response_length"],
                "avg_grad_norm": float(sum(step_grad_norms) / len(step_grad_norms)) if step_grad_norms else 0.0,
                "avg_clip_frac": float(sum(step_clip_fracs) / len(step_clip_fracs)) if step_clip_fracs else 0.0,
            }) + "\n")
            metrics_file.flush()
            if not args.no_wandb:
                wandb.log({
                    "eval/accuracy": metrics["accuracy"],
                    "eval/format_rate": metrics["format_rate"],
                    "eval/avg_reward": metrics["avg_reward"],
                    "eval/avg_token_entropy": metrics["avg_token_entropy"],
                    "eval/avg_response_length": metrics["avg_response_length"],
                    "eval_step": eval_step,
                })

    # -----------------------------------------------------------------------
    # Final eval and save
    # -----------------------------------------------------------------------
    if val_examples and not args.skip_eval:
        print("\nRunning final evaluation ...")
        policy.eval()
        with torch.no_grad():
            final = run_eval(
                policy, vllm_model, tokenizer,
                val_examples, prompt_template, eval_params,
                train_device, n_eval=min(2048, len(val_examples)),
                reward_fn=reward_fn,
            )
        print(f"Final accuracy: {final['accuracy']:.4f}")
        (output_path / "final_eval.json").write_text(
            json.dumps({k: v for k, v in final.items() if k != "examples"}, indent=2)
        )

    import os
    user = os.environ.get("USER", "user")
    save_name = f"grpo_{args.run_name}"
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
    parser = argparse.ArgumentParser(description="GRPO training on MATH with verified rewards")
    # Paths
    parser.add_argument("--model", default="/data/a5-alignment/models/Qwen2.5-Math-1.5B")
    parser.add_argument("--data", default="/data/a5-alignment/MATH/train.jsonl")
    parser.add_argument("--val_data", default="/data/a5-alignment/MATH/validation.jsonl")
    parser.add_argument("--output", default="results/section7")
    parser.add_argument("--max_train_examples", type=int, default=None)
    # GRPO hyperparameters
    parser.add_argument("--n_grpo_steps", type=int, default=200)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--rollout_batch_size", type=int, default=256)
    parser.add_argument("--epochs_per_rollout_batch", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--advantage_eps", type=float, default=1e-6)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False,
                        help="Enable gradient checkpointing to reduce activation memory (~single-GPU runs)")
    parser.add_argument("--use_std_normalization", action="store_true", default=True)
    parser.add_argument("--no_std_normalization", dest="use_std_normalization", action="store_false")
    parser.add_argument(
        "--loss_type",
        default="reinforce_with_baseline",
        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
    )
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument(
        "--length_norm",
        default="masked_mean",
        choices=["masked_mean", "masked_normalize"],
        help="Per-example loss aggregation: masked_mean or masked_normalize by max_response_tokens",
    )
    parser.add_argument(
        "--prompt_type",
        default="r1_zero",
        choices=["r1_zero", "question_only"],
        help="Prompt template and reward function to use",
    )
    # Generation
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_response_tokens", type=int, default=1024)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    # Evaluation
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--n_eval_examples", type=int, default=1024)
    parser.add_argument("--skip_eval", action="store_true")
    # Devices
    parser.add_argument("--train_device", default="cuda:0")
    parser.add_argument("--vllm_device", default="cuda:1")
    # Logging
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", default="cs336-alignment-grpo")
    parser.add_argument("--run_name", default="grpo_reinforce")
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
