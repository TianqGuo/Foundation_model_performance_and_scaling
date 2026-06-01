"""DPO fine-tuning on Anthropic HH preference data.

Run via part_6_5.sh or directly:
    uv run python cs336_alignment/section5_dpo/train_dpo.py [args]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.section5_dpo.dataset import load_hh_dataset, split_train_val
from cs336_alignment.section5_dpo.dpo import _sequence_log_prob, per_instance_dpo_loss


def _fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _compute_reward_accuracy(
    policy: torch.nn.Module,
    examples: list[dict],
    tokenizer,
    policy_device: str,
    alpaca_template: str,
) -> float:
    """Implicit reward accuracy: fraction where log π(yw) > log π(yl)."""
    eos_id = tokenizer.eos_token_id
    n_correct = 0
    policy.eval()
    with torch.no_grad():
        for ex in examples:
            chosen_text = alpaca_template.format(
                instruction=ex["instruction"], response=ex["chosen"]
            )
            rejected_text = alpaca_template.format(
                instruction=ex["instruction"], response=ex["rejected"]
            )
            chosen_ids_list = tokenizer.encode(chosen_text, add_special_tokens=True) + [eos_id]
            rejected_ids_list = tokenizer.encode(rejected_text, add_special_tokens=True) + [eos_id]

            chosen_ids = torch.tensor(chosen_ids_list, dtype=torch.long, device=policy_device)
            rejected_ids = torch.tensor(rejected_ids_list, dtype=torch.long, device=policy_device)

            lp_chosen = _sequence_log_prob(policy, chosen_ids)
            lp_rejected = _sequence_log_prob(policy, rejected_ids)

            if lp_chosen.item() > lp_rejected.item():
                n_correct += 1
    policy.train()
    return n_correct / len(examples)


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA not available. DPO training requires 2 GPUs.")
    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        raise SystemExit(f"ERROR: DPO requires 2 GPUs, found {n_gpus}.")

    policy_device = args.policy_device
    ref_device = args.ref_device
    print(f"Policy model on {policy_device}, reference model on {ref_device}")

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- W&B ---
    if not args.no_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
        wandb.define_metric("train_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("val/*", step_metric="train_step")

    # --- Tokenizer ---
    print(f"Loading tokenizer from {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Models: policy (π_θ) on GPU 0, reference (π_ref) on GPU 1 ---
    print(f"Loading policy model on {policy_device} ...")
    policy = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(policy_device)
    policy.train()

    print(f"Loading reference model on {ref_device} ...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(ref_device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # --- Alpaca template ---
    alpaca_template = (
        Path(__file__).parent.parent / "prompts" / "alpaca_sft.prompt"
    ).read_text().rstrip("\n")

    # --- Dataset ---
    print(f"Loading HH dataset from {args.data_dir} ...")
    all_examples = load_hh_dataset(args.data_dir)
    print(f"  {len(all_examples)} single-turn examples across all 4 files")

    train_examples, val_examples = split_train_val(all_examples, n_val=args.n_val, seed=args.seed)
    print(f"  Train: {len(train_examples)}  Val: {len(val_examples)}")

    # --- Optimizer (RMSprop — AdamW too memory-intensive for 2× 8B) ---
    optimizer = torch.optim.RMSprop(policy.parameters(), lr=args.lr)
    optimizer.zero_grad()

    # --- Metrics file ---
    metrics_path = output_path / f"train_metrics_{args.run_name}.jsonl"
    metrics_file = open(metrics_path, "w")

    best_val_acc = -1.0  # ensures first validation always saves a checkpoint
    best_ckpt_dir = Path(args.checkpoint_dir) / "best"
    best_ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_step = 0
    microbatch_count = 0
    t_start = time.time()
    t_last_step = time.time()

    total_steps = (len(train_examples) // args.gradient_accumulation_steps) * args.n_epochs
    print(
        f"\nStarting DPO: {args.n_epochs} epoch(s) | β={args.beta} | lr={args.lr} | "
        f"grad_accum={args.gradient_accumulation_steps} | total_steps~{total_steps}\n"
    )

    for epoch in range(args.n_epochs):
        random.shuffle(train_examples)

        for ex in train_examples:
            if args.max_steps is not None and train_step >= args.max_steps:
                break

            try:
                loss = per_instance_dpo_loss(
                    lm=policy,
                    lm_ref=ref_model,
                    tokenizer=tokenizer,
                    beta=args.beta,
                    prompt=ex["instruction"],
                    response_chosen=ex["chosen"],
                    response_rejected=ex["rejected"],
                )
                (loss / args.gradient_accumulation_steps).backward()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("WARNING: OOM — skipping example")
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

                now = time.time()
                step_time = now - t_last_step
                elapsed = now - t_start
                eta = step_time * ((args.max_steps or total_steps) - train_step)
                t_last_step = now

                loss_val = loss.item()
                print(
                    f"  epoch={epoch+1} step={train_step}/{args.max_steps or total_steps} "
                    f"loss={loss_val:.4f} | elapsed={_fmt_time(elapsed)} "
                    f"step={step_time:.1f}s eta={_fmt_time(eta)}"
                )

                row: dict = {
                    "train_step": train_step,
                    "epoch": epoch + 1,
                    "train_loss": loss_val,
                    "elapsed_sec": round(elapsed, 1),
                    "step_time_sec": round(step_time, 1),
                    "eta_sec": round(eta, 1),
                }

                if not args.no_wandb:
                    import wandb
                    wandb.log({"train/loss": loss_val, "train_step": train_step})

                # Periodic validation
                if train_step % args.val_interval == 0:
                    val_acc = _compute_reward_accuracy(
                        policy, val_examples, tokenizer, policy_device, alpaca_template
                    )
                    print(f"  [val] step={train_step} reward_acc={val_acc:.3f} (best={best_val_acc:.3f})")
                    row["val_reward_acc"] = val_acc

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        policy.save_pretrained(best_ckpt_dir)
                        tokenizer.save_pretrained(best_ckpt_dir)
                        print(f"  [val] New best checkpoint saved to {best_ckpt_dir}")

                    if not args.no_wandb:
                        import wandb
                        wandb.log({"val/reward_acc": val_acc, "train_step": train_step})

                metrics_file.write(json.dumps(row) + "\n")
                metrics_file.flush()

        if args.max_steps is not None and train_step >= args.max_steps:
            break

    # --- Final validation ---
    total_elapsed = time.time() - t_start
    print(f"\nTraining complete: {train_step} steps in {_fmt_time(total_elapsed)}")

    val_acc = _compute_reward_accuracy(
        policy, val_examples, tokenizer, policy_device, alpaca_template
    )
    print(f"Final reward accuracy: {val_acc:.4f}  Best: {best_val_acc:.4f}")

    (output_path / f"final_val_{args.run_name}.json").write_text(json.dumps({
        "val_reward_acc": val_acc,
        "best_val_reward_acc": max(0.0, best_val_acc),
        "train_steps": train_step,
        "total_elapsed_sec": round(total_elapsed, 1),
        "total_elapsed_human": _fmt_time(total_elapsed),
    }, indent=2))

    # Save final checkpoint
    final_ckpt_dir = Path(args.checkpoint_dir) / "final"
    final_ckpt_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(final_ckpt_dir)
    tokenizer.save_pretrained(final_ckpt_dir)
    print(f"Final checkpoint saved to {final_ckpt_dir}")
    print(f"Best checkpoint at {best_ckpt_dir}")

    metrics_file.close()
    print(f"Metrics saved to {metrics_path}")

    if not args.no_wandb:
        import wandb
        wandb.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO fine-tuning on Anthropic HH")
    # Paths
    parser.add_argument("--model", default="assets/sft_ultrachat",
                        help="SFT checkpoint to fine-tune (used for both policy and reference)")
    parser.add_argument("--data-dir", default="data/hh",
                        help="Directory containing HH .jsonl.gz files")
    parser.add_argument("--output", default="results/section5")
    parser.add_argument("--checkpoint-dir", default="assets/dpo_hh",
                        help="Directory to save best and final checkpoints")
    # Training
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=64)
    parser.add_argument("--n-val", type=int, default=200,
                        help="Number of validation examples held out")
    parser.add_argument("--val-interval", type=int, default=50,
                        help="Validate every N optimizer steps")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Stop after N optimizer steps (smoke test)")
    # Devices
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--ref-device", default="cuda:1")
    # Logging
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="dpo_hh")
    parser.add_argument("--wandb-project", default="cs336-part6-dpo")
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())