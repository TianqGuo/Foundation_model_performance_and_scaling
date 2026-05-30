"""Instruction fine-tuning (SFT) on safety-augmented UltraChat-200K.

Run via part_6_3.sh or directly:
    uv run python cs336_alignment/section3_sft/train_sft.py [args]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


def train(args: argparse.Namespace) -> None:
    # Lazy imports so this module is importable without GPU / heavy deps
    import wandb

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA not available. SFT training requires a GPU.")
    device = args.device
    print(f"Training on {device}")

    # --- Wandb ---
    if not args.no_wandb:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
        wandb.define_metric("train_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("val/*", step_metric="train_step")

    # --- Tokenizer and model ---
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    model.train()

    # --- Datasets ---
    from cs336_alignment.section3_sft.dataset import PackedSFTDataset, iterate_batches

    print(f"Building training dataset from {args.train_data} ...")
    train_dataset = PackedSFTDataset(
        tokenizer=tokenizer,
        dataset_path=args.train_data,
        seq_length=args.seq_length,
        shuffle=True,
    )
    print(f"  {len(train_dataset)} training sequences")

    val_dataset = None
    if args.val_data and Path(args.val_data).exists():
        print(f"Building validation dataset from {args.val_data} ...")
        val_dataset = PackedSFTDataset(
            tokenizer=tokenizer,
            dataset_path=args.val_data,
            seq_length=args.seq_length,
            shuffle=False,
        )
        print(f"  {len(val_dataset)} validation sequences")

    train_loader = iterate_batches(train_dataset, args.micro_batch_size, shuffle=True)

    # --- Scheduler ---
    total_steps = (len(train_dataset) // args.micro_batch_size) // args.gradient_accumulation_steps * args.n_epochs
    warmup_steps = max(1, int(0.03 * total_steps))
    print(f"Total optimizer steps: {total_steps}  Warmup steps: {warmup_steps}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    optimizer.zero_grad()

    # Per-step metrics file — written incrementally so results survive crashes
    metrics_path = output_path / f"train_metrics_{args.run_name}.jsonl"
    metrics_file = open(metrics_path, "w")

    train_step = 0
    microbatch_count = 0
    max_steps = args.max_steps  # None = no limit

    print(
        f"\nStarting SFT: {args.n_epochs} epoch(s) | seq_length={args.seq_length} | "
        f"micro_bs={args.micro_batch_size} | grad_accum={args.gradient_accumulation_steps} | "
        f"eff_batch={args.micro_batch_size * args.gradient_accumulation_steps} | lr={args.lr}\n"
    )

    for epoch in range(args.n_epochs):
        for batch in train_loader:
            if max_steps is not None and train_step >= max_steps:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            try:
                logits = model(input_ids).logits  # (bs, seq_len, vocab)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                )
                (loss / args.gradient_accumulation_steps).backward()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("WARNING: OOM — skipping microbatch")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    microbatch_count = 0
                    continue
                raise

            microbatch_count += 1

            if microbatch_count % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                train_step += 1

                lr_now = scheduler.get_last_lr()[0]
                loss_val = loss.item()
                print(
                    f"  epoch={epoch+1} step={train_step}/{total_steps} "
                    f"loss={loss_val:.4f} lr={lr_now:.2e}"
                )

                row: dict = {
                    "train_step": train_step,
                    "epoch": epoch + 1,
                    "train_loss": loss_val,
                    "lr": lr_now,
                }

                if not args.no_wandb:
                    wandb.log({"train/loss": loss_val, "train/lr": lr_now, "train_step": train_step})

                # Periodic validation loss
                if val_dataset is not None and train_step % args.val_interval == 0:
                    val_loss = _compute_val_loss(model, val_dataset, args, device)
                    print(f"  [val] step={train_step} val_loss={val_loss:.4f}")
                    row["val_loss"] = val_loss
                    if not args.no_wandb:
                        wandb.log({"val/loss": val_loss, "train_step": train_step})

                metrics_file.write(json.dumps(row) + "\n")
                metrics_file.flush()

    # Final validation loss
    if val_dataset is not None:
        val_loss = _compute_val_loss(model, val_dataset, args, device)
        print(f"\nFinal val_loss: {val_loss:.4f}")
        (output_path / f"final_val_{args.run_name}.json").write_text(
            json.dumps({"val_loss": val_loss, "train_steps": train_step}, indent=2)
        )

    # --- Save checkpoint ---
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    print(f"Checkpoint saved to {checkpoint_dir}")

    metrics_file.close()
    print(f"Training metrics saved to {metrics_path}")

    if not args.no_wandb:
        wandb.finish()


def _compute_val_loss(model, val_dataset, args, device: str, n_batches: int = 50) -> float:
    from cs336_alignment.section3_sft.dataset import iterate_batches

    val_loader = iterate_batches(val_dataset, args.micro_batch_size, shuffle=False)
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            if n >= n_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            total_loss += loss.item()
            n += 1
    model.train()
    return total_loss / max(n, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT on safety-augmented UltraChat-200K")
    # Paths
    parser.add_argument(
        "--model",
        default="/data/a5-alignment/models/Llama-3.1-8B",
    )
    parser.add_argument(
        "--train_data",
        default="/data/a5-alignment/safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz",
    )
    parser.add_argument(
        "--val_data",
        default="/data/a5-alignment/safety_augmented_ultrachat_200k_single_turn/test.jsonl.gz",
    )
    parser.add_argument("--output", default="results/section3")
    parser.add_argument(
        "--checkpoint_dir",
        default="assets/sft_ultrachat",
        help="Directory to save the fine-tuned model checkpoint",
    )
    # Sequence / batch
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--n_epochs", type=int, default=1)
    # Optimizer
    parser.add_argument("--lr", type=float, default=2e-5)
    # Validation
    parser.add_argument("--val_interval", type=int, default=100,
                        help="Log validation loss every N optimizer steps")
    # Device
    parser.add_argument("--device", default="cuda:0")
    # Logging
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", default="cs336-part6-sft")
    parser.add_argument("--run_name", default="sft_ultrachat")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Stop after this many optimizer steps (smoke-test / ablation)")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
