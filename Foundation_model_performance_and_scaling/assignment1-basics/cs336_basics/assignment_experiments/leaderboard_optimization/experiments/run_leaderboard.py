"""
Final 1.5-Hour Leaderboard Run

Runs optimized model on OpenWebText for exactly 1.5 hours.
This is the final submission for the CS336 leaderboard.

Target: Beat 5.0 validation loss baseline
Expected: ~3.3-3.5 validation loss

Usage:
    uv run python experiments/run_leaderboard.py \
        --config configs/optimized_1.5hr.json \
        --time-limit 5400 \
        --output-dir cs336_basics/basics/runs/leaderboard_final
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def run_leaderboard_experiment(config_path: Path, output_dir: Path, time_limit: int):
    """
    Run full leaderboard experiment with proper evaluation and checkpointing.

    Args:
        config_path: Path to configuration file
        output_dir: Output directory for results
        time_limit: Maximum time in seconds (default: 5400 = 1.5 hours)
    """
    import torch
    import numpy as np
    from cs336_basics.assignment_experiments.leaderboard_optimization.modifications.weight_tied_transformer import WeightTiedTransformerLM

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    print("="*80)
    print("CS336 LEADERBOARD RUN - 1.5 HOUR SUBMISSION")
    print("="*80)
    print(f"\nOptimizations:")
    print(f"  ✓ Weight tying (36% parameter reduction)")
    print(f"  ✓ Larger batch size ({config['batch_size']})")
    print(f"  ✓ Scaled learning rate ({config['learning_rate']})")
    print(f"  ✓ Mixed precision (bfloat16)")
    print(f"\nTarget: Beat 5.0 validation loss")
    print(f"Expected: 3.3-3.5 validation loss")
    print(f"\nConfiguration:")
    print(f"  - Model: Weight-tied Transformer")
    print(f"  - Vocabulary: {config['vocab_size']:,}")
    print(f"  - Parameters: ~28.8M (vs 45.2M baseline)")
    print(f"  - Batch size: {config['batch_size']}")
    print(f"  - Learning rate: {config['learning_rate']}")
    print(f"  - Warmup iterations: {config['warmup_iters']}")
    print(f"  - Time limit: {time_limit} seconds ({time_limit/3600:.1f} hours)")
    print(f"  - Output: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_save_path = output_dir / "config.json"
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=2)

    # Create model
    print("\nCreating weight-tied model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if config.get("dtype") == "bfloat16" else torch.float32

    model = WeightTiedTransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config.get("rope_theta", 10000.0),
        tie_weights=config.get("use_weight_tying", True),
        embedding_scale=config.get("embedding_scale", True),
        device=device,
        dtype=dtype,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {total_params:,} parameters")

    # Load data
    print("\nLoading OpenWebText data...")
    train_data = np.load(config["train_data_path"], mmap_mode="r")
    val_data = np.load(config["val_data_path"], mmap_mode="r")

    print(f"✓ Train data: {len(train_data):,} tokens")
    print(f"✓ Val data: {len(val_data):,} tokens")

    # Setup optimizer
    print("\nSetting up optimizer and scheduler...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(config.get("beta1", 0.9), config.get("beta2", 0.999)),
        weight_decay=config.get("weight_decay", 0.1),
    )

    # Learning rate scheduler (cosine with warmup)
    def get_lr(iteration, warmup_iters, max_lr, min_lr, max_iters):
        """Get learning rate with warmup and cosine decay."""
        if iteration < warmup_iters:
            return max_lr * iteration / warmup_iters
        if iteration > max_iters:
            return min_lr
        decay_ratio = (iteration - warmup_iters) / (max_iters - warmup_iters)
        coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    # Training loop
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")

    model.train()
    start_time = time.time()
    iteration = 0
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    batch_size = config["batch_size"]
    context_length = config["context_length"]
    eval_interval = config.get("eval_interval", 400)
    checkpoint_interval = config.get("checkpoint_interval", 4000)

    # Metrics for plotting
    metrics = {
        "iteration": [],
        "wallclock_time": [],
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
    }

    while True:
        # Check time limit
        elapsed = time.time() - start_time
        if elapsed >= time_limit:
            print(f"\n✓ Time limit reached ({elapsed/3600:.2f} hours)")
            break

        # Get learning rate
        lr = get_lr(
            iteration,
            config["warmup_iters"],
            config["learning_rate"],
            config.get("min_learning_rate", config["learning_rate"] * 0.1),
            60000,  # Max iterations (won't reach in 1.5 hours)
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Training step
        indices = torch.randint(0, len(train_data) - context_length - 1, (batch_size,))
        batch_data = torch.stack([
            torch.from_numpy(train_data[i:i+context_length+1].astype(np.int64))
            for i in indices
        ])

        input_ids = batch_data[:, :-1].to(device)
        targets = batch_data[:, 1:].to(device)

        # Forward pass
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype == torch.bfloat16)):
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, config["vocab_size"]),
                targets.view(-1),
            )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if config.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

        optimizer.step()

        # Track loss
        train_losses.append(loss.item())

        # Evaluation
        if iteration % eval_interval == 0 or elapsed >= time_limit - 10:
            model.eval()
            with torch.no_grad():
                # Evaluate on validation set
                val_loss_sum = 0.0
                val_batches = 20

                for _ in range(val_batches):
                    val_indices = torch.randint(0, len(val_data) - context_length - 1, (batch_size,))
                    val_batch = torch.stack([
                        torch.from_numpy(val_data[i:i+context_length+1].astype(np.int64))
                        for i in val_indices
                    ])

                    val_input = val_batch[:, :-1].to(device)
                    val_target = val_batch[:, 1:].to(device)

                    with torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype == torch.bfloat16)):
                        val_logits = model(val_input)
                        val_loss = torch.nn.functional.cross_entropy(
                            val_logits.view(-1, config["vocab_size"]),
                            val_target.view(-1),
                        )
                    val_loss_sum += val_loss.item()

                avg_val_loss = val_loss_sum / val_batches
                val_losses.append(avg_val_loss)

                # Track best
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    # Save best checkpoint
                    checkpoint_path = output_dir / "best_model.pt"
                    torch.save({
                        "iteration": iteration,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": avg_val_loss,
                    }, checkpoint_path)

                # Log
                avg_train_loss = sum(train_losses[-eval_interval:]) / len(train_losses[-eval_interval:])
                print(f"[{elapsed/3600:.2f}h] Iter {iteration:5d}: "
                      f"train_loss = {avg_train_loss:.4f}, "
                      f"val_loss = {avg_val_loss:.4f}, "
                      f"lr = {lr:.2e}")

                # Track metrics
                metrics["iteration"].append(iteration)
                metrics["wallclock_time"].append(elapsed / 3600)
                metrics["train_loss"].append(avg_train_loss)
                metrics["val_loss"].append(avg_val_loss)
                metrics["learning_rate"].append(lr)

            model.train()

        # Periodic checkpoints
        if iteration % checkpoint_interval == 0 and iteration > 0:
            checkpoint_path = output_dir / f"checkpoint_{iteration}.pt"
            torch.save({
                "iteration": iteration,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, checkpoint_path)

        iteration += 1

    # Final results
    print("\n" + "="*80)
    print("LEADERBOARD RUN COMPLETED!")
    print("="*80)

    elapsed_hours = (time.time() - start_time) / 3600

    print(f"\nFinal Results:")
    print(f"  Total iterations: {iteration:,}")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Final validation loss: {val_losses[-1]:.4f}")
    print(f"  Training time: {elapsed_hours:.2f} hours")
    print(f"  Tokens processed: {iteration * batch_size * context_length:,}")

    # Perplexity
    perplexity = np.exp(best_val_loss)
    print(f"  Best perplexity: {perplexity:.2f}")

    # Compare to baseline
    baseline_loss = 5.0
    improvement = baseline_loss - best_val_loss
    print(f"\n  Baseline (target): {baseline_loss:.4f}")
    print(f"  Improvement: {improvement:.4f} ({improvement/baseline_loss*100:.1f}%)")

    if best_val_loss < baseline_loss:
        print(f"\n  ✓ SUCCESS! Beat baseline by {improvement:.4f}")
    else:
        print(f"\n  ✗ Did not beat baseline (missed by {-improvement:.4f})")

    # Save final results
    results = {
        "experiment_name": config.get("experiment_name", "leaderboard_optimized"),
        "description": config.get("description", "Weight tying + fast schedule + larger batch"),
        "configuration": config,
        "results": {
            "total_iterations": iteration,
            "best_val_loss": best_val_loss,
            "final_val_loss": val_losses[-1],
            "training_hours": elapsed_hours,
            "tokens_processed": iteration * batch_size * context_length,
            "perplexity": perplexity,
            "baseline_loss": baseline_loss,
            "improvement": improvement,
            "beat_baseline": best_val_loss < baseline_loss,
        },
        "metrics": metrics,
    }

    results_path = output_dir / "final_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {results_path}")

    # Save metrics CSV for plotting
    metrics_csv_path = output_dir / "metrics.csv"
    with open(metrics_csv_path, "w") as f:
        f.write("iteration,wallclock_hours,train_loss,val_loss,learning_rate\n")
        for i in range(len(metrics["iteration"])):
            f.write(f"{metrics['iteration'][i]},"
                   f"{metrics['wallclock_time'][i]:.4f},"
                   f"{metrics['train_loss'][i]:.4f},"
                   f"{metrics['val_loss'][i]:.4f},"
                   f"{metrics['learning_rate'][i]:.6e}\n")

    print(f"✓ Metrics saved to: {metrics_csv_path}")
    print("\nNext steps:")
    print("  1. Create submission package with create_submission.py")
    print("  2. Submit to GitHub leaderboard")


def main():
    parser = argparse.ArgumentParser(description="Final 1.5-hour leaderboard run")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cs336_basics/basics/runs/leaderboard_final"),
        help="Output directory",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=5400,
        help="Time limit in seconds (default: 5400 = 1.5 hours)",
    )
    args = parser.parse_args()

    try:
        run_leaderboard_experiment(args.config, args.output_dir, args.time_limit)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()
