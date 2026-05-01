"""
30-Minute OpenWebText Test

Tests optimized configuration on OpenWebText for 30 minutes.
This validates the full setup before committing to the 1.5-hour final run.

Expected results:
- Training should be stable
- Loss should be decreasing smoothly
- No OOM errors or crashes
- Iteration speed should be ~0.11s/iter with bfloat16

Usage:
    uv run python experiments/test_short_run.py --config configs/optimized_1.5hr.json --time-limit 1800
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def run_timed_experiment(config_path: Path, output_dir: Path, time_limit: int):
    """
    Run experiment with time limit.

    Args:
        config_path: Path to configuration file
        output_dir: Output directory for results
        time_limit: Maximum time in seconds
    """
    import torch
    from cs336_basics.assignment_experiments.leaderboard_optimization.modifications.weight_tied_transformer import WeightTiedTransformerLM

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    print("="*80)
    print(f"30-Minute OpenWebText Test")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  - Model: Weight-tied Transformer")
    print(f"  - Vocabulary: {config['vocab_size']:,}")
    print(f"  - Batch size: {config['batch_size']}")
    print(f"  - Learning rate: {config['learning_rate']}")
    print(f"  - Data type: {config['dtype']}")
    print(f"  - Time limit: {time_limit} seconds ({time_limit/60:.1f} minutes)")
    print(f"  - Output: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

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
    import numpy as np

    train_data = np.load(config["train_data_path"], mmap_mode="r")
    val_data = np.load(config["val_data_path"], mmap_mode="r")

    print(f"✓ Train data: {len(train_data):,} tokens")
    print(f"✓ Val data: {len(val_data):,} tokens")

    # Setup optimizer
    print("\nSetting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(config.get("beta1", 0.9), config.get("beta2", 0.999)),
        weight_decay=config.get("weight_decay", 0.1),
    )

    # Training loop with time limit
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")

    model.train()
    start_time = time.time()
    iteration = 0
    losses = []

    batch_size = config["batch_size"]
    context_length = config["context_length"]

    while True:
        # Check time limit
        elapsed = time.time() - start_time
        if elapsed >= time_limit:
            print(f"\n✓ Time limit reached ({elapsed:.1f}s)")
            break

        # Training step
        # Get random batch
        indices = torch.randint(0, len(train_data) - context_length - 1, (batch_size,))
        batch_data = torch.stack([
            torch.from_numpy(train_data[i:i+context_length+1].astype(np.int64))
            for i in indices
        ])

        input_ids = batch_data[:, :-1].to(device)
        targets = batch_data[:, 1:].to(device)

        # Forward pass
        with torch.autocast(device_type="cuda", dtype=dtype):
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

        # Log
        losses.append(loss.item())
        iteration += 1

        if iteration % 50 == 0:
            avg_loss = sum(losses[-50:]) / len(losses[-50:])
            print(f"[{elapsed:.0f}s] Iteration {iteration}: loss = {avg_loss:.4f}")

    # Final results
    print("\n" + "="*80)
    print("Test completed!")
    print("="*80)

    elapsed_hours = (time.time() - start_time) / 3600
    print(f"\nResults:")
    print(f"  Iterations completed: {iteration:,}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Average loss (last 100): {sum(losses[-100:])/len(losses[-100:]):.4f}")
    print(f"  Time elapsed: {elapsed_hours:.2f} hours")
    print(f"  Iterations/second: {iteration / (time.time() - start_time):.2f}")

    # Estimate full run
    estimated_iters_1_5hr = iteration * (5400 / time_limit)
    print(f"\nEstimated iterations in 1.5 hours: {estimated_iters_1_5hr:,.0f}")

    # Save results
    results = {
        "iterations": iteration,
        "final_loss": losses[-1],
        "avg_loss_last_100": sum(losses[-100:])/len(losses[-100:]),
        "time_hours": elapsed_hours,
        "iters_per_second": iteration / (time.time() - start_time),
        "estimated_iters_1_5hr": estimated_iters_1_5hr,
    }

    results_path = output_dir / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="30-minute OpenWebText test")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("cs336_basics/assignment_experiments/leaderboard_optimization/configs/optimized_1.5hr.json"),
        help="Configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cs336_basics/basics/runs/leaderboard_test_30min"),
        help="Output directory",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=1800,
        help="Time limit in seconds (default: 1800 = 30 minutes)",
    )
    args = parser.parse_args()

    try:
        run_timed_experiment(args.config, args.output_dir, args.time_limit)
        print("\n✓ Test successful! Ready for 1.5-hour final run.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()
