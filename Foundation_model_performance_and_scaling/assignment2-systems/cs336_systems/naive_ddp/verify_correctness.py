#!/usr/bin/env python3
# ==============================================================================
# Verify Naïve DDP Correctness
# ==============================================================================
#
# DESCRIPTION:
#   Verifies that naïve DDP training produces identical results to single-process
#   training by comparing final model weights after training on the same data.
#
# USAGE:
#   # Run with default settings
#   uv run python verify_correctness.py
#
#   # Custom configuration
#   uv run python verify_correctness.py --num-steps 50 --world-size 4
#
# OUTPUT:
#   Prints verification results and maximum weight difference
#
# ==============================================================================

import argparse
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from naive_ddp_trainer import NaiveDDPTrainer, setup_distributed, cleanup_distributed, shard_batch


class ToyModel(nn.Module):
    """Simple toy model for verification.

    A small MLP for testing DDP correctness.
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_synthetic_dataset(
    num_samples: int,
    input_dim: int,
    output_dim: int,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic dataset for testing.

    Args:
        num_samples: Number of samples
        input_dim: Input dimension
        output_dim: Output dimension (number of classes)
        seed: Random seed

    Returns:
        Tuple of (inputs, targets)
    """
    set_random_seed(seed)

    inputs = torch.randn(num_samples, input_dim)
    targets = torch.randint(0, output_dim, (num_samples,))

    return inputs, targets


def train_single_process(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data: tuple[torch.Tensor, torch.Tensor],
    num_steps: int,
    batch_size: int,
    device: str = "cuda",
) -> dict:
    """Train model using single-process (no DDP).

    Args:
        model: Model to train
        optimizer: Optimizer
        data: Tuple of (inputs, targets)
        num_steps: Number of training steps
        batch_size: Batch size
        device: Device to use

    Returns:
        Dictionary with training info
    """
    model = model.to(device)
    inputs, targets = data
    inputs = inputs.to(device)
    targets = targets.to(device)

    loss_fn = nn.CrossEntropyLoss()

    for step in range(num_steps):
        # Get batch
        start_idx = (step * batch_size) % len(inputs)
        end_idx = start_idx + batch_size

        batch_inputs = inputs[start_idx:end_idx]
        batch_targets = targets[start_idx:end_idx]

        # Training step
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = loss_fn(outputs, batch_targets)
        loss.backward()
        optimizer.step()

    return {"final_loss": loss.item()}


def train_ddp_worker(
    rank: int,
    world_size: int,
    model_state: dict,
    optimizer_config: dict,
    data: tuple[torch.Tensor, torch.Tensor],
    num_steps: int,
    batch_size: int,
    results_queue: mp.Queue,
):
    """Worker process for DDP training.

    Args:
        rank: Global rank
        world_size: Total number of processes
        model_state: Initial model state dict
        optimizer_config: Optimizer configuration
        data: Tuple of (inputs, targets)
        num_steps: Number of training steps
        batch_size: Global batch size (will be sharded)
        results_queue: Queue to send results back
    """
    try:
        # Setup distributed
        setup_distributed(rank, world_size, backend="nccl")

        device = f"cuda:{rank}"

        # Create model and load initial state
        model = ToyModel(
            input_dim=model_state["input_dim"],
            hidden_dim=model_state["hidden_dim"],
            output_dim=model_state["output_dim"],
        )
        model.load_state_dict(model_state["state_dict"])

        # Create optimizer
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=optimizer_config["lr"],
            momentum=optimizer_config["momentum"],
        )

        # Create trainer
        trainer = NaiveDDPTrainer(model, optimizer, rank, world_size, device=device)

        # Training loop
        inputs, targets = data

        for step in range(num_steps):
            # Get global batch
            start_idx = (step * batch_size) % len(inputs)
            end_idx = start_idx + batch_size

            batch_inputs = inputs[start_idx:end_idx]
            batch_targets = targets[start_idx:end_idx]

            # Shard batch across ranks
            local_inputs, local_targets = shard_batch(
                (batch_inputs, batch_targets),
                rank,
                world_size,
            )

            # Training step
            step_info = trainer.train_step(local_inputs, local_targets)

        # Only rank 0 reports results
        if rank == 0:
            # Get final model state
            # IMPORTANT: .detach().cpu().clone() creates fully independent CPU copy
            # - .detach() removes from autograd graph
            # - .cpu() moves to CPU memory
            # - .clone() creates independent copy (not just a view)
            final_state = {name: param.detach().cpu().clone() for name, param in model.named_parameters()}
            results_queue.put({
                "final_state": final_state,
                "final_loss": step_info["loss"],
            })

        cleanup_distributed()

    except Exception as e:
        if rank == 0:
            print(f"Error in DDP worker (rank {rank}): {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()


def compare_model_states(state1: dict, state2: dict, tolerance: float = 1e-6) -> tuple[bool, float]:
    """Compare two model state dicts.

    Args:
        state1: First model state
        state2: Second model state
        tolerance: Maximum allowed difference

    Returns:
        Tuple of (states_match, max_difference)
    """
    max_diff = 0.0

    for name in state1.keys():
        if name not in state2:
            return False, float('inf')

        diff = torch.abs(state1[name] - state2[name]).max().item()
        max_diff = max(max_diff, diff)

    states_match = max_diff < tolerance

    return states_match, max_diff


def main():
    parser = argparse.ArgumentParser(description="Verify naïve DDP correctness")

    parser.add_argument(
        "--num-steps",
        type=int,
        default=20,
        help="Number of training steps (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Global batch size (default: 32)",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
        help="Number of processes for DDP (default: 2)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of synthetic samples (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Naïve DDP Correctness Verification")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Training steps: {args.num_steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  World size: {args.world_size}")
    print(f"  Samples: {args.num_samples}")
    print(f"  Seed: {args.seed}")
    print()

    # Check GPU availability
    if not torch.cuda.is_available():
        print("✗ Error: CUDA not available")
        print("  This verification requires GPUs")
        return 1

    if torch.cuda.device_count() < args.world_size:
        print(f"✗ Error: Need {args.world_size} GPUs, but only {torch.cuda.device_count()} available")
        return 1

    # Create synthetic dataset
    print("Creating synthetic dataset...")
    input_dim = 128
    hidden_dim = 256
    output_dim = 10

    data = create_synthetic_dataset(args.num_samples, input_dim, output_dim, seed=args.seed)
    print(f"✓ Created dataset with {args.num_samples} samples")
    print()

    # Initialize model with same weights for both experiments
    set_random_seed(args.seed)
    reference_model = ToyModel(input_dim, hidden_dim, output_dim)
    initial_state = reference_model.state_dict()

    # Optimizer configuration
    optimizer_config = {
        "lr": 0.01,
        "momentum": 0.9,
    }

    # ========================================================================
    # Experiment 1: Single-process training
    # ========================================================================
    print("Running single-process training (reference)...")
    set_random_seed(args.seed)

    single_model = ToyModel(input_dim, hidden_dim, output_dim)
    single_model.load_state_dict(initial_state)

    single_optimizer = torch.optim.SGD(
        single_model.parameters(),
        lr=optimizer_config["lr"],
        momentum=optimizer_config["momentum"],
    )

    single_info = train_single_process(
        single_model,
        single_optimizer,
        data,
        args.num_steps,
        args.batch_size,
        device="cuda:0",
    )

    single_state = {name: param.detach().cpu().clone() for name, param in single_model.named_parameters()}
    print(f"✓ Single-process training complete")
    print(f"  Final loss: {single_info['final_loss']:.6f}")
    print()

    # ========================================================================
    # Experiment 2: DDP training
    # ========================================================================
    print(f"Running naïve DDP training ({args.world_size} processes)...")

    # Package model state for DDP workers
    model_state = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "state_dict": initial_state,
    }

    # Spawn DDP workers
    ctx = mp.get_context("spawn")
    results_queue = ctx.Queue()

    processes = []
    for rank in range(args.world_size):
        p = ctx.Process(
            target=train_ddp_worker,
            args=(
                rank,
                args.world_size,
                model_state,
                optimizer_config,
                data,
                args.num_steps,
                args.batch_size,
                results_queue,
            ),
        )
        p.start()
        processes.append(p)

    # Wait for completion
    for p in processes:
        p.join()

    # Get results
    if not results_queue.empty():
        ddp_results = results_queue.get()
        ddp_state = ddp_results["final_state"]
        ddp_loss = ddp_results["final_loss"]

        print(f"✓ Naïve DDP training complete")
        print(f"  Final loss: {ddp_loss:.6f}")
        print()
    else:
        print("✗ Error: No results from DDP training")
        return 1

    # ========================================================================
    # Compare results
    # ========================================================================
    print("Comparing model states...")
    print("-" * 80)

    states_match, max_diff = compare_model_states(single_state, ddp_state)

    print(f"Maximum weight difference: {max_diff:.2e}")

    if states_match:
        print()
        print("✓" * 40)
        print("✓ VERIFICATION PASSED")
        print("✓ Naïve DDP produces identical results to single-process training")
        print("✓" * 40)
        return 0
    else:
        print()
        print("✗" * 40)
        print("✗ VERIFICATION FAILED")
        print(f"✗ Weight difference ({max_diff:.2e}) exceeds tolerance (1e-6)")
        print("✗" * 40)
        return 1


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())