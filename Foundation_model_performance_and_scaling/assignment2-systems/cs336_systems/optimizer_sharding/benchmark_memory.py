"""
Memory profiling script for optimizer state sharding (Part a).

This script measures peak memory usage at three critical points:
1. After model initialization
2. Before optimizer step (after backward pass)
3. After optimizer step

Compares memory usage with and without optimizer state sharding.
"""

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Import model configs from cs336_basics
import sys
sys.path.append(str(Path(__file__).resolve().parents[2] / "assignment1-basics"))
from cs336_basics.transformer_training.model import TRANSFORMER_CONFIG_PRESETS, TransformerLM


def get_memory_stats():
    """Get current GPU memory statistics in GB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'max_allocated_gb': max_allocated
        }
    return None


def reset_peak_memory():
    """Reset peak memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def estimate_component_sizes(model, optimizer=None, has_gradients=False, is_sharded=False):
    """Estimate memory breakdown by component."""
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Master weights (FP32): 4 bytes per parameter
    weights_gb = (num_params * 4) / 1e9

    # Gradients (FP32): 4 bytes per parameter
    gradients_gb = (num_params * 4) / 1e9 if has_gradients else 0

    # Optimizer states (AdamW: 2 states per parameter, FP32)
    if optimizer is not None:
        if is_sharded:
            # Sharded: only ~1/world_size of optimizer states
            optimizer_states_gb = (num_params * 4 * 2) / 1e9 / world_size
        else:
            # Full: all optimizer states
            optimizer_states_gb = (num_params * 4 * 2) / 1e9
    else:
        optimizer_states_gb = 0

    return {
        'num_params': num_params,
        'weights_gb': weights_gb,
        'gradients_gb': gradients_gb,
        'optimizer_states_gb': optimizer_states_gb,
        'total_estimated_gb': weights_gb + gradients_gb + optimizer_states_gb
    }


def profile_memory(rank, world_size, args):
    """Run memory profiling on a single rank."""
    # Setup distributed
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.master_port

    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    # Get model config
    config = TRANSFORMER_CONFIG_PRESETS[args.model_size]

    # Results storage
    results = {}

    # ============================================================
    # Profile WITHOUT sharding
    # ============================================================
    if rank == 0:
        print("\n" + "="*80)
        print(f"Profiling WITHOUT optimizer state sharding (Standard DDP)")
        print("="*80)

    torch.manual_seed(42)
    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        attn_pdrop=config.attn_pdrop,
        residual_pdrop=config.residual_pdrop,
    ).to(device)

    # Point 1: After model initialization
    reset_peak_memory()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    mem_after_init = get_memory_stats()

    # Create optimizer (non-sharded)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Component breakdown after init
    components_after_init = estimate_component_sizes(model, optimizer, has_gradients=False, is_sharded=False)

    # Run forward + backward to generate gradients
    dummy_input = torch.randint(0, config.vocab_size, (args.batch_size, config.context_length)).to(device)
    dummy_target = torch.randint(0, config.vocab_size, (args.batch_size, config.context_length)).to(device)

    model.train()
    optimizer.zero_grad()

    logits = model(dummy_input)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, config.vocab_size),
        dummy_target.view(-1)
    )
    loss.backward()

    # Point 2: Before optimizer step (after backward)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    mem_before_step = get_memory_stats()
    components_before_step = estimate_component_sizes(model, optimizer, has_gradients=True, is_sharded=False)

    # Take optimizer step
    optimizer.step()

    # Point 3: After optimizer step
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    mem_after_step = get_memory_stats()
    components_after_step = estimate_component_sizes(model, optimizer, has_gradients=True, is_sharded=False)

    results['non_sharded'] = {
        'after_init': mem_after_init,
        'before_step': mem_before_step,
        'after_step': mem_after_step,
        'components_after_init': components_after_init,
        'components_before_step': components_before_step,
        'components_after_step': components_after_step,
    }

    # Clean up
    del model, optimizer, logits, loss
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ============================================================
    # Profile WITH sharding
    # ============================================================
    if rank == 0:
        print("\n" + "="*80)
        print(f"Profiling WITH optimizer state sharding")
        print("="*80)

    torch.manual_seed(42)
    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        attn_pdrop=config.attn_pdrop,
        residual_pdrop=config.residual_pdrop,
    ).to(device)

    # Point 1: After model initialization
    reset_peak_memory()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    mem_after_init_sharded = get_memory_stats()

    # Create sharded optimizer
    from cs336_systems.optimizer_sharding.optimizer import ShardedOptimizer
    optimizer_sharded = ShardedOptimizer(
        model.parameters(),
        torch.optim.AdamW,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Component breakdown after init (sharded)
    components_after_init_sharded = estimate_component_sizes(model, optimizer_sharded, has_gradients=False, is_sharded=True)

    # Run forward + backward
    model.train()
    optimizer_sharded.zero_grad()

    logits = model(dummy_input)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, config.vocab_size),
        dummy_target.view(-1)
    )
    loss.backward()

    # Point 2: Before optimizer step
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    mem_before_step_sharded = get_memory_stats()
    components_before_step_sharded = estimate_component_sizes(model, optimizer_sharded, has_gradients=True, is_sharded=True)

    # Take optimizer step
    optimizer_sharded.step()

    # Point 3: After optimizer step
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    mem_after_step_sharded = get_memory_stats()
    components_after_step_sharded = estimate_component_sizes(model, optimizer_sharded, has_gradients=True, is_sharded=True)

    results['sharded'] = {
        'after_init': mem_after_init_sharded,
        'before_step': mem_before_step_sharded,
        'after_step': mem_after_step_sharded,
        'components_after_init': components_after_init_sharded,
        'components_before_step': components_before_step_sharded,
        'components_after_step': components_after_step_sharded,
    }

    # Print results from rank 0
    if rank == 0:
        print_results(results, args)

    dist.destroy_process_group()


def print_results(results, args):
    """Print formatted results."""
    print("\n" + "="*80)
    print("MEMORY PROFILING RESULTS")
    print("="*80)
    print(f"Configuration: {args.model_size} model, world_size={args.world_size}, batch_size={args.batch_size}")
    print()

    # Extract data
    non_sharded = results['non_sharded']
    sharded = results['sharded']

    # Print component breakdown
    print("-" * 80)
    print("Component Breakdown (Estimated):")
    print("-" * 80)

    def print_components(label, components):
        print(f"\n{label}:")
        print(f"  Parameters:       {components['weights_gb']:.2f} GB")
        print(f"  Gradients:        {components['gradients_gb']:.2f} GB")
        print(f"  Optimizer states: {components['optimizer_states_gb']:.2f} GB")
        print(f"  Total (estimated): {components['total_estimated_gb']:.2f} GB")

    print_components("Without Sharding - After Init", non_sharded['components_after_init'])
    print_components("With Sharding - After Init", sharded['components_after_init'])

    savings_after_init = non_sharded['components_after_init']['optimizer_states_gb'] - sharded['components_after_init']['optimizer_states_gb']
    print(f"\n  Memory saved (optimizer states): {savings_after_init:.2f} GB")

    # Print measured memory at 3 points
    print("\n" + "-" * 80)
    print("Measured Peak Memory (GPU):")
    print("-" * 80)

    def print_point(point_name, non_sharded_mem, sharded_mem):
        print(f"\n{point_name}:")
        print(f"  Without sharding: {non_sharded_mem['max_allocated_gb']:.2f} GB")
        print(f"  With sharding:    {sharded_mem['max_allocated_gb']:.2f} GB")
        savings = non_sharded_mem['max_allocated_gb'] - sharded_mem['max_allocated_gb']
        savings_pct = (savings / non_sharded_mem['max_allocated_gb']) * 100
        print(f"  Savings:          {savings:.2f} GB ({savings_pct:.1f}%)")

    print_point("1. After Model Initialization", non_sharded['after_init'], sharded['after_init'])
    print_point("2. Before Optimizer Step", non_sharded['before_step'], sharded['before_step'])
    print_point("3. After Optimizer Step", non_sharded['after_step'], sharded['after_step'])

    # Save results to file
    output_dir = Path(__file__).resolve().parents[2] / "results" / "optimizer_sharding"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "memory_profile.txt"
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MEMORY PROFILING RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Configuration: {args.model_size} model, world_size={args.world_size}, batch_size={args.batch_size}\n")
        f.write("\n")
        f.write("-" * 80 + "\n")
        f.write("Measured Peak Memory (GPU):\n")
        f.write("-" * 80 + "\n")

        for point_name, non_sharded_mem, sharded_mem in [
            ("1. After Model Initialization", non_sharded['after_init'], sharded['after_init']),
            ("2. Before Optimizer Step", non_sharded['before_step'], sharded['before_step']),
            ("3. After Optimizer Step", non_sharded['after_step'], sharded['after_step'])
        ]:
            savings = non_sharded_mem['max_allocated_gb'] - sharded_mem['max_allocated_gb']
            savings_pct = (savings / non_sharded_mem['max_allocated_gb']) * 100
            f.write(f"\n{point_name}:\n")
            f.write(f"  Without sharding: {non_sharded_mem['max_allocated_gb']:.2f} GB\n")
            f.write(f"  With sharding:    {sharded_mem['max_allocated_gb']:.2f} GB\n")
            f.write(f"  Savings:          {savings:.2f} GB ({savings_pct:.1f}%)\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("Component Breakdown (Estimated):\n")
        f.write("-" * 80 + "\n")

        for label, components in [
            ("Without Sharding - After Optimizer Step", non_sharded['components_after_step']),
            ("With Sharding - After Optimizer Step", sharded['components_after_step'])
        ]:
            f.write(f"\n{label}:\n")
            f.write(f"  Parameters:       {components['weights_gb']:.2f} GB\n")
            f.write(f"  Gradients:        {components['gradients_gb']:.2f} GB\n")
            f.write(f"  Optimizer states: {components['optimizer_states_gb']:.2f} GB\n")
            f.write(f"  Total (estimated): {components['total_estimated_gb']:.2f} GB\n")

    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Profile memory usage with/without optimizer sharding")
    parser.add_argument('--model-size', type=str, default='xl', choices=['large', 'xl'],
                       help='Model size')
    parser.add_argument('--world-size', type=int, default=2,
                       help='Number of GPUs')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--master-port', type=str, default='29500',
                       help='Master port for distributed training')

    args = parser.parse_args()

    # Run multi-process
    mp.spawn(
        profile_memory,
        args=(args.world_size, args),
        nprocs=args.world_size,
        join=True
    )


if __name__ == '__main__':
    main()