#!/usr/bin/env python3
"""
Quick test to verify distributed setup is working correctly.

This is a simple script based on the PyTorch distributed tutorial to ensure
that the distributed communication setup works before running full benchmarks.
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size):
    """Initialize the distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def distributed_demo(rank, world_size):
    """Simple distributed demo - generate random tensors and all-reduce them."""
    setup(rank, world_size)

    # Generate random data
    data = torch.randint(0, 10, (3,))
    print(f"rank {rank} data (before all-reduce): {data}")

    # All-reduce (sum)
    dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
    print(f"rank {rank} data (after all-reduce): {data}")

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 4

    print("Testing distributed setup with 4 processes using Gloo backend...")
    print("Expected: Each rank shows different data before all-reduce,")
    print("          identical sums after all-reduce")
    print("-" * 80)

    mp.set_start_method("spawn", force=True)
    mp.spawn(fn=distributed_demo, args=(world_size,), nprocs=world_size, join=True)

    print("-" * 80)
    print("✓ Distributed setup test complete!")
    print("\nIf you see identical values after all-reduce, the setup is working correctly.")