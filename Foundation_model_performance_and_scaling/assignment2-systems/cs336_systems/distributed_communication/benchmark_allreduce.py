#!/usr/bin/env python3
# ==============================================================================
# All-Reduce Benchmarking Script
# ==============================================================================
#
# DESCRIPTION:
#   Benchmarks the all-reduce collective operation in PyTorch distributed
#   communication. Tests various backends (Gloo+CPU, NCCL+GPU), data sizes
#   (1MB to 1GB), and number of processes (2, 4, 6).
#
# USAGE:
#   # Run with default settings (all combinations)
#   uv run python benchmark_allreduce.py
#
#   # Run specific configuration
#   uv run python benchmark_allreduce.py --backend nccl --device cuda \
#       --data-sizes 1 10 100 --num-processes 2 4
#
#   # Save to custom location
#   uv run python benchmark_allreduce.py --output results/custom.csv
#
# OUTPUT:
#   CSV file with benchmark results (default: results/distributed_communication/allreduce_benchmark.csv)
#
# REQUIREMENTS:
#   - For NCCL+GPU: Up to 6 GPUs
#   - Each benchmarking run takes <5 minutes
#
# ==============================================================================

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank: int, world_size: int, backend: str, master_port: str = "29500"):
    """Initialize the distributed process group.

    Args:
        rank: Global rank of this process
        world_size: Total number of processes
        backend: Communication backend ('gloo' or 'nccl')
        master_port: Port for master process
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    # For multi-GPU, set device for this rank
    if backend == "nccl":
        torch.cuda.set_device(rank)


def cleanup():
    """Clean up the distributed process group."""
    dist.destroy_process_group()


def benchmark_allreduce_worker(
    rank: int,
    world_size: int,
    backend: str,
    device: str,
    data_size_mb: float,
    num_warmup: int,
    num_iters: int,
    results_queue: mp.Queue,
    master_port: str,
):
    """Worker process for benchmarking all-reduce operation.

    Args:
        rank: Global rank of this process
        world_size: Total number of processes
        backend: Communication backend ('gloo' or 'nccl')
        device: Device type ('cpu' or 'cuda')
        data_size_mb: Size of data tensor in megabytes
        num_warmup: Number of warm-up iterations
        num_iters: Number of measured iterations
        results_queue: Queue to send results back to main process
        master_port: Port for master process
    """
    try:
        # Setup process group
        setup(rank, world_size, backend, master_port)

        # Calculate tensor size (float32 = 4 bytes)
        num_elements = int(data_size_mb * 1024 * 1024 / 4)

        # Create random data tensor on appropriate device
        if device == "cuda":
            data = torch.randn(num_elements, device=f"cuda:{rank}", dtype=torch.float32)
        else:
            data = torch.randn(num_elements, device="cpu", dtype=torch.float32)

        # Warm-up iterations
        for _ in range(num_warmup):
            dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
            if device == "cuda":
                torch.cuda.synchronize()

        # Measured iterations
        timings = []
        for _ in range(num_iters):
            if device == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()
            dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)

            if device == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            timings.append(end_time - start_time)

        # Calculate statistics for this rank
        avg_time = sum(timings) / len(timings)
        std_time = (sum((t - avg_time) ** 2 for t in timings) / len(timings)) ** 0.5

        # Gather all timings to rank 0
        all_avg_times = [None] * world_size
        all_std_times = [None] * world_size
        dist.all_gather_object(all_avg_times, avg_time)
        dist.all_gather_object(all_std_times, std_time)

        # Only rank 0 reports results
        if rank == 0:
            # Aggregate across all ranks
            global_avg_time = sum(all_avg_times) / len(all_avg_times)
            global_std_time = sum(all_std_times) / len(all_std_times)

            results_queue.put({
                "backend": backend,
                "device": device,
                "world_size": world_size,
                "data_size_mb": data_size_mb,
                "avg_time_s": global_avg_time,
                "std_time_s": global_std_time,
                "bandwidth_gb_s": (data_size_mb / 1024) / global_avg_time if global_avg_time > 0 else 0,
            })

        cleanup()

    except Exception as e:
        if rank == 0:
            print(f"Error in worker (rank {rank}): {e}", file=sys.stderr)
            results_queue.put({
                "backend": backend,
                "device": device,
                "world_size": world_size,
                "data_size_mb": data_size_mb,
                "avg_time_s": None,
                "std_time_s": None,
                "bandwidth_gb_s": None,
                "error": str(e),
            })


def run_benchmark(
    backend: str,
    device: str,
    world_size: int,
    data_size_mb: float,
    num_warmup: int = 5,
    num_iters: int = 100,
    master_port: str = "29500",
) -> Dict[str, Any]:
    """Run a single benchmark configuration.

    Args:
        backend: Communication backend ('gloo' or 'nccl')
        device: Device type ('cpu' or 'cuda')
        world_size: Number of processes to spawn
        data_size_mb: Size of data tensor in megabytes
        num_warmup: Number of warm-up iterations (default: 5)
        num_iters: Number of measured iterations (default: 100)
        master_port: Port for master process

    Returns:
        Dictionary with benchmark results
    """
    # Check GPU availability for NCCL
    if backend == "nccl" and not torch.cuda.is_available():
        return {
            "backend": backend,
            "device": device,
            "world_size": world_size,
            "data_size_mb": data_size_mb,
            "avg_time_s": None,
            "std_time_s": None,
            "bandwidth_gb_s": None,
            "error": "CUDA not available",
        }

    if backend == "nccl" and torch.cuda.device_count() < world_size:
        return {
            "backend": backend,
            "device": device,
            "world_size": world_size,
            "data_size_mb": data_size_mb,
            "avg_time_s": None,
            "std_time_s": None,
            "bandwidth_gb_s": None,
            "error": f"Not enough GPUs (need {world_size}, have {torch.cuda.device_count()})",
        }

    # Create queue for results
    ctx = mp.get_context("spawn")
    results_queue = ctx.Queue()

    # Spawn worker processes
    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=benchmark_allreduce_worker,
            args=(
                rank,
                world_size,
                backend,
                device,
                data_size_mb,
                num_warmup,
                num_iters,
                results_queue,
                master_port,
            ),
        )
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Get results from queue
    if not results_queue.empty():
        return results_queue.get()
    else:
        return {
            "backend": backend,
            "device": device,
            "world_size": world_size,
            "data_size_mb": data_size_mb,
            "avg_time_s": None,
            "std_time_s": None,
            "bandwidth_gb_s": None,
            "error": "No results received",
        }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark all-reduce collective operation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Backend and device options
    parser.add_argument(
        "--backend",
        type=str,
        nargs="+",
        default=["gloo", "nccl"],
        choices=["gloo", "nccl"],
        help="Communication backend(s) to test (default: both gloo and nccl)",
    )
    parser.add_argument(
        "--device",
        type=str,
        nargs="+",
        default=["cpu", "cuda"],
        choices=["cpu", "cuda"],
        help="Device(s) to test (default: both cpu and cuda)",
    )

    # Data size options
    parser.add_argument(
        "--data-sizes",
        type=float,
        nargs="+",
        default=[1, 10, 100, 1000],
        help="Data sizes in MB to test (default: 1 10 100 1000)",
    )

    # Process count options
    parser.add_argument(
        "--num-processes",
        type=int,
        nargs="+",
        default=[2, 4, 6],
        help="Number of processes to test (default: 2 4 6)",
    )

    # Benchmarking parameters
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=5,
        help="Number of warm-up iterations (default: 5)",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=100,
        help="Number of measured iterations (default: 100)",
    )

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default="results/distributed_communication/allreduce_benchmark.csv",
        help="Output CSV file path (default: results/distributed_communication/allreduce_benchmark.csv)",
    )

    # Port for avoiding conflicts when running multiple benchmarks
    parser.add_argument(
        "--master-port",
        type=str,
        default="29500",
        help="Master port for distributed communication (default: 29500)",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate all combinations of configurations
    results = []
    total_configs = len(args.backend) * len(args.device) * len(args.data_sizes) * len(args.num_processes)
    config_idx = 0

    print(f"Running {total_configs} benchmark configurations...")
    print(f"Output will be saved to: {args.output}")
    print("-" * 80)

    for backend in args.backend:
        for device in args.device:
            # Skip invalid combinations
            if backend == "gloo" and device == "cuda":
                print(f"⊗ Skipping: gloo+cuda (use nccl for GPU)")
                continue
            if backend == "nccl" and device == "cpu":
                print(f"⊗ Skipping: nccl+cpu (use gloo for CPU)")
                continue

            for world_size in args.num_processes:
                for data_size_mb in args.data_sizes:
                    config_idx += 1

                    print(f"[{config_idx}/{total_configs}] Testing: {backend}+{device}, "
                          f"{world_size} processes, {data_size_mb}MB data...")

                    result = run_benchmark(
                        backend=backend,
                        device=device,
                        world_size=world_size,
                        data_size_mb=data_size_mb,
                        num_warmup=args.num_warmup,
                        num_iters=args.num_iters,
                        master_port=args.master_port,
                    )

                    results.append(result)

                    # Print result summary
                    if result.get("error"):
                        print(f"  ✗ Error: {result['error']}")
                    else:
                        print(f"  ✓ Avg time: {result['avg_time_s']*1000:.3f}ms ± "
                              f"{result['std_time_s']*1000:.3f}ms, "
                              f"Bandwidth: {result['bandwidth_gb_s']:.2f} GB/s")

                    # Small delay between runs to allow cleanup
                    time.sleep(0.5)

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)

    print("-" * 80)
    print(f"✓ Results saved to {args.output}")
    print("\nSummary statistics:")
    print(df.groupby(["backend", "device", "world_size"])["avg_time_s"].describe())


if __name__ == "__main__":
    # For multiprocessing on Windows/WSL
    mp.set_start_method("spawn", force=True)
    main()