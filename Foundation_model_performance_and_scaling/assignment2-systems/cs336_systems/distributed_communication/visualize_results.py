#!/usr/bin/env python3
# ==============================================================================
# Visualize All-Reduce Benchmark Results
# ==============================================================================
#
# DESCRIPTION:
#   Creates plots and tables from all-reduce benchmark results to analyze
#   the impact of backend, data size, and number of processes on performance.
#   Supports both separate (CPU + GPU) files and combined files.
#
# USAGE:
#   # Combine CPU and GPU results, then visualize
#   uv run python visualize_results.py \
#       --cpu-results ../../results/distributed_communication/gloo_cpu_benchmark.csv \
#       --gpu-results ../../results/distributed_communication/nccl_gpu_benchmark.csv \
#       --output-dir ../../results/distributed_communication/
#
#   # Visualize single file (CPU only or GPU only)
#   uv run python visualize_results.py \
#       --input ../../results/distributed_communication/gloo_cpu_benchmark.csv
#
#   # Visualize pre-combined file
#   uv run python visualize_results.py \
#       --input ../../results/distributed_communication/combined_benchmark.csv
#
# OUTPUT:
#   - combined_benchmark.csv: Merged CPU + GPU results (if using --cpu-results/--gpu-results)
#   - Plots: PNG files showing performance comparisons
#   - Tables: Markdown tables for writeup
#   - Summary: Text analysis of results
#
# ==============================================================================

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_and_clean_data(input_path: str) -> pd.DataFrame:
    """Load benchmark results and filter out error cases.

    Args:
        input_path: Path to CSV file with benchmark results

    Returns:
        DataFrame with valid benchmark results
    """
    df = pd.read_csv(input_path)

    # Filter out error cases
    df_valid = df[df["avg_time_s"].notna()].copy()

    # Add derived metrics
    df_valid["avg_time_ms"] = df_valid["avg_time_s"] * 1000
    df_valid["std_time_ms"] = df_valid["std_time_s"] * 1000

    # Create readable labels
    df_valid["backend_device"] = df_valid["backend"] + "+" + df_valid["device"]

    return df_valid


def create_performance_plots(df: pd.DataFrame, output_dir: Path):
    """Create plots showing performance across different configurations.

    Args:
        df: DataFrame with benchmark results
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)

    # Plot 1: Time vs Data Size (for each backend/device and world size)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("All-Reduce Performance: Impact of Data Size and Process Count", fontsize=14, fontweight="bold")

    for idx, backend_device in enumerate(df["backend_device"].unique()):
        ax = axes[idx // 2, idx % 2]
        df_subset = df[df["backend_device"] == backend_device]

        for world_size in sorted(df_subset["world_size"].unique()):
            df_ws = df_subset[df_subset["world_size"] == world_size]
            ax.plot(
                df_ws["data_size_mb"],
                df_ws["avg_time_ms"],
                marker="o",
                label=f"{world_size} processes",
                linewidth=2,
            )

        ax.set_xlabel("Data Size (MB)", fontsize=11)
        ax.set_ylabel("Average Time (ms)", fontsize=11)
        ax.set_title(f"{backend_device}", fontsize=12, fontweight="bold")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "allreduce_time_vs_datasize.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 2: Bandwidth vs Data Size
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("All-Reduce Bandwidth: Impact of Data Size and Process Count", fontsize=14, fontweight="bold")

    for idx, backend_device in enumerate(df["backend_device"].unique()):
        ax = axes[idx // 2, idx % 2]
        df_subset = df[df["backend_device"] == backend_device]

        for world_size in sorted(df_subset["world_size"].unique()):
            df_ws = df_subset[df_subset["world_size"] == world_size]
            ax.plot(
                df_ws["data_size_mb"],
                df_ws["bandwidth_gb_s"],
                marker="o",
                label=f"{world_size} processes",
                linewidth=2,
            )

        ax.set_xlabel("Data Size (MB)", fontsize=11)
        ax.set_ylabel("Bandwidth (GB/s)", fontsize=11)
        ax.set_title(f"{backend_device}", fontsize=12, fontweight="bold")
        ax.set_xscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "allreduce_bandwidth_vs_datasize.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 3: Comparison across backends for fixed data size
    fixed_data_size = 100  # MB
    df_fixed = df[df["data_size_mb"] == fixed_data_size]

    if not df_fixed.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Backend Comparison (Data Size: {fixed_data_size}MB)", fontsize=14, fontweight="bold")

        # Time comparison
        for backend_device in df_fixed["backend_device"].unique():
            df_subset = df_fixed[df_fixed["backend_device"] == backend_device]
            ax1.plot(
                df_subset["world_size"],
                df_subset["avg_time_ms"],
                marker="o",
                label=backend_device,
                linewidth=2,
            )

        ax1.set_xlabel("Number of Processes", fontsize=11)
        ax1.set_ylabel("Average Time (ms)", fontsize=11)
        ax1.set_title("Latency vs Process Count", fontsize=12, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bandwidth comparison
        for backend_device in df_fixed["backend_device"].unique():
            df_subset = df_fixed[df_fixed["backend_device"] == backend_device]
            ax2.plot(
                df_subset["world_size"],
                df_subset["bandwidth_gb_s"],
                marker="o",
                label=backend_device,
                linewidth=2,
            )

        ax2.set_xlabel("Number of Processes", fontsize=11)
        ax2.set_ylabel("Bandwidth (GB/s)", fontsize=11)
        ax2.set_title("Bandwidth vs Process Count", fontsize=12, fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f"allreduce_backend_comparison_{fixed_data_size}mb.png", dpi=300, bbox_inches="tight")
        plt.close()

    print(f"✓ Plots saved to {output_dir}/")


def create_summary_tables(df: pd.DataFrame, output_dir: Path):
    """Create summary tables in markdown format.

    Args:
        df: DataFrame with benchmark results
        output_dir: Directory to save tables
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Table 1: Full results
    table1 = df[["backend", "device", "world_size", "data_size_mb", "avg_time_ms", "std_time_ms", "bandwidth_gb_s"]].copy()
    table1["avg_time_ms"] = table1["avg_time_ms"].round(3)
    table1["std_time_ms"] = table1["std_time_ms"].round(3)
    table1["bandwidth_gb_s"] = table1["bandwidth_gb_s"].round(2)

    with open(output_dir / "full_results.md", "w") as f:
        f.write("# All-Reduce Benchmark Results\n\n")
        f.write(table1.to_markdown(index=False))
        f.write("\n")

    # Table 2: Summary by configuration
    summary = df.groupby(["backend", "device", "world_size", "data_size_mb"]).agg({
        "avg_time_ms": ["mean", "std"],
        "bandwidth_gb_s": ["mean", "std"],
    }).round(3)

    with open(output_dir / "summary_by_config.md", "w") as f:
        f.write("# Summary by Configuration\n\n")
        f.write(summary.to_markdown())
        f.write("\n")

    print(f"✓ Tables saved to {output_dir}/")


def generate_analysis(df: pd.DataFrame, output_dir: Path):
    """Generate text analysis of benchmark results.

    Args:
        df: DataFrame with benchmark results
        output_dir: Directory to save analysis
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis = []
    analysis.append("=" * 80)
    analysis.append("All-Reduce Benchmark Analysis")
    analysis.append("=" * 80)
    analysis.append("")

    # Overall statistics
    analysis.append("## Overall Statistics")
    analysis.append("")
    analysis.append(f"Total configurations tested: {len(df)}")
    analysis.append(f"Backends: {', '.join(df['backend'].unique())}")
    analysis.append(f"Devices: {', '.join(df['device'].unique())}")
    analysis.append(f"Process counts: {', '.join(map(str, sorted(df['world_size'].unique())))}")
    analysis.append(f"Data sizes: {', '.join(map(str, sorted(df['data_size_mb'].unique())))} MB")
    analysis.append("")

    # Backend comparison
    analysis.append("## Backend Comparison")
    analysis.append("")
    for backend_device in df["backend_device"].unique():
        df_subset = df[df["backend_device"] == backend_device]
        avg_time = df_subset["avg_time_ms"].mean()
        avg_bandwidth = df_subset["bandwidth_gb_s"].mean()
        analysis.append(f"- {backend_device}:")
        analysis.append(f"  - Average time: {avg_time:.3f} ms")
        analysis.append(f"  - Average bandwidth: {avg_bandwidth:.2f} GB/s")
    analysis.append("")

    # Impact of data size
    analysis.append("## Impact of Data Size")
    analysis.append("")
    for backend_device in df["backend_device"].unique():
        df_subset = df[df["backend_device"] == backend_device]
        grouped = df_subset.groupby("data_size_mb")["avg_time_ms"].mean()
        analysis.append(f"- {backend_device}:")
        for size, time in grouped.items():
            analysis.append(f"  - {size} MB: {time:.3f} ms")
    analysis.append("")

    # Impact of process count
    analysis.append("## Impact of Process Count")
    analysis.append("")
    for backend_device in df["backend_device"].unique():
        df_subset = df[df["backend_device"] == backend_device]
        grouped = df_subset.groupby("world_size")["avg_time_ms"].mean()
        analysis.append(f"- {backend_device}:")
        for ws, time in grouped.items():
            analysis.append(f"  - {ws} processes: {time:.3f} ms")
    analysis.append("")

    # Key observations
    analysis.append("## Key Observations")
    analysis.append("")

    # Find fastest configuration
    fastest = df.loc[df["avg_time_ms"].idxmin()]
    analysis.append(f"1. Fastest configuration: {fastest['backend']}+{fastest['device']} "
                   f"with {fastest['world_size']} processes and {fastest['data_size_mb']} MB data "
                   f"({fastest['avg_time_ms']:.3f} ms)")

    # Find highest bandwidth
    highest_bw = df.loc[df["bandwidth_gb_s"].idxmax()]
    analysis.append(f"2. Highest bandwidth: {highest_bw['backend']}+{highest_bw['device']} "
                   f"with {highest_bw['world_size']} processes and {highest_bw['data_size_mb']} MB data "
                   f"({highest_bw['bandwidth_gb_s']:.2f} GB/s)")

    # Scaling with process count
    for backend_device in df["backend_device"].unique():
        df_subset = df[df["backend_device"] == backend_device]
        if len(df_subset) > 0:
            # Look at largest data size
            max_data_size = df_subset["data_size_mb"].max()
            df_max_size = df_subset[df_subset["data_size_mb"] == max_data_size]
            if len(df_max_size) > 1:
                times = df_max_size.groupby("world_size")["avg_time_ms"].mean()
                if len(times) > 1:
                    scaling = times.iloc[-1] / times.iloc[0]
                    analysis.append(f"3. {backend_device} scaling ({max_data_size} MB data): "
                                   f"{scaling:.2f}x time increase from {times.index[0]} to {times.index[-1]} processes")

    analysis.append("")
    analysis.append("=" * 80)

    # Save analysis
    with open(output_dir / "analysis.txt", "w") as f:
        f.write("\n".join(analysis))

    # Also print to console
    print("\n".join(analysis))


def combine_cpu_gpu_results(cpu_path: str, gpu_path: str, output_path: str) -> pd.DataFrame:
    """Combine CPU and GPU benchmark results into a single DataFrame.

    Args:
        cpu_path: Path to CPU benchmark CSV
        gpu_path: Path to GPU benchmark CSV
        output_path: Path to save combined CSV

    Returns:
        Combined DataFrame
    """
    print(f"Loading CPU results from {cpu_path}...")
    df_cpu = pd.read_csv(cpu_path)
    print(f"✓ Loaded {len(df_cpu)} CPU benchmark results")

    print(f"Loading GPU results from {gpu_path}...")
    df_gpu = pd.read_csv(gpu_path)
    print(f"✓ Loaded {len(df_gpu)} GPU benchmark results")

    # Combine DataFrames
    df_combined = pd.concat([df_cpu, df_gpu], ignore_index=True)

    # Save combined results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(output_path, index=False)
    print(f"✓ Saved combined results to {output_path}")
    print(f"  Total: {len(df_combined)} benchmark results")

    return df_combined


def main():
    parser = argparse.ArgumentParser(
        description="Visualize all-reduce benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine CPU and GPU results, then visualize
  python visualize_results.py \\
      --cpu-results ../../results/distributed_communication/gloo_cpu_benchmark.csv \\
      --gpu-results ../../results/distributed_communication/nccl_gpu_benchmark.csv

  # Visualize single file (CPU only)
  python visualize_results.py \\
      --input ../../results/distributed_communication/gloo_cpu_benchmark.csv

  # Visualize pre-combined file
  python visualize_results.py \\
      --input ../../results/distributed_communication/combined_benchmark.csv
        """,
    )

    # Option 1: Separate CPU and GPU files
    parser.add_argument(
        "--cpu-results",
        type=str,
        help="Path to CPU (Gloo) benchmark results CSV",
    )
    parser.add_argument(
        "--gpu-results",
        type=str,
        help="Path to GPU (NCCL) benchmark results CSV",
    )

    # Option 2: Single input file (combined or single backend)
    parser.add_argument(
        "--input",
        type=str,
        help="Path to single benchmark results CSV (combined or single backend)",
    )

    # Output directory
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../../results/distributed_communication/",
        help="Output directory for plots and tables (default: ../../results/distributed_communication/)",
    )

    args = parser.parse_args()

    # Determine input mode
    if args.cpu_results and args.gpu_results:
        # Mode 1: Combine CPU and GPU results
        print("=" * 80)
        print("Mode: Combining CPU and GPU benchmark results")
        print("=" * 80)
        print("")

        # Check if files exist
        if not Path(args.cpu_results).exists():
            print(f"Error: CPU results file not found: {args.cpu_results}")
            print("Please run: ./run_benchmark_cpu.sh")
            return

        if not Path(args.gpu_results).exists():
            print(f"Error: GPU results file not found: {args.gpu_results}")
            print("Please run: ./run_benchmark_gpu.sh on H100 instance")
            return

        # Combine and save
        output_dir = Path(args.output_dir)
        combined_path = output_dir / "combined_benchmark.csv"
        df_raw = combine_cpu_gpu_results(args.cpu_results, args.gpu_results, combined_path)
        print("")

        # Clean data
        df = load_and_clean_data(combined_path)

    elif args.input:
        # Mode 2: Single input file
        print("=" * 80)
        print("Mode: Visualizing single benchmark file")
        print("=" * 80)
        print("")

        if not Path(args.input).exists():
            print(f"Error: Input file not found: {args.input}")
            print("Please run benchmark script first.")
            return

        # Load data
        print(f"Loading data from {args.input}...")
        df = load_and_clean_data(args.input)
        print(f"✓ Loaded {len(df)} valid benchmark results")

    else:
        # Try default paths
        print("=" * 80)
        print("Mode: Auto-detecting benchmark files")
        print("=" * 80)
        print("")

        output_dir = Path(args.output_dir)
        cpu_default = output_dir / "gloo_cpu_benchmark.csv"
        gpu_default = output_dir / "nccl_gpu_benchmark.csv"
        combined_default = output_dir / "combined_benchmark.csv"

        # Check for combined file first
        if combined_default.exists():
            print(f"Found combined results: {combined_default}")
            df = load_and_clean_data(str(combined_default))
        # Check for separate CPU and GPU files
        elif cpu_default.exists() and gpu_default.exists():
            print(f"Found CPU results: {cpu_default}")
            print(f"Found GPU results: {gpu_default}")
            df_raw = combine_cpu_gpu_results(str(cpu_default), str(gpu_default), str(combined_default))
            df = load_and_clean_data(str(combined_default))
        # Check for CPU only
        elif cpu_default.exists():
            print(f"Found CPU results only: {cpu_default}")
            df = load_and_clean_data(str(cpu_default))
        # Check for GPU only
        elif gpu_default.exists():
            print(f"Found GPU results only: {gpu_default}")
            df = load_and_clean_data(str(gpu_default))
        else:
            print("Error: No benchmark results found in default locations")
            print(f"  Expected one of:")
            print(f"    - {combined_default}")
            print(f"    - {cpu_default} and {gpu_default}")
            print(f"    - {cpu_default}")
            print(f"    - {gpu_default}")
            print("")
            print("Please run benchmark scripts first:")
            print("  ./run_benchmark_cpu.sh")
            print("  ./run_benchmark_gpu.sh (on H100)")
            return

    print("")

    if len(df) == 0:
        print("Error: No valid results found in input file(s)")
        return

    # Create visualizations
    output_dir = Path(args.output_dir)

    print("=" * 80)
    print(f"Generating visualizations for {len(df)} benchmark results")
    print("=" * 80)
    print("")

    print("Generating plots...")
    create_performance_plots(df, output_dir)

    print("\nGenerating summary tables...")
    create_summary_tables(df, output_dir)

    print("\nGenerating analysis...")
    generate_analysis(df, output_dir)

    print(f"\n✓ All outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()