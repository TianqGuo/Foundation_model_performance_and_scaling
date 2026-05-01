#!/usr/bin/env python3
"""
Analyze Nsight profiles from WSL2 environment.

Since WSL2 doesn't capture GPU kernel data, this script extracts useful
information from the CUDA API traces and NVTX ranges that ARE captured.

Usage:
    python analyze_wsl_profiles.py <path_to_nsys_sqlite_file>

Example:
    python analyze_wsl_profiles.py ../../results/nsight_profiles/part_a/small_forward_ctx512.sqlite
"""

import argparse
import sqlite3
from pathlib import Path
from typing import Dict, List


def analyze_nvtx_ranges(db_path: Path) -> Dict:
    """Extract NVTX range timing information."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # First, get the string mapping
    cursor.execute("SELECT id, value FROM StringIds")
    string_map = {id: value for id, value in cursor.fetchall()}

    # Query NVTX ranges
    query = """
    SELECT textId, text, start, end, (end - start) as duration
    FROM NVTX_EVENTS
    WHERE eventType = 59  -- PushPop ranges
    ORDER BY duration DESC
    """

    try:
        cursor.execute(query)
        ranges = cursor.fetchall()

        results = {}
        for textId, text, start, end, duration in ranges:
            # Get text from textId if text field is empty
            range_name = text if text else string_map.get(textId, f"Unknown_{textId}")
            # Convert nanoseconds to milliseconds
            duration_ms = duration / 1_000_000
            results[range_name] = duration_ms

        return results
    except sqlite3.Error as e:
        print(f"Error querying NVTX data: {e}")
        return {}
    finally:
        conn.close()


def analyze_cuda_api(db_path: Path) -> Dict:
    """Extract CUDA API call timing information."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query CUDA API calls
    query = """
    SELECT nameId, COUNT(*) as count, SUM(end - start) as total_time
    FROM CUPTI_ACTIVITY_KIND_RUNTIME
    GROUP BY nameId
    ORDER BY total_time DESC
    """

    try:
        cursor.execute(query)
        api_calls = cursor.fetchall()

        # Get name mapping
        name_query = """
        SELECT id, value FROM StringIds
        """
        cursor.execute(name_query)
        name_map = {id: value for id, value in cursor.fetchall()}

        results = {}
        for nameId, count, total_time in api_calls:
            name = name_map.get(nameId, f"Unknown_{nameId}")
            time_ms = total_time / 1_000_000
            results[name] = {"count": count, "time_ms": time_ms}

        return results
    except sqlite3.Error as e:
        print(f"Error querying CUDA API data: {e}")
        return {}
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze Nsight profiles from WSL2")
    parser.add_argument("sqlite_file", type=Path, help="Path to .sqlite file")
    args = parser.parse_args()

    if not args.sqlite_file.exists():
        print(f"Error: File not found: {args.sqlite_file}")
        return

    print("=" * 70)
    print(f"Analyzing: {args.sqlite_file.name}")
    print("=" * 70)
    print()

    # Analyze NVTX ranges
    print("NVTX Ranges (Top 10):")
    print("-" * 70)
    nvtx_ranges = analyze_nvtx_ranges(args.sqlite_file)

    # Separate warmup and forward steps
    warmup_time = nvtx_ranges.get("warmup", 0)
    forward_steps = {k: v for k, v in nvtx_ranges.items() if k.startswith("forward_step_")}

    if warmup_time > 0:
        print(f"  Warmup: {warmup_time:,.2f} ms")

    if forward_steps:
        avg_forward = sum(forward_steps.values()) / len(forward_steps)
        print(f"  Forward steps (avg): {avg_forward:,.2f} ms")
        print(f"  Forward steps (count): {len(forward_steps)}")
        print()
        print("  Individual forward steps:")
        for step_name in sorted(forward_steps.keys()):
            print(f"    {step_name}: {forward_steps[step_name]:,.2f} ms")

    print()

    # Analyze CUDA API
    print("CUDA API Calls (Top 10):")
    print("-" * 70)
    cuda_api = analyze_cuda_api(args.sqlite_file)

    for i, (name, data) in enumerate(sorted(cuda_api.items(), key=lambda x: x[1]["time_ms"], reverse=True)[:10]):
        print(f"  {i+1}. {name}")
        print(f"     Calls: {data['count']:,}")
        print(f"     Total time: {data['time_ms']:,.2f} ms")
        if data['count'] > 0:
            print(f"     Avg time: {data['time_ms'] / data['count']:,.2f} ms")
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY FOR ASSIGNMENT QUESTIONS:")
    print("=" * 70)

    if forward_steps:
        avg_forward = sum(forward_steps.values()) / len(forward_steps)
        print(f"\nPart (a) - Forward pass time:")
        print(f"  Average forward pass: {avg_forward:,.2f} ms")
        print(f"  (Compare this with Python timing from part_b_results.csv)")

    # Check for cudaLaunchKernel
    if "cudaLaunchKernel" in cuda_api:
        kernel_time = cuda_api["cudaLaunchKernel"]["time_ms"]
        kernel_count = cuda_api["cudaLaunchKernel"]["count"]
        print(f"\n  Total cudaLaunchKernel time: {kernel_time:,.2f} ms")
        print(f"  Number of kernel launches: {kernel_count:,}")
        print(f"  Avg per kernel: {kernel_time / kernel_count:.2f} ms")

    print("\n" + "=" * 70)
    print("NOTE: WSL2 Limitation")
    print("=" * 70)
    print("""
WSL2 cannot capture GPU kernel execution data. For detailed kernel analysis
(Parts b-e), you need to:

1. Run on native Linux (e.g., your school's compute center with A100)
2. OR open the .nsys-rep file in Nsight Systems GUI on Windows
   (The GUI can sometimes access more data than the CLI)

The data above comes from CUDA API traces and NVTX ranges, which provide
timing information but not kernel-level details.
""")


if __name__ == "__main__":
    main()
