#!/bin/bash

# Benchmark script for optimizer state sharding
# Runs both memory profiling (Part a) and speed benchmarking (Part b)

set -e  # Exit on error

# Default parameters
MODEL_SIZE=${MODEL_SIZE:-xl}
WORLD_SIZE=${WORLD_SIZE:-2}
BATCH_SIZE=${BATCH_SIZE:-2}
NUM_STEPS=${NUM_STEPS:-10}
WARMUP_STEPS=${WARMUP_STEPS:-5}

echo "================================================================================"
echo "Optimizer State Sharding Benchmarks"
echo "================================================================================"
echo "Configuration:"
echo "  Model size: $MODEL_SIZE"
echo "  World size: $WORLD_SIZE GPUs"
echo "  Batch size: $BATCH_SIZE"
echo "  Benchmark steps: $NUM_STEPS"
echo "  Warmup steps: $WARMUP_STEPS"
echo ""

# Part (a): Memory profiling
echo "================================================================================"
echo "Part (a): Memory Profiling"
echo "================================================================================"
echo ""
echo "Profiling peak memory usage at 3 points:"
echo "  1. After model initialization"
echo "  2. Before optimizer step (after backward)"
echo "  3. After optimizer step"
echo ""

uv run python benchmark_memory.py \
    --model-size "$MODEL_SIZE" \
    --world-size "$WORLD_SIZE" \
    --batch-size "$BATCH_SIZE"

echo ""
echo "Memory profiling complete!"
echo ""

# Part (b): Speed benchmarking
echo "================================================================================"
echo "Part (b): Training Speed Impact"
echo "================================================================================"
echo ""
echo "Measuring time per iteration with/without optimizer sharding..."
echo ""

uv run python benchmark_speed.py \
    --model-size "$MODEL_SIZE" \
    --world-size "$WORLD_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --num-steps "$NUM_STEPS" \
    --warmup-steps "$WARMUP_STEPS"

echo ""
echo "Speed benchmarking complete!"
echo ""

# Summary
echo "================================================================================"
echo "Benchmarking Complete!"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  Memory:  ../../results/optimizer_sharding/memory_profile.txt"
echo "  Speed:   ../../results/optimizer_sharding/speed_comparison.csv"
echo ""
echo "Next step: Review ACCOUNTING_COMMENTARY.md for analysis of all three parts."
echo ""