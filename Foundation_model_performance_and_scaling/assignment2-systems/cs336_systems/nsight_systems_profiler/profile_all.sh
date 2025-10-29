#!/bin/bash
# ==============================================================================
# Nsight Systems Profiling - Run All Model Sizes and Context Lengths
# ==============================================================================
#
# This script profiles all model sizes (small, medium, large, xl, 2.7B) with
# different context lengths (128, 256, 512, 1024) as required by the assignment.
#
# USAGE:
#   cd cs336_systems/nsight_systems_profiler
#   ./profile_all.sh
#
# REQUIREMENTS:
#   - NVIDIA Nsight Systems installed (nsys command available)
#   - GPU with sufficient memory (A100 recommended for xl/2.7B)
#
# OUTPUT:
#   - Results saved to: ../../results/nsight_profiles/
#   - Each profile: <model>_ctx<context>_<type>.nsys-rep
#
# NOTES:
#   - Larger models may OOM with larger context lengths
#   - Each profile takes 1-5 minutes depending on model size
#   - Total runtime: 1-3 hours for all combinations
#
# ==============================================================================

set -e

echo "=========================================="
echo "Nsight Systems Profiling - All Models"
echo "=========================================="
echo ""

# Create output directory
OUTPUT_DIR="../../results/nsight_profiles"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Model sizes to profile
MODELS=("small" "medium" "large" "xl" "2.7B")

# Context lengths to test
CONTEXTS=(128 256 512 1024)

# Profile types
TYPES=("forward" "forward_backward" "training")

# Function to profile a single configuration
profile_config() {
    local model=$1
    local ctx=$2
    local profile_type=$3
    local output_file="${OUTPUT_DIR}/${model}_ctx${ctx}_${profile_type}.nsys-rep"

    echo "Profiling: $model, ctx=$ctx, type=$profile_type"

    # Check if file already exists
    if [ -f "$output_file" ]; then
        echo "  ⚠ Profile already exists, skipping"
        return 0
    fi

    # Run nsys profile
    if uv run nsys profile \
        -o "$output_file" \
        --force-overwrite true \
        --trace=cuda,nvtx --stats=true \
        --python-backtrace=cuda \
        python -m cs336_systems.nsight_systems_profiler.profile_model \
        --model-size "$model" \
        --context-length "$ctx" \
        --batch-size 4 \
        --warmup-steps 5 \
        --measure-steps 10 \
        --profile-type "$profile_type" \
        2>&1 | tee "${output_file%.nsys-rep}.log"; then
        echo "  ✓ Complete: $output_file"
        return 0
    else
        echo "  ✗ Failed (likely OOM)"
        return 1
    fi
}

# Track statistics
total=0
success=0
failed=0

echo "Starting profiling sweep..."
echo "This will take 1-3 hours depending on your GPU"
echo ""

# Profile each combination
for model in "${MODELS[@]}"; do
    for ctx in "${CONTEXTS[@]}"; do
        for ptype in "${TYPES[@]}"; do
            total=$((total + 1))

            if profile_config "$model" "$ctx" "$ptype"; then
                success=$((success + 1))
            else
                failed=$((failed + 1))
            fi

            echo ""
        done
    done
done

echo "=========================================="
echo "Profiling Complete!"
echo "=========================================="
echo "Total runs: $total"
echo "Successful: $success"
echo "Failed (OOM): $failed"
echo ""
echo "Results: $OUTPUT_DIR"
echo ""
echo "To view profiles:"
echo "  1. Download .nsys-rep files to your local machine"
echo "  2. Open with Nsight Systems GUI"
echo "  3. Filter by NVTX ranges to exclude warmup"
echo "  4. Check 'CUDA GPU Kernel Summary' in Stats System View"
echo ""
