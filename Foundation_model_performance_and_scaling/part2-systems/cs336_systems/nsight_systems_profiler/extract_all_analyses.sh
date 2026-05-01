#!/bin/bash
# ==============================================================================
# Extract analysis summaries from all nsight profiles
# ==============================================================================
#
# This script extracts timing data from .sqlite files and creates text summaries
# instead of committing large binary .nsys-rep/.sqlite files to git.
#
# USAGE:
#   cd cs336_systems/nsight_systems_profiler
#   ./extract_all_analyses.sh
#
# OUTPUT:
#   - Text summaries in ../../results/nsight_profiles/ANALYSIS_SUMMARY.txt
#   - Individual analyses for each profile
#
# ==============================================================================

set -e

echo "=========================================="
echo "Extracting Nsight Profile Analyses"
echo "=========================================="
echo ""

RESULTS_DIR="../../results/nsight_profiles"
OUTPUT_FILE="${RESULTS_DIR}/ANALYSIS_SUMMARY.txt"

# Create header
cat > "$OUTPUT_FILE" << 'EOF'
================================================================================
Nsight Systems Profiling Analysis Summary
================================================================================
Generated from H100 profiling run on Lightning AI

This file contains extracted timing data from nsight profiles.
Full .nsys-rep files are NOT committed to git (too large, contain binary data).

To regenerate profiles:
  cd cs336_systems/nsight_systems_profiler
  ./profile_part_a.sh    # or other part scripts

================================================================================

EOF

echo "Extracting analyses from all .sqlite files..."
echo ""

# Find all sqlite files
sqlite_files=$(find "$RESULTS_DIR" -name "*.sqlite" | sort)

if [ -z "$sqlite_files" ]; then
    echo "No .sqlite files found in $RESULTS_DIR"
    exit 1
fi

# Process each sqlite file
for sqlite_file in $sqlite_files; do
    echo "Processing: $(basename $sqlite_file)"

    # Get relative path for better readability
    rel_path=$(realpath --relative-to="$RESULTS_DIR" "$sqlite_file")

    # Add section header to summary
    echo "" >> "$OUTPUT_FILE"
    echo "========================================" >> "$OUTPUT_FILE"
    echo "Profile: $rel_path" >> "$OUTPUT_FILE"
    echo "========================================" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"

    # Run analysis and append to summary
    python analyze_wsl_profiles.py "$sqlite_file" >> "$OUTPUT_FILE" 2>&1 || {
        echo "Warning: Analysis failed for $sqlite_file" >> "$OUTPUT_FILE"
    }

    echo "" >> "$OUTPUT_FILE"
    echo "----------------------------------------" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
done

echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo ""
echo "Summary saved to: $OUTPUT_FILE"
echo ""
echo "File sizes:"
du -h "$OUTPUT_FILE"
echo ""
echo "This text summary can be safely committed to git."
echo "Do NOT commit .nsys-rep or .sqlite files (too large)."
echo ""
