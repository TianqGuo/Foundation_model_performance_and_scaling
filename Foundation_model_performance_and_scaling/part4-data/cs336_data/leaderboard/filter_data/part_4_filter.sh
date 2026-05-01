#!/bin/bash
# ==============================================================================
# Section 4 – Filter Data (Problem: filter_data)
# ==============================================================================
#
# USAGE:
#   cd cs336_data/leaderboard/filter_data
#   ./part_4_filter.sh [--limit N]        # local test (default: all files)
#
# WHAT IT DOES:
#   Filters CC WET files in parallel to produce LM training data.
#   Each WET file is processed independently by a worker process.
#
#   Pipeline per document:
#     1. Min length      — skip if < 100 chars
#     2. Language ID     — keep English (score >= 0.65)
#     3. Gopher rules    — word count 50-100k, mean word len 3-10,
#                          ellipsis <= 30%, alpha words >= 80%
#     4. Quality clf     — keep if wiki-probability >= 0.3
#     5. NSFW filter     — discard if NSFW confidence >= 0.8
#     6. PII masking     — mask emails, phones, IPs (always applied)
#
# OUTPUT:
#   ${ROOT}/data/filtered/        One .txt per WET file (one doc per line)
#   ${ROOT}/data/filtered/filter_stats.json  Per-stage rejection counts + timing
#
# NOTES:
#   - On the Together cluster, WET files are at /data/CC/CC*.warc.wet.gz
#   - For Slurm parallelism across nodes, adapt to submitit (see Requirements.md)
#   - Use --limit 5 for a quick local smoke test before running on all 5000 files
#   - quality_classifier.bin must exist at cs336_data/assets/ (run part_2_7.sh first)
#
# ==============================================================================

set -e
cd "$(dirname "$0")"
ROOT="$(cd ../../.. && pwd)"

INPUT_DIR="${ROOT}/data/CC"
OUTPUT_DIR="${ROOT}/data/filtered"

# Allow --limit argument for quick testing
LIMIT_ARG=""
if [ "$1" = "--limit" ] && [ -n "$2" ]; then
    LIMIT_ARG="--limit $2"
    echo "Running in test mode: processing at most $2 WET file(s)"
fi

echo "=== Part 4: Filter Data ==="
echo "  Input:   ${INPUT_DIR}"
echo "  Output:  ${OUTPUT_DIR}"
echo ""

cd "${ROOT}"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python -m cs336_data.leaderboard.filter_data.filter \
    --input-dir  "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --lang-threshold    0.65 \
    --quality-threshold 0.3  \
    --nsfw-threshold    0.8  \
    ${LIMIT_ARG}

echo ""
echo "=== Done. Results in ${OUTPUT_DIR} ==="
