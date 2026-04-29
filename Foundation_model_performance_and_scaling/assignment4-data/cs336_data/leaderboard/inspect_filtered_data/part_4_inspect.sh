#!/bin/bash
# ==============================================================================
# Section 4 – Inspect Filtered Data (Problem: inspect_filtered_data)
# ==============================================================================
#
# USAGE:
#   cd cs336_data/leaderboard/inspect_filtered_data
#   ./part_4_inspect.sh
#
# WHAT IT DOES:
#   (a) Samples 5 random kept examples from the filtered .txt output.
#   (b) Re-processes the WET file to collect one rejected example per
#       filter stage (empty, non_english, gopher_fail, low_quality, nsfw).
#   (c) Writes a markdown report with commentary on each example and
#       observations about the pipeline.
#
# OUTPUT:
#   results/leaderboard/inspect_filtered_data_answers.md
#
# NOTES:
#   - Requires the filter pipeline to have already been run (part_4_filter.sh).
#   - Re-processing the WET file for rejected examples takes ~2-3 minutes.
#
# ==============================================================================

set -e
cd "$(dirname "$0")"
ROOT="$(cd ../../.. && pwd)"

FILTERED_TXT="${ROOT}/data/filtered/CC-MAIN-20250417135010-20250417165010-00065.txt"
WET_FILE="${ROOT}/data/CC/CC-MAIN-20250417135010-20250417165010-00065.warc.gz"
OUTPUT="${ROOT}/results/leaderboard/inspect_filtered_data_answers.md"

# Use WET file if available, otherwise fall back to WARC (WET uses .warc.wet.gz naming)
WET_GZ="${ROOT}/data/CC/CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz"
if [ -f "${WET_GZ}" ]; then
    WET_FILE="${WET_GZ}"
fi

echo "=== Part 4: Inspect Filtered Data ==="
echo "  Filtered txt: ${FILTERED_TXT}"
echo "  WET file:     ${WET_FILE}"
echo "  Output:       ${OUTPUT}"
echo ""

cd "${ROOT}"
.venv/bin/python -m cs336_data.leaderboard.inspect_filtered_data.inspect \
    --filtered-txt "${FILTERED_TXT}" \
    --wet-file     "${WET_FILE}" \
    --output       "${OUTPUT}" \
    --n-kept 5 --seed 42 \
    --lang-threshold    0.65 \
    --quality-threshold 0.3  \
    --nsfw-threshold    0.8

echo ""
echo "=== Done. Report at ${OUTPUT} ==="
