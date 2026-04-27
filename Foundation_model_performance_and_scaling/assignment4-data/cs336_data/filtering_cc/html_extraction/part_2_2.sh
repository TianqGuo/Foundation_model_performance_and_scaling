#!/bin/bash
# ==============================================================================
# Section 2.2 – HTML to Text Conversion (Problem: extract_text, part b)
# ==============================================================================
#
# USAGE:
#   cd cs336_data/filtering_cc/html_extraction
#   ./part_2_2.sh
#
# WHAT IT DOES:
#   1. Verifies the extract_text test passes.
#   2. Compares our extraction output against CC's WET file for 20 records.
#
# OUTPUT:
#   ../../../results/filtering_cc/wet_comparison.txt
#
# NOTES:
#   - Requires WARC/WET files in ../../../data/CC/
#     (run look_at_data/part_2_1.sh first to download them).
#   - Part (b) written answer: fill in results/filtering_cc/extract_text_answers.md
#     after reviewing the output.
#
# ==============================================================================

set -e
cd "$(dirname "$0")"

DATA_DIR="../../../data/CC"
RESULTS_DIR="../../../results/filtering_cc"

WARC_FILENAME="CC-MAIN-20250417135010-20250417165010-00065.warc.gz"
WET_FILENAME="CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz"
WARC_PATH="${DATA_DIR}/${WARC_FILENAME}"
WET_PATH="${DATA_DIR}/${WET_FILENAME}"

# ── Step 1: Run test ──────────────────────────────────────────────────────────
echo "=== Step 1: Run extract_text test ==="
cd ../../..
.venv/bin/pytest -k test_extract_text_from_html_bytes -v
cd cs336_data/filtering_cc/html_extraction
echo ""

# ── Step 2: Check data files exist ───────────────────────────────────────────
echo "=== Step 2: Check CC sample files ==="
if [ ! -e "${WARC_PATH}" ] || [ ! -e "${WET_PATH}" ]; then
    echo "ERROR: WARC/WET files not found in ${DATA_DIR}."
    echo "       Run look_at_data/part_2_1.sh first to download them."
    exit 1
fi
echo "✓ Files found."
echo ""

# ── Step 3: Compare extraction vs WET ────────────────────────────────────────
echo "=== Step 3: Compare our extraction vs WET file (20 records) ==="
../../../.venv/bin/python compare_wet.py \
    --warc "${WARC_PATH}" \
    --wet  "${WET_PATH}" \
    --n 20 \
    --output "${RESULTS_DIR}/wet_comparison.txt"

echo ""
echo "=== Done ==="
echo "Review: ${RESULTS_DIR}/wet_comparison.txt"
echo "Then fill in written answer: ${RESULTS_DIR}/extract_text_answers.md"