#!/bin/bash
# ==============================================================================
# Section 2.1 – Looking at the Data (Problem: look_at_cc)
# ==============================================================================
#
# USAGE:
#   cd cs336_data/filtering_cc/look_at_data
#   ./part_2_1.sh
#
# WHAT IT DOES:
#   1. Downloads the sample WARC and WET files from Common Crawl (if not present).
#      On the Together cluster, soft-links /data/CC/ instead of downloading.
#   2. Runs explore_cc.py to extract observations for parts (a), (b), (d).
#   3. Saves output to results/filtering_cc/look_at_cc_observations.txt.
#
# OUTPUT:
#   ../../../results/filtering_cc/look_at_cc_observations.txt
#
# NOTES:
#   - Files are ~500 MB each. Download only runs once; subsequent runs skip.
#   - Part (c) is a written analysis — no script output needed.
#   - Run on the H100 instance for faster download. Local RTX 4090 works too.
#
# ==============================================================================

set -e
cd "$(dirname "$0")"

CLUSTER_CC_DIR="/data/CC"
DATA_DIR="../../../data/CC"
RESULTS_DIR="../../../results/filtering_cc"

WARC_FILENAME="CC-MAIN-20250417135010-20250417165010-00065.warc.gz"
WET_FILENAME="CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz"
WARC_URL="https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-18/segments/1744889135610.12/warc/${WARC_FILENAME}"
WET_URL="https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-18/segments/1744889135610.12/wet/${WET_FILENAME}"

WARC_PATH="${DATA_DIR}/${WARC_FILENAME}"
WET_PATH="${DATA_DIR}/${WET_FILENAME}"

mkdir -p "${DATA_DIR}"
mkdir -p "${RESULTS_DIR}"

# ── Download or link files ────────────────────────────────────────────────────
handle_file() {
    local filename=$1
    local url=$2
    local dest="${DATA_DIR}/${filename}"
    local cluster_src="${CLUSTER_CC_DIR}/${filename}"

    if [ -e "${dest}" ]; then
        echo "✓ ${filename} already exists, skipping."
    elif [ -f "${cluster_src}" ]; then
        echo "→ Found ${filename} on cluster, creating symlink..."
        ln -s "${cluster_src}" "${dest}"
    else
        echo "↓ Downloading ${filename} (~500 MB)..."
        wget --show-progress -q "${url}" -O "${dest}"
        echo "✓ Download complete."
    fi
}

echo "=== Step 1: Acquire CC sample files ==="
handle_file "${WARC_FILENAME}" "${WARC_URL}"
handle_file "${WET_FILENAME}" "${WET_URL}"

# ── Run analysis ──────────────────────────────────────────────────────────────
echo ""
echo "=== Step 2: Run analysis (parts a, b, d) ==="
../../../.venv/bin/python explore_cc.py \
    --warc "${WARC_PATH}" \
    --wet  "${WET_PATH}" \
    --output "${RESULTS_DIR}/look_at_cc_observations.txt"

echo ""
echo "=== Done ==="
echo "Observations: ${RESULTS_DIR}/look_at_cc_observations.txt"
echo ""
echo "To browse files interactively:"
echo "  zcat ${WARC_PATH} | less   # part (a): raw HTML"
echo "  zcat ${WET_PATH}  | less   # part (b): extracted text"
