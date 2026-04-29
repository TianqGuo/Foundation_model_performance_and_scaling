#!/bin/bash
# ==============================================================================
# Section 4 – Download WET Files (Problem: download_wet)
# ==============================================================================
#
# USAGE:
#   cd cs336_data/leaderboard/download_wet
#   ./part_4_download.sh [--n 100]
#
# WHAT IT DOES:
#   Downloads N WET files from Common Crawl CC-MAIN-2025-18 into data/CC/.
#   Each WET file is ~200 MB compressed and yields ~11M filtered tokens.
#
# ROUGH SIZING:
#   100 files  →  ~20 GB download  →  ~1.1B tokens
#   600 files  →  ~120 GB download →  ~6.6B tokens  (enough for 100K steps)
#
# NOTES:
#   - Set N higher for leaderboard training (need ≥ 600 files for 100K steps
#     at batch_size=128, context_length=512).
#   - Uses 8 parallel HTTP connections by default (adjust with --workers).
#   - Files already present in data/CC/ are skipped automatically.
#
# ==============================================================================

set -e
cd "$(dirname "$0")"
ROOT="$(cd ../../.. && pwd)"
N="${1:-100}"   # default 100; pass e.g. "./part_4_download.sh 600" for more

echo "=== Part 4: Download WET Files ==="
echo "  Crawl:  CC-MAIN-2025-18"
echo "  Count:  ${N} files"
echo "  Output: ${ROOT}/data/CC/"
echo ""

cd "${ROOT}"
.venv/bin/python -m cs336_data.leaderboard.download_wet.download_wet \
    --n          "${N}" \
    --output-dir "${ROOT}/data/CC"

echo ""
echo "=== Done. WET files at ${ROOT}/data/CC/ ==="
