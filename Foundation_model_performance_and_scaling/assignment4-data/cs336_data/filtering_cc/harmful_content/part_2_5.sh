#!/bin/bash
# ==============================================================================
# Section 2.5 – Harmful Content Classification (Problem: harmful_content)
# ==============================================================================
#
# USAGE:
#   cd cs336_data/filtering_cc/harmful_content
#   ./part_2_5.sh
#
# WHAT IT DOES:
#   1. Downloads Dolma NSFW and hate-speech models via get_assets.sh
#      (skips if already present).
#   2. Runs both harmful content classifier tests.
#
# ==============================================================================

set -e
cd "$(dirname "$0")"

echo "=== Step 1: Acquire Dolma classifier models ==="
cd ../../..
bash get_assets.sh
cd cs336_data/filtering_cc/harmful_content
echo ""

echo "=== Step 2: Run harmful content tests ==="
cd ../../..
.venv/bin/pytest -k "test_classify_nsfw or test_classify_toxic" -v

echo ""
echo "=== Done ==="