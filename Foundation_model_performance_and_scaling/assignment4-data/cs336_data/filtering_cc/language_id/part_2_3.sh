#!/bin/bash
# ==============================================================================
# Section 2.3 – Language Identification (Problem: language_identification)
# ==============================================================================
#
# USAGE:
#   cd cs336_data/filtering_cc/language_id
#   ./part_2_3.sh
#
# WHAT IT DOES:
#   1. Downloads lid.176.bin via get_assets.sh (skips if already present).
#   2. Runs both language identification tests.
#
# OUTPUT:
#   Pass/fail from pytest.
#
# NOTES:
#   - lid.176.bin (~126 MB) is downloaded to cs336_data/assets/ on first run.
#   - On the Together cluster, get_assets.sh symlinks from /data/classifiers/.
#
# ==============================================================================

set -e
cd "$(dirname "$0")"

# ── Step 1: Acquire lid.176.bin ───────────────────────────────────────────────
echo "=== Step 1: Acquire lid.176.bin ==="
cd ../../..
bash get_assets.sh
cd cs336_data/filtering_cc/language_id
echo ""

# ── Step 2: Run tests ─────────────────────────────────────────────────────────
echo "=== Step 2: Run language identification tests ==="
cd ../../..
.venv/bin/pytest -k test_identify_language -v
cd cs336_data/filtering_cc/language_id

echo ""
echo "=== Done ==="