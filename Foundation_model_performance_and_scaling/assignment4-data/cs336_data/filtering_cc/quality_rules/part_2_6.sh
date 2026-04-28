#!/bin/bash
# ==============================================================================
# Section 2.6 – Gopher Quality Filters (Problem: gopher_quality_filters)
# ==============================================================================
#
# USAGE:
#   cd cs336_data/filtering_cc/quality_rules
#   ./part_2_6.sh
#
# WHAT IT DOES:
#   Runs all seven Gopher quality filter tests.
#
# ==============================================================================

set -e
cd "$(dirname "$0")"

echo "=== Run Gopher quality filter tests ==="
cd ../../..
.venv/bin/pytest -k "test_gopher" -v

echo ""
echo "=== Done ==="