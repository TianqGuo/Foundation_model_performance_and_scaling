#!/bin/bash
# ==============================================================================
# Section 3.1 – Exact Line Deduplication (Problem: exact_deduplication)
# ==============================================================================
#
# USAGE:
#   cd cs336_data/deduplication/exact_line
#   ./part_3_1.sh
#
# WHAT IT DOES:
#   Runs the exact line deduplication test. The function makes two passes over
#   the input files: first to count line occurrences (keyed by MD5 hash), then
#   to rewrite each file keeping only lines that appear exactly once.
#
# OUTPUT:
#   Deduplicated files written to the output directory (see test for tmp_path).
#
# NOTES:
#   - Lines that appear in more than one file are removed from all copies.
#   - Empty files are produced when all lines of a document are duplicates.
#   - Hash collisions are negligible for typical corpus sizes with MD5.
#
# ==============================================================================

set -e
cd "$(dirname "$0")"
ROOT="$(cd ../../.. && pwd)"

echo "=== Section 3.1: Exact Line Deduplication ==="
cd "${ROOT}"
.venv/bin/pytest -k test_exact_line_deduplication -v

echo ""
echo "=== Done ==="
