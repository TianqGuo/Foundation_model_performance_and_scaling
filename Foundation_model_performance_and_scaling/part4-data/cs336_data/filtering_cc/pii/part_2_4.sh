#!/bin/bash
# ==============================================================================
# Section 2.4 – PII Masking (Problem: mask_pii)
# ==============================================================================
#
# USAGE:
#   cd cs336_data/filtering_cc/pii
#   ./part_2_4.sh
#
# WHAT IT DOES:
#   Runs all five PII masking tests (emails, phones, IPs).
#
# ==============================================================================

set -e
cd "$(dirname "$0")"

echo "=== Run PII masking tests ==="
cd ../../..
.venv/bin/pytest -k "test_mask" -v

echo ""
echo "=== Done ==="