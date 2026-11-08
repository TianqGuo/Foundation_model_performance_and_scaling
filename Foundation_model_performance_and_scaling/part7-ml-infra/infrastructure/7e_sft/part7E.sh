#!/usr/bin/env bash
# Usage: part7E.sh check | prepare [options] | train --check --bundle PATH
# Host workflow is CPU-only until image/checkpoint/E2E integration is complete.
set -euo pipefail
part7_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
case "${1:---help}" in
  check|prepare)
    exec python3 "$part7_root/infrastructure/7e_sft/prepare.py" "$@"
    ;;
  train)
    shift
    exec python3 "$part7_root/infrastructure/7e_sft/run_7e.py" "$@"
    ;;
  --help)
    echo 'part7E.sh check | prepare [options] | train --check --bundle PATH'
    echo 'train --cloud is an internal RayJob helper; image/checkpoint/E2E remain pending.'
    ;;
  *) echo 'Unknown action; use --help.' >&2; exit 2 ;;
esac
