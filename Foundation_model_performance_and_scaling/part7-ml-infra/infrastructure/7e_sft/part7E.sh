#!/usr/bin/env bash
# Usage: part7E.sh check | prepare [--output PATH] [--source PATH] [--model-revision SHA]
# CPU preparation only. Cloud training/image/E2E commands are not implemented yet.
set -euo pipefail
part7_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
case "${1:---help}" in
  check|prepare)
    exec python3 "$part7_root/infrastructure/7e_sft/prepare.py" "$@"
    ;;
  --help)
    echo 'part7E.sh check | prepare [options]; no cloud execution yet'
    ;;
  *) echo 'Unknown action; use --help.' >&2; exit 2 ;;
esac
