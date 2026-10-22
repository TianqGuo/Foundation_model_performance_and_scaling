#!/usr/bin/env bash
# Usage: part7D.sh {render|image|benchmark|e2e} [options]
# Cloud execution requires --cloud; image publication additionally requires --publish.
set -euo pipefail
part7_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
case "${1:---help}" in
  e2e)
    shift
    exec python3 "$part7_root/infrastructure/7d_serve/e2e.py" "$@"
    ;;
  benchmark)
    shift
    exec python3 "$part7_root/infrastructure/7d_serve/benchmark.py" "$@"
    ;;
  image)
    shift
    exec python3 "$part7_root/infrastructure/7d_serve/image_publish.py" "$@"
    ;;
  render)
    shift
    exec python3 "$part7_root/infrastructure/7d_serve/render.py" "$@"
    ;;
  --help)
    echo 'part7D.sh render --image REGISTRY/IMAGE@sha256:DIGEST --node-name NODE [--output PATH]'
    echo 'part7D.sh image --check | --cloud [--publish] [--repository USER/REPO]'
    echo 'part7D.sh benchmark --cloud --prompts JSONL --output REPORT [options] (future E2E helper)'
    echo 'part7D.sh e2e --check --image IMAGE_DIGEST (cloud mode remains unverified)'
    echo 'Serving compatibility and full cloud E2E remain pending; do not provision yet.'
    ;;
  *) echo 'Unknown action; use --help.' >&2; exit 2 ;;
esac
