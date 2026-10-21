#!/usr/bin/env bash
# Usage: part7D.sh render --image SERVING_IMAGE_DIGEST --node-name NODE [options]
# CPU-only rendering in this increment; cloud E2E runner is not ready yet.
set -euo pipefail
part7_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
case "${1:---help}" in
  image)
    shift
    exec bash "$part7_root/infrastructure/7d_serve/build_image.sh" "$@"
    ;;
  render)
    shift
    exec python3 "$part7_root/infrastructure/7d_serve/render.py" "$@"
    ;;
  --help)
    echo 'part7D.sh render --image REGISTRY/IMAGE@sha256:DIGEST --node-name NODE [--output PATH]'
    echo 'part7D.sh image --cloud [LOCAL_TAG] (candidate build/import gate only; no push)'
    echo 'Serving compatibility and full cloud E2E remain pending; do not provision yet.'
    ;;
  *) echo 'Unknown action; use --help.' >&2; exit 2 ;;
esac
