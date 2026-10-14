#!/usr/bin/env bash
# Usage: part7C.sh setup|render|phase|job|experiment [CLI options] | build --cloud BASE_DIGEST LOCAL_TAG
# render/job --check are CPU-only; job --cloud submits to an existing cluster.
set -euo pipefail
part7_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
case "${1:---help}" in
  setup)
    shift
    exec python3 "$part7_root/infrastructure/7c_kuberay/setup_cluster.py" "$@"
    ;;
  render)
    shift
    exec python3 "$part7_root/infrastructure/7c_kuberay/render.py" "$@"
    ;;
  cluster)
    shift
    exec python3 "$part7_root/infrastructure/7c_kuberay/cluster_check.py" "$@"
    ;;
  experiment)
    shift
    exec python3 "$part7_root/infrastructure/7c_kuberay/experiment.py" "$@"
    ;;
  job)
    shift
    exec python3 "$part7_root/infrastructure/7c_kuberay/job_control.py" "$@"
    ;;
  phase)
    shift
    exec python3 "$part7_root/infrastructure/7c_kuberay/run_7c.py" "$@"
    ;;
  build)
    shift
    exec bash "$part7_root/infrastructure/7c_kuberay/build_image.sh" "$@"
    ;;
  --help)
    echo 'part7C.sh render --image REGISTRY/IMAGE@sha256:DIGEST --node-name NODE [--namespace part7] [--pvc part7-workspace] [--secret part7-huggingface] [--output PATH]'
    echo 'part7C.sh build --cloud BASE_IMAGE@sha256:DIGEST LOCAL_TAG'
    echo 'part7C.sh job --check --manifest PATH | --cloud --manifest PATH --context CONTEXT --output PATH [--cleanup-on-success]'
    echo 'part7C.sh experiment --check|--cloud --image IMAGE_DIGEST --node-name NODE [--context CONTEXT] [--data BUNDLE]'
    echo 'part7C.sh cluster --context CONTEXT --node-name NODE [--operator-namespace NAMESPACE] [--output PATH]'
    echo 'part7C.sh setup --check|--cloud [--output PATH] (fresh Ubuntu GPU VM only)'
    ;;
  *) echo 'Unknown action; use --help.' >&2; exit 2 ;;
esac
