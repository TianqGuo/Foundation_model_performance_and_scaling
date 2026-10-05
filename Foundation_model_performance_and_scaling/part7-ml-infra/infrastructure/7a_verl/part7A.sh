#!/usr/bin/env bash
# CLOUD ONLY. Usage: bash infrastructure/7a_verl/part7A.sh [run_7a.py path/model overrides]
# Creates/reuses the isolated environment, prepares/checks data, and runs 7A.
# Examples: bash infrastructure/7a_verl/part7A.sh; bash infrastructure/7a_verl/part7A.sh --model-path /data/models/Qwen
# Transfer source and either the prepared bundle or Part 5 references/data first.
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
  echo 'Cloud usage: bash infrastructure/7a_verl/part7A.sh [--sync-env] [--model-path PATH] [--model-id REPO] [--model-revision REV] [--data PATH] [--output PATH]'
  exit 0
fi
sync_environment=0
if [[ "${1:-}" == --sync-env ]]; then
  sync_environment=1
  shift
fi
part7_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$part7_root"
mkdir -p results/7a
execution_logs="$(mktemp -d "$part7_root/results/7a/execution_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
exec > >(tee "$execution_logs/console.log") 2>&1
echo "Execution logs: $execution_logs"

python_path="$part7_root/.venv-7a/bin/python"
if [[ "$sync_environment" == 1 && -d .venv-7a ]]; then
  timeout --signal=INT --kill-after=60s 30m bash infrastructure/7a_verl/environment/setup_7a.sh --cloud --sync-existing
elif [[ ! -d .venv-7a ]]; then
  timeout --signal=INT --kill-after=60s 30m bash infrastructure/7a_verl/environment/setup_7a.sh --cloud
elif [[ ! -f .venv-7a/setup_inputs.sha256 ]]; then
  echo 'Existing environment has no successful setup record. Preserve it and use a fresh environment directory.' >&2
  exit 1
fi
sha256sum --check .venv-7a/setup_inputs.sha256
source "$part7_root/infrastructure/7a_verl/environment/cuda_7a.sh"
select_7a_cuda
"$python_path" -c 'import torch, vllm, verl, flash_attn, ray; assert torch.cuda.is_available() and torch.cuda.device_count() == 2, "Two cloud GPUs required"'

# Inspect only the data override; pass model/path options unchanged to argparse.
data_bundle="$part7_root/results/7a/step2_math_final"
runner_args=("$@")
for ((argument=0; argument<${#runner_args[@]}; argument++)); do
  if [[ "${runner_args[argument]}" == --data ]]; then
    data_bundle="${runner_args[argument+1]:?--data requires a path}"
  elif [[ "${runner_args[argument]}" == --data=* ]]; then
    data_bundle="${runner_args[argument]#--data=}"
  fi
done
if [[ ! -f "$data_bundle/manifest.json" ]]; then
  echo 'Preparing MATH bundle from discovered Part 5 references/data.'
  "$python_path" workloads/math_workload.py --output "$data_bundle" --drop-empty-solutions
fi
"$python_path" infrastructure/7a_verl/run_7a.py --check-config "${runner_args[@]}" > "$execution_logs/resolved_preview.yaml"
timeout --signal=INT --kill-after=60s 60m "$python_path" -u infrastructure/7a_verl/run_7a.py --cloud "${runner_args[@]}"
echo "7A execution finished. Review results/7a and retain artifacts before releasing the instance."
