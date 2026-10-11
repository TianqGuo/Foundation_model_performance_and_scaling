#!/usr/bin/env bash
# CLOUD ONLY. Usage: bash infrastructure/7b_ray/part7B.sh [--sync-env] [experiment.py options]
# Reuses the 7A environment; runs baseline, planned stop, direct Hub retention,
# pinned download and native resume. No VM provisioning or Git changes.
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
  echo 'Cloud: bash infrastructure/7b_ray/part7B.sh [--sync-env] [--repo OWNER/REPO] [--data PATH] [--output PATH] [--model-id REPO] [--model-path PATH] [--model-revision COMMIT]'
  echo 'Requires two cloud GPUs, 100 GiB free disk and private Hub storage for one full checkpoint; prompts once for a write token if needed.'
  exit 0
fi
part7_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$part7_root"
sync_environment=0
runner_args=()
for argument in "$@"; do
  case "$argument" in
    --sync-env) sync_environment=1 ;;
    *) runner_args+=("$argument") ;;
  esac
done
mkdir -p results/7b
execution_logs="$(mktemp -d "$part7_root/results/7b/execution_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
exec > >(tee "$execution_logs/console.log") 2>&1
echo "Execution logs: $execution_logs"
export PART7_EXECUTION_LOGS="$execution_logs"
python_path="$part7_root/.venv-7a/bin/python"
if [[ "$sync_environment" == 1 && -d .venv-7a ]]; then
  timeout --signal=INT --kill-after=60s 30m bash infrastructure/7a_verl/environment/setup_7a.sh --cloud --sync-existing
elif [[ ! -d .venv-7a ]]; then
  timeout --signal=INT --kill-after=60s 30m bash infrastructure/7a_verl/environment/setup_7a.sh --cloud
fi
sha256sum --check .venv-7a/setup_inputs.sha256
source infrastructure/7a_verl/environment/cuda_7a.sh
select_7a_cuda
"$python_path" -c 'import torch, vllm, verl, flash_attn, ray; assert torch.cuda.is_available() and torch.cuda.device_count() == 2, "Two cloud GPUs required"'
data_bundle="$part7_root/results/7a/step2_math_final"
run_output=""
for ((argument=0; argument<${#runner_args[@]}; argument++)); do
  case "${runner_args[argument]}" in
    --data) data_bundle="${runner_args[argument+1]:?--data requires a path}" ;;
    --data=*) data_bundle="${runner_args[argument]#--data=}" ;;
    --output) run_output="${runner_args[argument+1]:?--output requires a path}" ;;
    --output=*) run_output="${runner_args[argument]#--output=}" ;;
  esac
done
if [[ -z "$run_output" ]]; then
  run_output="$part7_root/results/7b/recovery_$(basename "$execution_logs")"
  runner_args+=(--output "$run_output")
fi
if [[ ! -f "$data_bundle/manifest.json" ]]; then
  "$python_path" workloads/math_workload.py --output "$data_bundle" --drop-empty-solutions
fi
# Authenticate in the foreground: timeout creates a process group that cannot
# read the terminal, so a first-time hidden-token prompt would receive SIGTTIN.
"$python_path" -u infrastructure/7b_ray/experiment.py --cloud --authenticate-only "${runner_args[@]}"
timeout --signal=INT --kill-after=60s 60m "$python_path" -u infrastructure/7b_ray/experiment.py --cloud "${runner_args[@]}"
mkdir -p "$run_output/execution" "$run_output/environment"
cp "$execution_logs/console.log" "$run_output/execution/"
cp -a results/7a/environment/. "$run_output/environment/"
echo '7B experiment finished. Retain logs and review recovery_review.json before releasing the instance.'
