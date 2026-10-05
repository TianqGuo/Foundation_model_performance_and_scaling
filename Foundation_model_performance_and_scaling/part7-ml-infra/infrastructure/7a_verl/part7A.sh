#!/usr/bin/env bash
# CLOUD ONLY. Usage: bash infrastructure/7a_verl/part7A.sh [run_7a.py path/model overrides]
# Creates/reuses the isolated environment, prepares/checks data, and runs 7A.
# Examples: bash infrastructure/7a_verl/part7A.sh; bash infrastructure/7a_verl/part7A.sh --model-path /data/models/Qwen
# Transfer source and either the prepared bundle or Part 5 references/data first.
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
  echo 'Cloud usage: bash infrastructure/7a_verl/part7A.sh [--sync-env] [--upload-hf] [--model-path PATH] [--model-id REPO] [--model-revision REV] [--data PATH] [--output PATH]'
  echo '--upload-hf reuses Hugging Face login or prompts once, verifies token owner, and chooses private run repositories automatically.'
  echo 'Optional uploads: HF_MODEL_REPO=owner/model HF_ARTIFACT_REPO=owner/run HF_UPLOAD_CHECKPOINT=1 (private repos; authenticated cloud host)'
  exit 0
fi
sync_environment=0
upload_hf=0
runner_args=()
for argument in "$@"; do
  case "$argument" in
    --sync-env) sync_environment=1 ;;
    --upload-hf) upload_hf=1 ;;
    *) runner_args+=("$argument") ;;
  esac
done
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
run_output=""
for ((argument=0; argument<${#runner_args[@]}; argument++)); do
  if [[ "${runner_args[argument]}" == --data ]]; then
    data_bundle="${runner_args[argument+1]:?--data requires a path}"
  elif [[ "${runner_args[argument]}" == --data=* ]]; then
    data_bundle="${runner_args[argument]#--data=}"
  elif [[ "${runner_args[argument]}" == --output ]]; then
    run_output="${runner_args[argument+1]:?--output requires a path}"
  elif [[ "${runner_args[argument]}" == --output=* ]]; then
    run_output="${runner_args[argument]#--output=}"
  fi
done
if [[ -z "$run_output" ]]; then
  run_output="$part7_root/results/7a/smoke_$(basename "$execution_logs")"
  runner_args+=(--output "$run_output")
fi
if [[ "$upload_hf" == 1 ]]; then
  HF_MODEL_REPO="${HF_MODEL_REPO:-auto}"
  HF_ARTIFACT_REPO="${HF_ARTIFACT_REPO:-auto}"
fi
if [[ "${HF_UPLOAD_CHECKPOINT:-0}" != 0 && -z "${HF_ARTIFACT_REPO:-}" ]]; then
  echo 'HF_UPLOAD_CHECKPOINT requires HF_ARTIFACT_REPO.' >&2
  exit 1
fi
if [[ -n "${HF_MODEL_REPO:-}" || -n "${HF_ARTIFACT_REPO:-}" ]]; then
  prepare_args=()
  if [[ -n "${HF_MODEL_REPO:-}" ]]; then prepare_args+=(--model-repo "$HF_MODEL_REPO"); fi
  if [[ -n "${HF_ARTIFACT_REPO:-}" ]]; then prepare_args+=(--artifact-repo "$HF_ARTIFACT_REPO"); fi
  "$python_path" infrastructure/7a_verl/hf_artifacts.py --cloud --prepare-upload \
    --run "$run_output" --destinations-file "$execution_logs/hf_destinations.json" "${prepare_args[@]}"
  HF_MODEL_REPO="$("$python_path" -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_repo"] or "")' "$execution_logs/hf_destinations.json")"
  HF_ARTIFACT_REPO="$("$python_path" -c 'import json,sys; print(json.load(open(sys.argv[1]))["artifact_repo"] or "")' "$execution_logs/hf_destinations.json")"
fi
if [[ ! -f "$data_bundle/manifest.json" ]]; then
  echo 'Preparing MATH bundle from discovered Part 5 references/data.'
  "$python_path" workloads/math_workload.py --output "$data_bundle" --drop-empty-solutions
fi
"$python_path" infrastructure/7a_verl/run_7a.py --check-config "${runner_args[@]}" > "$execution_logs/resolved_preview.yaml"
timeout --signal=INT --kill-after=60s 60m "$python_path" -u infrastructure/7a_verl/run_7a.py --cloud "${runner_args[@]}"
mkdir -p "$run_output/execution" "$run_output/environment"
cp "$execution_logs/console.log" "$execution_logs/resolved_preview.yaml" "$run_output/execution/"
cp -a results/7a/environment/. "$run_output/environment/"
upload_args=()
if [[ -n "${HF_MODEL_REPO:-}" ]]; then upload_args+=(--model-repo "$HF_MODEL_REPO"); fi
if [[ -n "${HF_ARTIFACT_REPO:-}" ]]; then upload_args+=(--artifact-repo "$HF_ARTIFACT_REPO"); fi
if [[ "${HF_UPLOAD_CHECKPOINT:-0}" == 1 ]]; then upload_args+=(--include-checkpoint); fi
if (( ${#upload_args[@]} )); then
  "$python_path" -u infrastructure/7a_verl/hf_artifacts.py --cloud --run "$run_output" "${upload_args[@]}"
fi
echo "7A execution finished. Review results/7a and retain artifacts before releasing the instance."
