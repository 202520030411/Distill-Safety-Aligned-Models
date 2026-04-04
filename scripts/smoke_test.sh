#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/autodl-tmp/Distill-Safety-Aligned-Models"
MODEL_DIR="/root/autodl-tmp/LLM/Llama-3.2-1B-Instruct"
SMOKE_DATA_DIR="${REPO_ROOT}/data/processed_smoke"
LOG_DIR="${REPO_ROOT}/logs/smoke"
mkdir -p "${LOG_DIR}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

run_smoke () {
  local variant="$1"
  local module="$2"
  shift 2
  local output_dir="${REPO_ROOT}/outputs/smoke_${variant}"
  local log_file="${LOG_DIR}/${variant}_$(date -u +%Y%m%d_%H%M%S).log"

  mkdir -p "${output_dir}" "${LOG_DIR}/${variant}"
  python -m "${module}" \
    --model_local_dir "${MODEL_DIR}" \
    --output_dir "${output_dir}" \
    --logging_dir "${LOG_DIR}/${variant}" \
    --use_smoke_data \
    --smoke_data_dir "${SMOKE_DATA_DIR}" \
    --smoke_train_size 24 \
    --smoke_eval_size 8 \
    --max_length 512 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --max_steps 2 \
    --learning_rate 1e-4 \
    --logging_steps 1 \
    --save_steps 2 \
    --eval_steps 2 \
    --save_total_limit 1 \
    "$@" 2>&1 | tee "${log_file}"
}

run_smoke baseline train.sft_baseline
run_smoke with_refusals train.sft_with_refusals
run_smoke weighted train.sft_weighted --refusal_weight 2.5
