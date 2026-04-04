#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/autodl-tmp/Distill-Safety-Aligned-Models"
MODEL_DIR="/root/autodl-tmp/LLM/Llama-3.2-1B-Instruct"
OUTPUT_DIR="${REPO_ROOT}/outputs/baseline"
LOG_DIR="${REPO_ROOT}/logs/baseline"
LOG_FILE="${LOG_DIR}/run_$(date -u +%Y%m%d_%H%M%S).log"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

cd "${REPO_ROOT}"
python -m train.sft_baseline \
  --train_file "${REPO_ROOT}/data/train.jsonl" \
  --eval_file "${REPO_ROOT}/data/val.jsonl" \
  --model_local_dir "${MODEL_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --logging_dir "${LOG_DIR}" \
  --max_length 1024 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 3 \
  --learning_rate 2e-4 \
  --logging_steps 10 \
  --save_steps 200 \
  --eval_steps 200 \
  --save_total_limit 2 \
  "$@" 2>&1 | tee "${LOG_FILE}"
