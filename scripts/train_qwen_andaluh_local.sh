#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${MCM_ARTIFACT_ROOT:-$PWD}"
MODEL_NAME="${MCM_MODEL_NAME:-Qwen/Qwen3.5-0.8B-Base}"
RUN_SLUG="${MCM_RUN_SLUG:-qwen_andaluh_08b}"
BASE_OUT_DIR="${MCM_BASE_OUT_DIR:-${ARTIFACT_ROOT}/data/processed/base}"
TOKENIZER_DIR="${MCM_TOKENIZER_DIR:-${ARTIFACT_ROOT}/outputs/tokenizers/qwen35_andaluh}"
LORA_R="${MCM_LORA_R:-16}"
LORA_ALPHA="${MCM_LORA_ALPHA:-32}"
CPT_LR="${MCM_CPT_LR:-5e-5}"
SFT_LR="${MCM_SFT_LR:-1e-4}"
ORPO_LR="${MCM_ORPO_LR:-5e-6}"
ORPO_BETA="${MCM_ORPO_BETA:-0.1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-12}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-12}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-12}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-${ARTIFACT_ROOT}/.hf_cache}"
export WANDB_PROJECT="${WANDB_PROJECT:-Qwen-Andaluh}"

start_gpu_log() {
  local run_id="$1"
  mkdir -p "${ARTIFACT_ROOT}/reports/gpu_history"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi \
      --query-gpu=timestamp,name,memory.used,memory.total,power.draw,temperature.gpu,utilization.gpu \
      --format=csv \
      -l 5 > "${ARTIFACT_ROOT}/reports/gpu_history/${run_id}.csv" &
    echo "$!" > "${ARTIFACT_ROOT}/reports/gpu_history/${run_id}.pid"
  fi
}

stop_gpu_log() {
  local run_id="$1"
  local pid_file="${ARTIFACT_ROOT}/reports/gpu_history/${run_id}.pid"
  if [[ -f "${pid_file}" ]]; then
    local pid
    pid="$(cat "${pid_file}")"
    kill "${pid}" 2>/dev/null || true
    rm -f "${pid_file}"
  fi
}

run_cpt="${RUN_SLUG}_cpt"
start_gpu_log "${run_cpt}"
trap 'stop_gpu_log "${run_cpt}"' EXIT
uv run accelerate launch src/marichatmen/train/train_cpt.py \
  --model_name "${MODEL_NAME}" \
  --tokenizer_name "${TOKENIZER_DIR}" \
  --train_file "${BASE_OUT_DIR}/cpt_train.jsonl" \
  --valid_file "${BASE_OUT_DIR}/cpt_valid.jsonl" \
  --spanish_eval_file "${BASE_OUT_DIR}/cpt_spanish_valid.jsonl" \
  --andaluh_eval_file "${BASE_OUT_DIR}/cpt_andaluh_valid.jsonl" \
  --output_dir "${ARTIFACT_ROOT}/outputs/${RUN_SLUG}_cpt" \
  --max_seq_length "${MCM_MAX_SEQ_LENGTH:-512}" \
  --max_steps "${MCM_CPT_MAX_STEPS:-20}" \
  --learning_rate "${CPT_LR}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "${MCM_GRAD_ACCUM:-16}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout 0.05 \
  --resize_token_embeddings true \
  --train_embeddings true \
  --gradient_checkpointing true \
  --bf16 true \
  --report_to "${MCM_REPORT_TO:-none}" \
  --run_name "${run_cpt}"
trap - EXIT
stop_gpu_log "${run_cpt}"

run_sft="${RUN_SLUG}_sft"
start_gpu_log "${run_sft}"
trap 'stop_gpu_log "${run_sft}"' EXIT
uv run accelerate launch src/marichatmen/train/train_sft.py \
  --model_name "${MODEL_NAME}" \
  --tokenizer_name "${TOKENIZER_DIR}" \
  --base_adapter "${ARTIFACT_ROOT}/outputs/${RUN_SLUG}_cpt/final_adapter" \
  --train_file "${BASE_OUT_DIR}/sft_train.jsonl" \
  --valid_file "${BASE_OUT_DIR}/sft_valid.jsonl" \
  --output_dir "${ARTIFACT_ROOT}/outputs/${RUN_SLUG}_sft" \
  --max_seq_length "${MCM_MAX_SEQ_LENGTH:-512}" \
  --max_steps "${MCM_SFT_MAX_STEPS:-20}" \
  --learning_rate "${SFT_LR}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "${MCM_GRAD_ACCUM:-16}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout 0.05 \
  --resize_token_embeddings true \
  --train_embeddings true \
  --gradient_checkpointing true \
  --bf16 true \
  --report_to "${MCM_REPORT_TO:-none}" \
  --run_name "${run_sft}"
trap - EXIT
stop_gpu_log "${run_sft}"

run_orpo="${RUN_SLUG}_orpo"
start_gpu_log "${run_orpo}"
trap 'stop_gpu_log "${run_orpo}"' EXIT
uv run accelerate launch src/marichatmen/train/train_orpo.py \
  --model_name "${MODEL_NAME}" \
  --tokenizer_name "${TOKENIZER_DIR}" \
  --sft_adapter "${ARTIFACT_ROOT}/outputs/${RUN_SLUG}_sft/final_adapter" \
  --train_file "${BASE_OUT_DIR}/orpo_train.jsonl" \
  --valid_file "${BASE_OUT_DIR}/orpo_valid.jsonl" \
  --output_dir "${ARTIFACT_ROOT}/outputs/${RUN_SLUG}_orpo" \
  --max_seq_length "${MCM_MAX_SEQ_LENGTH:-512}" \
  --max_steps "${MCM_ORPO_MAX_STEPS:-10}" \
  --learning_rate "${ORPO_LR}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "${MCM_GRAD_ACCUM:-16}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --resize_token_embeddings true \
  --train_embeddings true \
  --beta "${ORPO_BETA}" \
  --max_completion_length "${MCM_ORPO_MAX_COMPLETION_LENGTH:-384}" \
  --max_grad_norm "${MCM_ORPO_MAX_GRAD_NORM:-0.3}" \
  --gradient_checkpointing true \
  --bf16 true \
  --report_to "${MCM_REPORT_TO:-none}" \
  --run_name "${run_orpo}"
