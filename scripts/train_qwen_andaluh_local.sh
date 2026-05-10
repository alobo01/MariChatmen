#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

ARTIFACT_ROOT="${MCM_ARTIFACT_ROOT:-${PWD}/.artifacts}"
MODEL_NAME="${MCM_MODEL_NAME:-Qwen/Qwen3.5-0.8B-Base}"
RUN_SLUG="${MCM_RUN_SLUG:-qwen_andaluh_08b}"
BASE_OUT_DIR="${MCM_BASE_OUT_DIR:-${ARTIFACT_ROOT}/data/processed/base}"
TOKENIZER_DIR="${MCM_TOKENIZER_DIR:-${ARTIFACT_ROOT}/outputs/tokenizers/qwen35_andaluh}"
DEFAULT_GENERATION_PROBE_FILE="${MCM_GENERATION_PROBE_FILE:-examples/prompts/qwen_andaluh_5.txt}"
CPT_ADAPTER_DIR="${ARTIFACT_ROOT}/outputs/${RUN_SLUG}_cpt/final_adapter"
SFT_ADAPTER_DIR="${ARTIFACT_ROOT}/outputs/${RUN_SLUG}_sft/final_adapter"
CPT_TRAIN_FILE="${MCM_CPT_TRAIN_FILE:-${BASE_OUT_DIR}/cpt_train.jsonl}"
CPT_VALID_FILE="${MCM_CPT_VALID_FILE:-${BASE_OUT_DIR}/cpt_valid.jsonl}"
CPT_SPANISH_EVAL_FILE="${MCM_CPT_SPANISH_EVAL_FILE:-${BASE_OUT_DIR}/cpt_spanish_valid.jsonl}"
CPT_ANDALUH_EVAL_FILE="${MCM_CPT_ANDALUH_EVAL_FILE:-${BASE_OUT_DIR}/cpt_andaluh_valid.jsonl}"
SFT_TRAIN_FILE="${MCM_SFT_TRAIN_FILE:-${BASE_OUT_DIR}/sft_train.jsonl}"
SFT_VALID_FILE="${MCM_SFT_VALID_FILE:-${BASE_OUT_DIR}/sft_valid.jsonl}"
ORPO_TRAIN_FILE="${MCM_ORPO_TRAIN_FILE:-${BASE_OUT_DIR}/orpo_train.jsonl}"
ORPO_VALID_FILE="${MCM_ORPO_VALID_FILE:-${BASE_OUT_DIR}/orpo_valid.jsonl}"
LORA_R="${MCM_LORA_R:-16}"
LORA_ALPHA="${MCM_LORA_ALPHA:-32}"
CPT_LR="${MCM_CPT_LR:-5e-5}"
SFT_LR="${MCM_SFT_LR:-5e-5}"
ORPO_LR="${MCM_ORPO_LR:-3e-6}"
ORPO_BETA="${MCM_ORPO_BETA:-0.05}"
WEIGHT_DECAY="${MCM_WEIGHT_DECAY:-0.01}"
CPT_LORA_DROPOUT="${MCM_CPT_LORA_DROPOUT:-0.05}"
SFT_LORA_DROPOUT="${MCM_SFT_LORA_DROPOUT:-0.08}"
ORPO_LORA_DROPOUT="${MCM_ORPO_LORA_DROPOUT:-0.08}"
EARLY_STOPPING_PATIENCE="${MCM_EARLY_STOPPING_PATIENCE:-0}"
EARLY_STOPPING_THRESHOLD="${MCM_EARLY_STOPPING_THRESHOLD:-0.0}"
LOAD_BEST_MODEL_AT_END="${MCM_LOAD_BEST_MODEL_AT_END:-false}"
EVAL_STEPS="${MCM_EVAL_STEPS:-50}"
SAVE_STEPS="${MCM_SAVE_STEPS:-100}"
DATALOADER_NUM_WORKERS="${MCM_DATALOADER_NUM_WORKERS:-4}"
DATALOADER_PREFETCH_FACTOR="${MCM_DATALOADER_PREFETCH_FACTOR:-2}"
PREPROCESSING_NUM_WORKERS="${MCM_PREPROCESSING_NUM_WORKERS:-8}"
EVAL_BATCH_SIZE="${MCM_EVAL_BATCH_SIZE:-4}"
CPU_BUDGET="${MCM_CPU_BUDGET:-12}"
TORCH_NUM_THREADS="${MCM_TORCH_NUM_THREADS:-$(( CPU_BUDGET > DATALOADER_NUM_WORKERS ? CPU_BUDGET - DATALOADER_NUM_WORKERS : 1 ))}"
RUN_CPT="${MCM_RUN_CPT:-1}"
RUN_SFT="${MCM_RUN_SFT:-1}"
RUN_ORPO="${MCM_RUN_ORPO:-0}"
HOLD_AFTER_CPT="${MCM_HOLD_AFTER_CPT:-0}"

DEFAULT_TRAIN_BATCH_SIZE=1
DEFAULT_GRAD_ACCUM=16
case "${MODEL_NAME}" in
  *0.8B*|*0_8B*)
    # The local RTX 5060 has enough headroom for 0.8B QLoRA at micro-batch 4.
    # This keeps the effective batch near 16 while reducing launcher/eval overhead
    # relative to the earlier micro-batch 1/2 diagnostic runs.
    DEFAULT_TRAIN_BATCH_SIZE=4
    DEFAULT_GRAD_ACCUM=4
    ;;
esac

tokenizer_for_adapter_or_default() {
  local adapter_dir="$1"
  if [[ "${MCM_FORCE_TOKENIZER_DIR:-0}" != "1" && -f "${adapter_dir}/tokenizer_config.json" ]]; then
    printf '%s\n' "${adapter_dir}"
    return
  fi
  printf '%s\n' "${TOKENIZER_DIR}"
}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${TORCH_NUM_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${TORCH_NUM_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${TORCH_NUM_THREADS}}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-${TORCH_NUM_THREADS}}"
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

if [[ "${RUN_CPT}" == "1" ]]; then
  run_cpt="${RUN_SLUG}_cpt"
  start_gpu_log "${run_cpt}"
  trap 'stop_gpu_log "${run_cpt}"' EXIT
  uv run accelerate launch src/marichatmen/train/train_cpt.py \
  --model_name "${MODEL_NAME}" \
  --tokenizer_name "${TOKENIZER_DIR}" \
  --train_file "${CPT_TRAIN_FILE}" \
  --valid_file "${CPT_VALID_FILE}" \
  --spanish_eval_file "${CPT_SPANISH_EVAL_FILE}" \
  --andaluh_eval_file "${CPT_ANDALUH_EVAL_FILE}" \
  --output_dir "${ARTIFACT_ROOT}/outputs/${RUN_SLUG}_cpt" \
  --max_seq_length "${MCM_MAX_SEQ_LENGTH:-512}" \
  --max_steps "${MCM_CPT_MAX_STEPS:-20}" \
  --learning_rate "${CPT_LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --per_device_train_batch_size "${MCM_CPT_BATCH_SIZE:-${MCM_TRAIN_BATCH_SIZE:-${DEFAULT_TRAIN_BATCH_SIZE}}}" \
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${MCM_GRAD_ACCUM:-${DEFAULT_GRAD_ACCUM}}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}" \
  --preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${CPT_LORA_DROPOUT}" \
  --resize_token_embeddings true \
  --train_embeddings "${MCM_CPT_TRAIN_EMBEDDINGS:-${MCM_TRAIN_EMBEDDINGS:-true}}" \
  --gradient_checkpointing true \
  --bf16 true \
  --eval_steps "${EVAL_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --load_best_model_at_end "${LOAD_BEST_MODEL_AT_END}" \
  --early_stopping_patience "${EARLY_STOPPING_PATIENCE}" \
  --early_stopping_threshold "${EARLY_STOPPING_THRESHOLD}" \
  --generation_probe_file "${DEFAULT_GENERATION_PROBE_FILE}" \
  --generation_probe_output "${MCM_GENERATION_PROBE_OUTPUT:-}" \
  --generation_probe_limit "${MCM_GENERATION_PROBE_LIMIT:-5}" \
  --generation_probe_max_new_tokens "${MCM_GENERATION_PROBE_MAX_NEW_TOKENS:-128}" \
  --report_to "${MCM_REPORT_TO:-none}" \
  --run_name "${run_cpt}"
  trap - EXIT
  stop_gpu_log "${run_cpt}"
fi

if [[ "${HOLD_AFTER_CPT}" == "1" ]]; then
  echo "MCM_HOLD_AFTER_CPT=1; stopping after CPT before SFT/ORPO."
  exit 0
fi

if [[ "${RUN_SFT}" == "1" ]]; then
  run_sft="${RUN_SLUG}_sft"
  sft_tokenizer="$(tokenizer_for_adapter_or_default "${CPT_ADAPTER_DIR}")"
  start_gpu_log "${run_sft}"
  trap 'stop_gpu_log "${run_sft}"' EXIT
  uv run accelerate launch src/marichatmen/train/train_sft.py \
  --model_name "${MODEL_NAME}" \
  --tokenizer_name "${sft_tokenizer}" \
  --base_adapter "${CPT_ADAPTER_DIR}" \
  --train_file "${SFT_TRAIN_FILE}" \
  --valid_file "${SFT_VALID_FILE}" \
  --output_dir "${ARTIFACT_ROOT}/outputs/${RUN_SLUG}_sft" \
  --max_seq_length "${MCM_MAX_SEQ_LENGTH:-512}" \
  --max_steps "${MCM_SFT_MAX_STEPS:-20}" \
  --learning_rate "${SFT_LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --per_device_train_batch_size "${MCM_SFT_BATCH_SIZE:-${MCM_TRAIN_BATCH_SIZE:-${DEFAULT_TRAIN_BATCH_SIZE}}}" \
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${MCM_GRAD_ACCUM:-${DEFAULT_GRAD_ACCUM}}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}" \
  --preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${SFT_LORA_DROPOUT}" \
  --resize_token_embeddings true \
  --train_embeddings "${MCM_SFT_TRAIN_EMBEDDINGS:-${MCM_TRAIN_EMBEDDINGS:-true}}" \
  --gradient_checkpointing true \
  --bf16 true \
  --eval_steps "${EVAL_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --load_best_model_at_end "${LOAD_BEST_MODEL_AT_END}" \
  --early_stopping_patience "${EARLY_STOPPING_PATIENCE}" \
  --early_stopping_threshold "${EARLY_STOPPING_THRESHOLD}" \
  --generation_probe_file "${DEFAULT_GENERATION_PROBE_FILE}" \
  --generation_probe_output "${MCM_GENERATION_PROBE_OUTPUT:-}" \
  --generation_probe_limit "${MCM_GENERATION_PROBE_LIMIT:-5}" \
  --generation_probe_max_new_tokens "${MCM_GENERATION_PROBE_MAX_NEW_TOKENS:-128}" \
  --report_to "${MCM_REPORT_TO:-none}" \
  --run_name "${run_sft}"
  trap - EXIT
  stop_gpu_log "${run_sft}"
fi

if [[ "${RUN_ORPO}" == "1" ]]; then
  run_orpo="${RUN_SLUG}_orpo"
  orpo_tokenizer="$(tokenizer_for_adapter_or_default "${SFT_ADAPTER_DIR}")"
  start_gpu_log "${run_orpo}"
  trap 'stop_gpu_log "${run_orpo}"' EXIT
  uv run accelerate launch src/marichatmen/train/train_orpo.py \
  --model_name "${MODEL_NAME}" \
  --tokenizer_name "${orpo_tokenizer}" \
  --sft_adapter "${SFT_ADAPTER_DIR}" \
  --train_file "${ORPO_TRAIN_FILE}" \
  --valid_file "${ORPO_VALID_FILE}" \
  --output_dir "${ARTIFACT_ROOT}/outputs/${RUN_SLUG}_orpo" \
  --max_seq_length "${MCM_MAX_SEQ_LENGTH:-512}" \
  --max_steps "${MCM_ORPO_MAX_STEPS:-10}" \
  --learning_rate "${ORPO_LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --per_device_train_batch_size "${MCM_ORPO_BATCH_SIZE:-${MCM_TRAIN_BATCH_SIZE:-${DEFAULT_TRAIN_BATCH_SIZE}}}" \
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${MCM_GRAD_ACCUM:-${DEFAULT_GRAD_ACCUM}}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}" \
  --preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${ORPO_LORA_DROPOUT}" \
  --resize_token_embeddings true \
  --train_embeddings "${MCM_ORPO_TRAIN_EMBEDDINGS:-${MCM_TRAIN_EMBEDDINGS:-true}}" \
  --beta "${ORPO_BETA}" \
  --max_completion_length "${MCM_ORPO_MAX_COMPLETION_LENGTH:-384}" \
  --max_grad_norm "${MCM_ORPO_MAX_GRAD_NORM:-0.3}" \
  --gradient_checkpointing true \
  --bf16 true \
  --eval_steps "${EVAL_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --load_best_model_at_end "${LOAD_BEST_MODEL_AT_END}" \
  --early_stopping_patience "${EARLY_STOPPING_PATIENCE}" \
  --early_stopping_threshold "${EARLY_STOPPING_THRESHOLD}" \
  --generation_probe_file "${DEFAULT_GENERATION_PROBE_FILE}" \
  --generation_probe_output "${MCM_GENERATION_PROBE_OUTPUT:-}" \
  --generation_probe_limit "${MCM_GENERATION_PROBE_LIMIT:-5}" \
  --generation_probe_max_new_tokens "${MCM_GENERATION_PROBE_MAX_NEW_TOKENS:-128}" \
  --report_to "${MCM_REPORT_TO:-none}" \
  --run_name "${run_orpo}"
  trap - EXIT
  stop_gpu_log "${run_orpo}"
fi
