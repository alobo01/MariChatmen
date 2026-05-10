#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${MCM_ARTIFACT_ROOT:-${PWD}/.artifacts}"
MODEL_NAME="${MCM_MODEL_NAME:-Qwen/Qwen3.5-0.8B}"
BASE_SLUG="${MCM_BASE_RUN_SLUG:-qwen_andaluh_08b}"
BASE_STAGE="${MCM_PERSONA_BASE_STAGE:-orpo}"
RUN_SLUG="${MCM_RUN_SLUG:-marichatmen_08b}"
PERSONA_OUT_DIR="${MCM_PERSONA_OUT_DIR:-${ARTIFACT_ROOT}/data/processed/persona}"
TOKENIZER_DIR="${MCM_TOKENIZER_DIR:-${ARTIFACT_ROOT}/outputs/tokenizers/qwen35_andaluh}"
BASE_ADAPTER_DIR="${ARTIFACT_ROOT}/outputs/${BASE_SLUG}_${BASE_STAGE}/final_adapter"
PERSONA_SFT_ADAPTER_DIR="${ARTIFACT_ROOT}/outputs/${RUN_SLUG}_sft/final_adapter"
PERSONA_ORPO_ADAPTER_DIR="${ARTIFACT_ROOT}/outputs/${RUN_SLUG}_orpo/final_adapter"
PERSONA_SFT_TRAIN_FILE="${MCM_PERSONA_SFT_TRAIN_FILE:-${PERSONA_OUT_DIR}/persona_sft_train.jsonl}"
PERSONA_SFT_VALID_FILE="${MCM_PERSONA_SFT_VALID_FILE:-${PERSONA_OUT_DIR}/persona_sft_valid.jsonl}"
PERSONA_ORPO_TRAIN_FILE="${MCM_PERSONA_ORPO_TRAIN_FILE:-${PERSONA_OUT_DIR}/persona_orpo_train.jsonl}"
PERSONA_ORPO_VALID_FILE="${MCM_PERSONA_ORPO_VALID_FILE:-${PERSONA_OUT_DIR}/persona_orpo_valid.jsonl}"
PERSONA_POLICYOPT_PROMPTS_FILE="${MCM_PERSONA_POLICYOPT_PROMPTS_FILE:-${PERSONA_OUT_DIR}/persona_grpo_prompts.jsonl}"
PERSONA_SFT_MAX_SEQ_LENGTH="${MCM_PERSONA_SFT_MAX_SEQ_LENGTH:-${MCM_MAX_SEQ_LENGTH:-256}}"
PERSONA_ORPO_MAX_SEQ_LENGTH="${MCM_PERSONA_ORPO_MAX_SEQ_LENGTH:-${MCM_MAX_SEQ_LENGTH:-512}}"
PERSONA_ORPO_MAX_PROMPT_TOKENS="${MCM_PERSONA_ORPO_MAX_PROMPT_TOKENS:-384}"
PERSONA_ORPO_MIN_COMPLETION_TOKENS="${MCM_PERSONA_ORPO_MIN_COMPLETION_TOKENS:-8}"
LORA_R="${MCM_LORA_R:-16}"
LORA_ALPHA="${MCM_LORA_ALPHA:-32}"
SFT_LR="${MCM_PERSONA_SFT_LR:-${MCM_SFT_LR:-5e-5}}"
ORPO_LR="${MCM_PERSONA_ORPO_LR:-${MCM_ORPO_LR:-5e-6}}"
WEIGHT_DECAY="${MCM_WEIGHT_DECAY:-0.01}"
SFT_LORA_DROPOUT="${MCM_PERSONA_SFT_LORA_DROPOUT:-${MCM_SFT_LORA_DROPOUT:-0.05}}"
ORPO_LORA_DROPOUT="${MCM_PERSONA_ORPO_LORA_DROPOUT:-${MCM_ORPO_LORA_DROPOUT:-0.05}}"
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
export WANDB_PROJECT="${WANDB_PROJECT:-MariChatmen}"

if [[ "${MCM_ALLOW_PERSONA_TRAINING:-0}" != "1" ]]; then
  echo "Persona training is disabled by default. Run Qwen-Andaluh gates first, then set MCM_ALLOW_PERSONA_TRAINING=1 explicitly." >&2
  exit 2
fi

uv run accelerate launch src/marichatmen/train/train_sft.py \
  --model_name "${MODEL_NAME}" \
  --tokenizer_name "$(tokenizer_for_adapter_or_default "${BASE_ADAPTER_DIR}")" \
  --base_adapter "${BASE_ADAPTER_DIR}" \
  --train_file "${PERSONA_SFT_TRAIN_FILE}" \
  --valid_file "${PERSONA_SFT_VALID_FILE}" \
  --output_dir "${ARTIFACT_ROOT}/outputs/${RUN_SLUG}_sft" \
  --max_seq_length "${PERSONA_SFT_MAX_SEQ_LENGTH}" \
  --max_steps "${MCM_PERSONA_SFT_MAX_STEPS:-2}" \
  --learning_rate "${SFT_LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${MCM_GRAD_ACCUM:-1}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}" \
  --preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${SFT_LORA_DROPOUT}" \
  --resize_token_embeddings true \
  --train_embeddings "${MCM_PERSONA_SFT_TRAIN_EMBEDDINGS:-${MCM_TRAIN_EMBEDDINGS:-true}}" \
  --gradient_checkpointing true \
  --bf16 true \
  --eval_steps "${EVAL_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --load_best_model_at_end "${LOAD_BEST_MODEL_AT_END}" \
  --early_stopping_patience "${EARLY_STOPPING_PATIENCE}" \
  --early_stopping_threshold "${EARLY_STOPPING_THRESHOLD}" \
  --report_to "${MCM_REPORT_TO:-none}" \
  --run_name "${RUN_SLUG}_sft"

uv run accelerate launch src/marichatmen/train/train_orpo.py \
  --model_name "${MODEL_NAME}" \
  --tokenizer_name "$(tokenizer_for_adapter_or_default "${PERSONA_SFT_ADAPTER_DIR}")" \
  --sft_adapter "${PERSONA_SFT_ADAPTER_DIR}" \
  --train_file "${PERSONA_ORPO_TRAIN_FILE}" \
  --valid_file "${PERSONA_ORPO_VALID_FILE}" \
  --output_dir "${ARTIFACT_ROOT}/outputs/${RUN_SLUG}_orpo" \
  --max_seq_length "${PERSONA_ORPO_MAX_SEQ_LENGTH}" \
  --max_steps "${MCM_PERSONA_ORPO_MAX_STEPS:-1}" \
  --learning_rate "${ORPO_LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${MCM_GRAD_ACCUM:-1}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}" \
  --preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${ORPO_LORA_DROPOUT}" \
  --resize_token_embeddings true \
  --train_embeddings "${MCM_PERSONA_ORPO_TRAIN_EMBEDDINGS:-${MCM_TRAIN_EMBEDDINGS:-true}}" \
  --beta 0.1 \
  --max_prompt_tokens "${PERSONA_ORPO_MAX_PROMPT_TOKENS}" \
  --min_completion_tokens "${PERSONA_ORPO_MIN_COMPLETION_TOKENS}" \
  --gradient_checkpointing true \
  --bf16 true \
  --eval_steps "${EVAL_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --load_best_model_at_end "${LOAD_BEST_MODEL_AT_END}" \
  --early_stopping_patience "${EARLY_STOPPING_PATIENCE}" \
  --early_stopping_threshold "${EARLY_STOPPING_THRESHOLD}" \
  --report_to "${MCM_REPORT_TO:-none}" \
  --run_name "${RUN_SLUG}_orpo"

if [[ "${MCM_RUN_PERSONA_POLICY_OPT:-0}" == "1" ]]; then
  uv run accelerate launch src/marichatmen/train/train_grpo.py \
    --model_name "${MODEL_NAME}" \
    --tokenizer_name "$(tokenizer_for_adapter_or_default "${PERSONA_ORPO_ADAPTER_DIR}")" \
    --resize_token_embeddings true \
    --sft_or_orpo_adapter "${PERSONA_ORPO_ADAPTER_DIR}" \
    --prompts_file "${PERSONA_POLICYOPT_PROMPTS_FILE}" \
    --output_dir "${ARTIFACT_ROOT}/outputs/${RUN_SLUG}_policyopt" \
    --max_prompt_length "${MCM_POLICYOPT_MAX_PROMPT_LENGTH:-128}" \
    --max_completion_length "${MCM_POLICYOPT_MAX_COMPLETION_LENGTH:-32}" \
    --num_generations "${MCM_POLICYOPT_NUM_GENERATIONS:-2}" \
    --max_steps "${MCM_PERSONA_POLICYOPT_MAX_STEPS:-1}" \
    --learning_rate 1e-6 \
    --per_device_train_batch_size "${MCM_POLICYOPT_BATCH_SIZE:-2}" \
    --gradient_accumulation_steps "${MCM_GRAD_ACCUM:-1}" \
    --lora_r 16 \
    --lora_alpha 32 \
    --bf16 true \
    --report_to "${MCM_REPORT_TO:-none}" \
    --run_name "${RUN_SLUG}_policyopt"
fi
