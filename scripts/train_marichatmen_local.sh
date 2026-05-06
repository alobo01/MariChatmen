#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${MCM_ARTIFACT_ROOT:-$PWD}"
MODEL_NAME="${MCM_MODEL_NAME:-Qwen/Qwen3.5-0.8B}"
BASE_SLUG="${MCM_BASE_RUN_SLUG:-qwen_andaluh_08b}"
RUN_SLUG="${MCM_RUN_SLUG:-marichatmen_08b}"
PERSONA_OUT_DIR="${MCM_PERSONA_OUT_DIR:-${ARTIFACT_ROOT}/data/processed/persona}"
TOKENIZER_DIR="${MCM_TOKENIZER_DIR:-${ARTIFACT_ROOT}/outputs/tokenizers/qwen35_andaluh}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-12}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-${ARTIFACT_ROOT}/.hf_cache}"
export WANDB_PROJECT="${WANDB_PROJECT:-MariChatmen}"

uv run accelerate launch src/marichatmen/train/train_sft.py \
  --model_name "${MODEL_NAME}" \
  --tokenizer_name "${TOKENIZER_DIR}" \
  --base_adapter "${ARTIFACT_ROOT}/outputs/${BASE_SLUG}_sft/final_adapter" \
  --train_file "${PERSONA_OUT_DIR}/persona_sft_train.jsonl" \
  --valid_file "${PERSONA_OUT_DIR}/persona_sft_valid.jsonl" \
  --output_dir "${ARTIFACT_ROOT}/outputs/${RUN_SLUG}_sft" \
  --max_seq_length "${MCM_MAX_SEQ_LENGTH:-256}" \
  --max_steps "${MCM_PERSONA_SFT_MAX_STEPS:-2}" \
  --learning_rate 5e-5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "${MCM_GRAD_ACCUM:-1}" \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --resize_token_embeddings true \
  --train_embeddings true \
  --gradient_checkpointing true \
  --bf16 true \
  --report_to "${MCM_REPORT_TO:-none}" \
  --run_name "${RUN_SLUG}_sft"

uv run accelerate launch src/marichatmen/train/train_orpo.py \
  --model_name "${MODEL_NAME}" \
  --tokenizer_name "${TOKENIZER_DIR}" \
  --sft_adapter "${ARTIFACT_ROOT}/outputs/${RUN_SLUG}_sft/final_adapter" \
  --train_file "${PERSONA_OUT_DIR}/persona_orpo_train.jsonl" \
  --valid_file "${PERSONA_OUT_DIR}/persona_orpo_valid.jsonl" \
  --output_dir "${ARTIFACT_ROOT}/outputs/${RUN_SLUG}_orpo" \
  --max_seq_length "${MCM_MAX_SEQ_LENGTH:-256}" \
  --max_steps "${MCM_PERSONA_ORPO_MAX_STEPS:-1}" \
  --learning_rate 5e-6 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "${MCM_GRAD_ACCUM:-1}" \
  --lora_r 16 \
  --lora_alpha 32 \
  --resize_token_embeddings true \
  --train_embeddings true \
  --beta 0.1 \
  --gradient_checkpointing true \
  --bf16 true \
  --report_to "${MCM_REPORT_TO:-none}" \
  --run_name "${RUN_SLUG}_orpo"

uv run accelerate launch src/marichatmen/train/train_grpo.py \
  --model_name "${MODEL_NAME}" \
  --tokenizer_name "${TOKENIZER_DIR}" \
  --resize_token_embeddings true \
  --sft_or_orpo_adapter "${ARTIFACT_ROOT}/outputs/${RUN_SLUG}_orpo/final_adapter" \
  --prompts_file "${PERSONA_OUT_DIR}/persona_grpo_prompts.jsonl" \
  --output_dir "${ARTIFACT_ROOT}/outputs/${RUN_SLUG}_grpo" \
  --max_prompt_length "${MCM_GRPO_MAX_PROMPT_LENGTH:-128}" \
  --max_completion_length "${MCM_GRPO_MAX_COMPLETION_LENGTH:-32}" \
  --num_generations "${MCM_GRPO_NUM_GENERATIONS:-2}" \
  --max_steps "${MCM_PERSONA_GRPO_MAX_STEPS:-1}" \
  --learning_rate 1e-6 \
  --per_device_train_batch_size "${MCM_GRPO_BATCH_SIZE:-2}" \
  --gradient_accumulation_steps "${MCM_GRAD_ACCUM:-1}" \
  --lora_r 16 \
  --lora_alpha 32 \
  --bf16 true \
  --report_to "${MCM_REPORT_TO:-none}" \
  --run_name "${RUN_SLUG}_grpo"
