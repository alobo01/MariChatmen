#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="${MCM_CODE_DIR:-/home/antonio/MariChatmen}"
DATA_ROOT="${MCM_ARTIFACT_ROOT:-/data2/antonio/MariChatmen}"
MODELS="${MCM_MODELS:-4b}"

cd "${CODE_DIR}"
mkdir -p "${DATA_ROOT}/run_locks" "${DATA_ROOT}/reports/logs"

exec 9>"${DATA_ROOT}/run_locks/gpu0.lock"
if ! flock -n 9; then
  echo "GPU0 is already reserved by another MariChatmen run."
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-12}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-12}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-12}"
export MCM_ARTIFACT_ROOT="${DATA_ROOT}"
export HF_HOME="${HF_HOME:-${DATA_ROOT}/.hf_cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${DATA_ROOT}/.uv_cache}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${DATA_ROOT}/.venv}"

uv sync
uv run python scripts/check_env.py
uv run python scripts/check_gpu.py

MCM_CPT_TRAIN="${MCM_CPT_TRAIN:-20000}" \
MCM_CPT_VALID="${MCM_CPT_VALID:-1000}" \
MCM_SFT_TRAIN="${MCM_SFT_TRAIN:-20000}" \
MCM_SFT_VALID="${MCM_SFT_VALID:-1000}" \
MCM_ORPO_TRAIN="${MCM_ORPO_TRAIN:-10000}" \
MCM_ORPO_VALID="${MCM_ORPO_VALID:-1000}" \
bash scripts/build_data.sh

for size in ${MODELS}; do
  case "${size}" in
    08b) model="Qwen/Qwen3.5-0.8B-Base"; slug="qwen_andaluh_08b"; grad=16; lora_r=16; lora_alpha=32 ;;
    2b) model="Qwen/Qwen3.5-2B-Base"; slug="qwen_andaluh_2b"; grad=24; lora_r=16; lora_alpha=32 ;;
    4b) model="Qwen/Qwen3.5-4B-Base"; slug="qwen_andaluh_4b"; grad=32; lora_r=32; lora_alpha=64 ;;
    *) echo "Unknown model size: ${size}"; exit 1 ;;
  esac

  MCM_MODEL_NAME="${model}" \
  MCM_RUN_SLUG="${slug}" \
  MCM_GRAD_ACCUM="${grad}" \
  MCM_MAX_SEQ_LENGTH="${MCM_MAX_SEQ_LENGTH:-1024}" \
  MCM_LORA_R="${lora_r}" \
  MCM_LORA_ALPHA="${lora_alpha}" \
  MCM_CPT_MAX_STEPS="${MCM_CPT_MAX_STEPS:-3000}" \
  MCM_SFT_MAX_STEPS="${MCM_SFT_MAX_STEPS:-3000}" \
  MCM_ORPO_MAX_STEPS="${MCM_ORPO_MAX_STEPS:-1000}" \
  MCM_CPT_LR="${MCM_CPT_LR:-5e-5}" \
  MCM_SFT_LR="${MCM_SFT_LR:-5e-5}" \
  MCM_ORPO_LR="${MCM_ORPO_LR:-3e-6}" \
  MCM_ORPO_BETA="${MCM_ORPO_BETA:-0.1}" \
  bash scripts/train_qwen_andaluh_local.sh
done
