#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${MCM_ARTIFACT_ROOT:-${PWD}/.artifacts}"
DATASET="${MCM_DATASET:-${ARTIFACT_ROOT}/data/raw/hf/villanova/data/*.parquet}"
BASE_OUT_DIR="${MCM_BASE_OUT_DIR:-${ARTIFACT_ROOT}/data/processed/base}"
PERSONA_FILE="${MCM_PERSONA_FILE:-data/persona/marichatmen_persona_sft_12000.jsonl}"
PERSONA_OUT_DIR="${MCM_PERSONA_OUT_DIR:-${ARTIFACT_ROOT}/data/processed/persona}"
SYSTEM_PROMPT="${MCM_SYSTEM_PROMPT:-Eres un asistente}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-12}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "${BASE_OUT_DIR}" "${ARTIFACT_ROOT}/reports/tokenizer" "${ARTIFACT_ROOT}/reports/plots"

if [[ "${MCM_ANALYZE_TOKENIZER:-1}" == "1" ]]; then
  uv run python -m marichatmen.data.tokenizer_impact \
    --model_name "${MCM_MODEL_NAME:-Qwen/Qwen3.5-0.8B}" \
    --dataset "${DATASET}" \
    --language spa \
    --allowed_licenses apache-2.0 MIT \
    --n_texts "${MCM_TOKENIZER_TEXTS:-2000}" \
    --new_tokens "${MCM_NEW_TOKENS:-1536}" \
    --mode "${MCM_TOKENIZER_MODE:-expand}" \
    --retrain_vocab_size "${MCM_RETRAIN_VOCAB_SIZE:-0}" \
    --variant sevillian_ce \
    --informal_strength "${MCM_BASE_INFORMAL_STRENGTH:-0.0}" \
    --save_tokenizer_dir "${MCM_TOKENIZER_DIR:-${ARTIFACT_ROOT}/outputs/tokenizers/qwen35_andaluh}" \
    --output_json "${ARTIFACT_ROOT}/reports/tokenizer/qwen_andaluh_tokenizer_impact.json" \
    --output_csv "${ARTIFACT_ROOT}/reports/tokenizer/qwen_andaluh_tokenizer_impact.csv" \
    --plot_svg "${ARTIFACT_ROOT}/reports/plots/qwen_andaluh_tokenizer_impact.svg" \
    --output_md "${ARTIFACT_ROOT}/reports/tokenizer/qwen_andaluh_tokenizer_impact.md"
fi

uv run python -m marichatmen.data.build_cpt \
  --dataset "${DATASET}" \
  --language spa \
  --allowed_licenses apache-2.0 MIT \
  --n_train "${MCM_CPT_TRAIN:-2000}" \
  --n_valid "${MCM_CPT_VALID:-200}" \
  --n_probe "${MCM_CPT_PROBE:-128}" \
  --andaluh_ratio "${MCM_CPT_ANDALUH_RATIO:-0.9}" \
  --user_andaluh_ratio "${MCM_CPT_USER_ANDALUH_RATIO:-0.3}" \
  --variant sevillian_ce \
  --informal_strength "${MCM_BASE_INFORMAL_STRENGTH:-0.0}" \
  --system_prompt "${SYSTEM_PROMPT}" \
  --fallback_fixture tests/fixtures/spanish_sft_fixture.jsonl \
  --out_dir "${BASE_OUT_DIR}"

uv run python -m marichatmen.data.build_sft \
  --dataset "${DATASET}" \
  --language spa \
  --allowed_licenses apache-2.0 MIT \
  --n_train "${MCM_SFT_TRAIN:-2000}" \
  --n_valid "${MCM_SFT_VALID:-200}" \
  --eval_prompts "${MCM_EVAL_PROMPTS:-80}" \
  --user_andaluh_ratio "${MCM_SFT_USER_ANDALUH_RATIO:-0.3}" \
  --assistant_andaluh_ratio 1.0 \
  --variant sevillian_ce \
  --informal_strength "${MCM_BASE_INFORMAL_STRENGTH:-0.0}" \
  --system_prompt "${SYSTEM_PROMPT}" \
  --fallback_fixture tests/fixtures/spanish_sft_fixture.jsonl \
  --out_dir "${BASE_OUT_DIR}"

TECH_GOLD_DIR="${MCM_TECHNICAL_GOLD_DIR:-${BASE_OUT_DIR}/technical_gold_sft}"
uv run python -m marichatmen.data.build_technical_gold \
  --out_dir "${TECH_GOLD_DIR}" \
  --n_train "${MCM_TECHNICAL_GOLD_TRAIN:-300}" \
  --n_valid "${MCM_TECHNICAL_GOLD_VALID:-60}"

SFT_RAW_DIR="${BASE_OUT_DIR}/_raw_sft_before_regularisation"
mkdir -p "${SFT_RAW_DIR}"
cp "${BASE_OUT_DIR}/sft_train.jsonl" "${SFT_RAW_DIR}/sft_train.jsonl"
cp "${BASE_OUT_DIR}/sft_valid.jsonl" "${SFT_RAW_DIR}/sft_valid.jsonl"
uv run python -m marichatmen.data.regularize_sft \
  --sft_train "${SFT_RAW_DIR}/sft_train.jsonl" \
  --sft_valid "${SFT_RAW_DIR}/sft_valid.jsonl" \
  --technical_train "${TECH_GOLD_DIR}/train.jsonl" \
  --technical_valid "${TECH_GOLD_DIR}/valid.jsonl" \
  --target_user_andaluh_ratio "${MCM_SFT_USER_ANDALUH_RATIO:-0.3}" \
  --variant sevillian_ce \
  --user_informal_strength "${MCM_BASE_INFORMAL_STRENGTH:-0.0}" \
  --out_dir "${BASE_OUT_DIR}"

uv run python -m marichatmen.data.build_orpo \
  --sft_train "${BASE_OUT_DIR}/sft_train.jsonl" \
  --n_train "${MCM_ORPO_TRAIN:-1000}" \
  --n_valid "${MCM_ORPO_VALID:-200}" \
  --rejected_types \
    standard_spanish_leak \
    weak_andaluh \
    reasoning_preamble \
    correct_style_wrong_answer \
    repetition_collapse \
    overtranscribed_garble \
    too_long_rambling \
  --enforce_length_ratio \
  --system_prompt "${SYSTEM_PROMPT}" \
  --out_dir "${BASE_OUT_DIR}"

uv run python -m marichatmen.data.build_grpo_prompts \
  --sft_train "${BASE_OUT_DIR}/sft_train.jsonl" \
  --n_prompts "${MCM_GRPO_PROMPTS:-200}" \
  --out_file "${BASE_OUT_DIR}/grpo_prompts.jsonl"

uv run python -m marichatmen.data.build_repair_rl_prompts \
  --sft_train "${BASE_OUT_DIR}/sft_train.jsonl" \
  --probe_file "${MCM_QWEN_PROBE_FILE:-examples/prompts/qwen_andaluh_neutral_probe_v1.txt}" \
  --n_prompts "${MCM_REPAIR_RL_PROMPTS:-300}" \
  --out_file "${BASE_OUT_DIR}/qwen_andaluh_repair_rl_prompts.jsonl"

if [[ -f "${PERSONA_FILE}" ]]; then
  uv run python -m marichatmen.data.build_persona_from_file \
    --persona_sft_file "${PERSONA_FILE}" \
    --out_dir "${PERSONA_OUT_DIR}" \
    --n_sft_train "${MCM_PERSONA_SFT_TRAIN:-256}" \
    --n_sft_valid "${MCM_PERSONA_SFT_VALID:-32}" \
    --n_orpo_train "${MCM_PERSONA_ORPO_TRAIN:-128}" \
    --n_orpo_valid "${MCM_PERSONA_ORPO_VALID:-32}" \
    --n_grpo "${MCM_PERSONA_GRPO:-64}" \
    --min_sft_reward "${MCM_PERSONA_MIN_SFT_REWARD:-0.35}" \
    --min_orpo_margin "${MCM_PERSONA_MIN_ORPO_MARGIN:-0.12}" \
    --max_scored_rows "${MCM_PERSONA_MAX_SCORED_ROWS:-0}"
fi
