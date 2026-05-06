# Scripts

Keep this folder small. The real implementation lives in `src/marichatmen/`;
these files are only reproducible entry points.

## Local Checks

```bash
uv run python scripts/check_env.py
uv run python scripts/check_gpu.py
bash scripts/run_tests.sh
```

## Data

Download only Villanova by default:

```bash
uv run python scripts/download_datasets.py --only villanova
bash scripts/build_data.sh
```

`build_data.sh` writes base Qwen-Andaluh data and, if
`data/persona/marichatmen_persona_sft_12000.jsonl` exists, persona splits.

## Local Training

```bash
bash scripts/train_qwen_andaluh_local.sh
bash scripts/train_marichatmen_local.sh
```

Both scripts default to Qwen3.5-0.8B and can be retargeted with environment
variables such as `MCM_MODEL_NAME`, `MCM_RUN_SLUG`, `MCM_ARTIFACT_ROOT`, and
`MCM_MAX_STEPS`.

## Conway

```bash
bash scripts/conway_run.sh
```

This uses GPU 0, at most 12 CPU threads, and `/data2/antonio/MariChatmen` for
large files.
