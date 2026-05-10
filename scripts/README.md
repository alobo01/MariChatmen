# Scripts

This directory intentionally contains only stable, public entrypoints. Reusable
implementation belongs in `src/marichatmen/`.

## Supported Scripts

- `check_env.py`: validate the Python environment.
- `check_gpu.py`: print local CUDA/GPU availability.
- `download_datasets.py`: download public source datasets into the artifact root.
- `build_data.sh`: build local CPT/SFT/ORPO data under `.artifacts/` or
  `MCM_ARTIFACT_ROOT`.
- `train_qwen_andaluh_local.sh`: run the neutral Qwen-Andaluh local workflow.
- `train_marichatmen_local.sh`: run the persona workflow after explicitly
  enabling persona training.
- `run_tests.sh`: compile and run the test suite.

Prefer the unified CLI for normal use:

```bash
marichatmen build-data --config configs/smoke/qwen35_08b_smoke.yaml
marichatmen train qwen-andaluh --config configs/smoke/qwen35_08b_sft_smoke.yaml
marichatmen eval qwen-gate --samples .artifacts/reports/samples.jsonl
```

Cluster launchers, checkpoint comparison helpers, watchers, and one-off
experiment scripts are local-only and ignored by git.
