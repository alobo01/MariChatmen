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

For the optional Spanish Wikipedia CPT source, copy the eswiki 2026-05-01 dump
to the artifact root and run the module directly:

```bash
uv run python -m marichatmen.data.build_wikipedia_cpt \
  --dump_file /data2/antonio/MariChatmen/data/raw/wikipedia/eswiki/20260501/eswiki-20260501-pages-articles-multistream.xml.bz2 \
  --out_dir /data2/antonio/MariChatmen/data/processed/cpt_wikipedia_eswiki_20260501
```

The Wikipedia rows are tracked as CC BY-SA 4.0/GFDL and should stay in a
separate manifest-backed CPT dataset.

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
large files. The default quality target is now `Qwen/Qwen3.5-4B-Base`; set
`MCM_MODELS="08b 2b"` only for smoke/baseline debugging.
