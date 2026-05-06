# MariChatmen

MariChatmen is a reproducible training pipeline for Andaluh EPA post-training.
The project is now staged deliberately:

```text
Qwen-Andaluh first, then MariChatmen.
```

`Qwen-Andaluh` is the non-persona assistant. It uses the neutral system prompt
`Eres un asistente`, reads Spanish or Andaluh, and should answer in Andaluh.
`MariChatmen` is trained later from a Qwen-Andaluh checkpoint after the accent
and Spanish-leak gates pass.

The project builds:

- license-filtered Spanish SFT data from `VillanovaAI/villanova-sft-2603`
- optional Spanish Wikipedia CPT data from the eswiki 2026-05-01 dump
- fragile-span-safe Andaluh EPA conversion
- SFT, ORPO, and optional GRPO QLoRA training entrypoints
- local GPU and timing logs
- MARI-AAS v1 accent/style evaluation plus MARI-PAS persona scoring
- side-by-side base/SFT/ORPO/GRPO stage examples

## Hugging Face Releases

- Qwen-Andaluh 0.8B LoRA:
  <https://huggingface.co/MariChatmen/qwen-andaluh-0.8b-lora>
- MariChatmen 0.8B LoRA:
  <https://huggingface.co/MariChatmen/marichatmen-0.8b-lora>
- MariChatmen Persona dataset:
  <https://huggingface.co/datasets/MariChatmen/MariChatmen-Persona>

## Setup

Install with Python 3.12 managed by `uv`:

```bash
uv python install 3.12
uv sync --python 3.12
```

Check the environment:

```bash
uv run python scripts/check_env.py
uv run python scripts/check_gpu.py
bash scripts/run_tests.sh
```

## Qwen-Andaluh Base Run

This is the non-persona base path. The 0.8B run is only for systems validation.
The next quality target is `Qwen/Qwen3.5-4B-Base` with a Qwen-compatible
expanded tokenizer, longer CPT, longer SFT, and stricter ORPO.

```bash
uv run python scripts/download_datasets.py --only villanova
bash scripts/build_data.sh
bash scripts/train_qwen_andaluh_local.sh
```

### Spanish Wikipedia CPT

The optional Wikipedia CPT source is the Spanish Wikipedia dump dated
2026-05-01:

```text
https://dumps.wikimedia.org/eswiki/20260501/
```

Expected local/Conway location:

```text
/data2/antonio/MariChatmen/data/raw/wikipedia/eswiki/20260501/eswiki-20260501-pages-articles-multistream.xml.bz2
```

Process it with:

```bash
uv run python -m marichatmen.data.build_wikipedia_cpt \
  --dump_file /data2/antonio/MariChatmen/data/raw/wikipedia/eswiki/20260501/eswiki-20260501-pages-articles-multistream.xml.bz2 \
  --out_dir /data2/antonio/MariChatmen/data/processed/cpt_wikipedia_eswiki_20260501 \
  --n_train 100000 \
  --n_valid 5000 \
  --n_probe 1000
```

Wikipedia-derived rows are tracked separately as CC BY-SA 4.0/GFDL text with
article/source metadata. Do not describe them as plain CC BY 4.0.

## MariChatmen Persona Run

After Qwen-Andaluh passes the accent gates, train the fictional persona from
the persona seed file in `data/persona/`. The same seed dataset is published on Hugging Face as
[`MariChatmen/MariChatmen-Persona`](https://huggingface.co/datasets/MariChatmen/MariChatmen-Persona)
under CC BY 4.0.

```bash
bash scripts/train_marichatmen_local.sh
```

## Conway

For Conway, keep code in `/home/antonio/MariChatmen` and all large artifacts in
`/data2/antonio/MariChatmen`. The helper enforces GPU 0 and 12 CPU threads.
By default it now runs the 4B-Base quality path, one model on GPU 0:

```bash
bash scripts/conway_run.sh
```

## Repository Layout

- `src/marichatmen/`: data builders, transliteration, training entrypoints, evaluation, and serving helpers.
- `scripts/`: small reproducible entry points only; implementation stays in `src/`.
- `configs/`: run configuration examples for the model ladder.
- `tests/`: focused unit tests and a tiny public fixture.
- `data/persona/`: synthetic persona seed data used to build persona splits.
- `blog/` and `showcase/`: publication material and reviewed sample outputs.

Generated datasets, reports, adapters, tokenizer experiments, and caches are
ignored by git. Use `MCM_ARTIFACT_ROOT` or stage-specific output variables to
place large artifacts on an external volume.

## Important Release Rules

- Do not publish raw Menuda Noche transcripts or copyrighted TV dialogue.
- Do not release rows from incompatible or unreviewed source licenses.
- Release the LoRA adapter before any merged model.
- Keep both ORPO and GRPO checkpoints if GRPO over-stylizes the model.

## Generated Outputs

These are generated locally and intentionally excluded from version control:

- `data/processed/**`: SFT, ORPO, GRPO, benchmark, and persona split files.
- `outputs/**`: tokenizers, checkpoints, adapters, and merged models.
- `reports/**`: training metrics, GPU logs, plots, samples, and PDF reports.
- `data/raw/**`: raw Hugging Face downloads and source parquet files.
