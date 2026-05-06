# MariChatmen

MariChatmen is a reproducible training pipeline for Andaluh EPA post-training.
The first model is `Qwen-Andaluh`: a non-persona assistant trained with the
system prompt `Eres un asistente`, able to read Spanish or Andaluh and answer
in Andaluh. MariChatmen is then trained later from Qwen-Andaluh as the persona
model.

The project builds:

- license-filtered Spanish SFT data from `VillanovaAI/villanova-sft-2603`
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

This is the non-persona base path. It uses the plain system prompt
`Eres un asistente`, reads Spanish or Andaluh, and always answers in Andaluh.

```bash
uv run python scripts/download_datasets.py --only villanova
bash scripts/build_data.sh
bash scripts/train_qwen_andaluh_local.sh
```

## MariChatmen Persona Run

After Qwen-Andaluh exists, train the fictional persona from the persona seed
file in `data/persona/`. The same seed dataset is published on Hugging Face as
[`MariChatmen/MariChatmen-Persona`](https://huggingface.co/datasets/MariChatmen/MariChatmen-Persona)
under CC BY 4.0.

```bash
bash scripts/train_marichatmen_local.sh
```

## Conway

For Conway, keep code in `/home/antonio/MariChatmen` and all large artifacts in
`/data2/antonio/MariChatmen`. The helper enforces GPU 0 and 12 CPU threads:

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
