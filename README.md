# MariChatmen

MariChatmen is a reproducible research pipeline for Andaluh EPA post-training,
created by Antonio Lobo-Santos.

The project builds a neutral Qwen-Andaluh assistant first, then optionally
trains the fictional MariChatmen persona from that base. The repository is
structured for public research use: source code, configs, documentation, tests,
small fixtures, and prompt examples live in git; generated datasets, adapters,
reports, logs, and model outputs stay in `.artifacts/` or external artifact
stores such as Hugging Face.

## Quickstart

```bash
uv python install 3.12
uv sync --python 3.12 --extra train --extra eval --extra dev
bash scripts/run_tests.sh
```

Build data and run the smoke training workflow:

```bash
marichatmen build-data --config configs/smoke/qwen35_08b_smoke.yaml
marichatmen train qwen-andaluh --config configs/smoke/qwen35_08b_sft_smoke.yaml
```

By default, generated artifacts go to `.artifacts/`. Set `MCM_ARTIFACT_ROOT`
to write checkpoints, processed data, and reports somewhere else.

## Published Artifacts

- Qwen-Andaluh 0.8B LoRA:
  <https://huggingface.co/MariChatmen/qwen-andaluh-0.8b-lora>
- MariChatmen 0.8B LoRA:
  <https://huggingface.co/MariChatmen/marichatmen-0.8b-lora>
- MariChatmen 2B Experimental LoRA:
  <https://huggingface.co/MariChatmen/MariChatmen-2B-Experimental>
- MariChatmen 4B Experimental LoRA:
  <https://huggingface.co/MariChatmen/MariChatmen-4B-Experimental>
- MariChatmen Persona dataset:
  <https://huggingface.co/datasets/MariChatmen/MariChatmen-Persona>
- MariChatmen Project Data:
  <https://huggingface.co/datasets/MariChatmen/MariChatmen-Project-Data>
- MariChatmen demo Space:
  <https://huggingface.co/spaces/MariChatmen/demo>

## Repository Layout

```text
src/marichatmen/       installable package
configs/              smoke, local, quality, and release run configs
docs/                 architecture, data, training, eval, and artifact policy
examples/             prompt sets and optional Hugging Face Space demo
scripts/              thin reproducible entrypoints
tests/                unit tests and tiny fixtures
```

## Documentation

- [Architecture](docs/architecture.md)
- [Data and licensing](docs/data.md)
- [Training](docs/training.md)
- [Evaluation](docs/evaluation.md)
- [Artifact policy](docs/artifact-policy.md)

## CLI

The `marichatmen` CLI exposes the supported public workflows:

```bash
marichatmen chat --model_name Qwen/Qwen3.5-0.8B
marichatmen benchmark --model_name ... --run_name smoke --output_jsonl .artifacts/reports/smoke.jsonl
marichatmen eval qwen-gate --samples .artifacts/reports/samples.jsonl
marichatmen train persona --config configs/smoke/qwen35_08b_sft_smoke.yaml --allow-persona
```

Config precedence is: CLI values, then `MCM_*` environment variables, then YAML
config defaults. Use `--set key=value` for any supported `MCM_*` option.

## Author

MariChatmen was designed, built, and released by **Antonio Lobo-Santos**.

The project combines the training pipeline, data builders, evaluation gates,
demo scaffolding, prompt resources, and public release structure needed to
adapt Qwen models toward Andalûh EPA and the MariChatmen persona.

## License

Apache-2.0. See [LICENSE](LICENSE).
