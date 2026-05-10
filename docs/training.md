# Training

The project is staged deliberately:

```text
Qwen-Andaluh first, then MariChatmen.
```

Qwen-Andaluh is the neutral assistant that answers in Andaluh EPA. MariChatmen
is a later fictional persona layer trained only after the neutral accent and
leakage gates pass.

## Workflows

```bash
marichatmen build-data --config configs/smoke/qwen35_08b_smoke.yaml
marichatmen train qwen-andaluh --config configs/smoke/qwen35_08b_sft_smoke.yaml
marichatmen train persona --config configs/smoke/qwen35_08b_sft_smoke.yaml --allow-persona
```

The CLI maps YAML keys to `MCM_*` environment variables. CLI `--set key=value`
overrides both environment variables and config values.

## Invariants

- Keep adapter-owned tokenizer files with every expanded-tokenizer LoRA.
- Do not silently load an adapter with a different tokenizer.
- Treat 0.8B runs as smoke/system validation, not release quality.
- Use 4B or larger models for quality claims.
- Persona training must be explicitly enabled with `--allow-persona` or
  `MCM_ALLOW_PERSONA_TRAINING=1`.
