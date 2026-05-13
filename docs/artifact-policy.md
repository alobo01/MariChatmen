# Artifact Policy

The repository should stay small and reproducible.

## Tracked

- Source code under `src/`
- Tests and tiny fixtures
- Config examples
- Prompt examples under `examples/prompts/`
- Documentation
- Optional demo scaffolding under `examples/hf-space/`

## Not Tracked

- Raw downloaded datasets
- Generated CPT/SFT/ORPO/GRPO data
- Tokenizer experiments
- Checkpoints, LoRA adapters, and merged models
- Training reports, plots, GPU logs, and generated showcase files
- Local caches and virtual environments

By default, local workflows write under `.artifacts/`. Use
`MCM_ARTIFACT_ROOT=/path/to/artifacts` for external disks or cluster storage.

## Published Artifact Stores

Current public Hugging Face artifacts live under the `MariChatmen`
organisation:

- `MariChatmen/qwen-andaluh-0.8b-lora`
- `MariChatmen/marichatmen-0.8b-lora`
- `MariChatmen/MariChatmen-2B-Experimental`
- `MariChatmen/MariChatmen-4B-Experimental`
- `MariChatmen/MariChatmen-Persona`
- `MariChatmen/MariChatmen-Project-Data`
- `MariChatmen/demo`

Do not commit recovered adapters or generated JSONL data to git. Stage large
publishable files under `.artifacts/hf_upload/` and upload them to Hugging
Face instead.
