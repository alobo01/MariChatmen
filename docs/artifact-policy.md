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
