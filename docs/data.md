# Data And Licensing

This repository tracks code, tiny fixtures, prompt examples, and documentation.
Full generated datasets are produced locally or released through Hugging Face.

## Public Dataset

The synthetic persona seed dataset is published as:

```text
MariChatmen/MariChatmen-Persona
```

It is not tracked in git as a large JSONL file. Use
`data/persona/README.md` and the Hugging Face dataset card as the public source
of truth.

## Source Rules

- Default public non-persona builds use Apache-2.0 and MIT rows.
- Wikipedia CPT rows must remain separate and retain CC BY-SA 4.0/GFDL
  attribution.
- Do not publish raw Menuda Noche transcripts or copyrighted TV dialogue.
- Protected spans include URLs, emails, code, commands, paths, model IDs,
  package names, units, and technical proper names.

## Generated Data

`scripts/build_data.sh` writes to `.artifacts/data/processed` by default. Set
`MCM_ARTIFACT_ROOT` for external storage.
