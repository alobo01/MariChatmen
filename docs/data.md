# Data And Licensing

This repository tracks code, tiny fixtures, prompt examples, and documentation.
Full generated datasets are produced locally or released through Hugging Face.

## Public Datasets

The synthetic persona seed dataset is published as:

```text
MariChatmen/MariChatmen-Persona
```

Curated project-authored experiment data is published as:

```text
MariChatmen/MariChatmen-Project-Data
```

Large JSONL files are not tracked in git. Use `data/persona/README.md`, this
document, and the Hugging Face dataset cards as the public source of truth.
The project-data release intentionally excludes external downloaded rows where
the main project contribution was filtering, protected-span handling, and
Andaluh rendering.

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
