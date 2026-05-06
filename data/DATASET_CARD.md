# MariChatmen Data Card

This repository ships code, tiny public fixtures, and the synthetic persona seed
data that is also published as `MariChatmen/MariChatmen-Persona`. Full generated SFT,
ORPO, GRPO, benchmark, tokenizer-analysis datasets, model adapters, and logs are
generated locally or on Conway and intentionally excluded from git.

## Public Persona Data

`data/persona/marichatmen_persona_sft_12000.jsonl` contains 12,000 synthetic
TRL-style conversational rows for the fictional MariChatmen persona. The
published persona seed dataset is released under CC BY 4.0.

It is released separately on Hugging Face as:

```text
MariChatmen/MariChatmen-Persona
```

The persona file is used to generate:

- Persona SFT splits;
- Persona ORPO `prompt` / `chosen` / `rejected` preference pairs;
- Persona GRPO prompt-only rows with expected reward features.

## Non-Persona Training Sources

The primary public SFT source for Qwen-Andalûh is:

- `VillanovaAI/villanova-sft-2603`
  <https://huggingface.co/datasets/VillanovaAI/villanova-sft-2603>

Additional Spanish sources evaluated for future or optional builds:

- `Iker/OpenHermes-2.5-Spanish`
- `CohereLabs/aya_dataset`
- `projecte-aina/MentorES`
- `bertin-project/alpaca-spanish`
- `hlhdatscience/guanaco-spanish-dataset`
- `somosnlp/SMC-instruct`

Evaluation-only datasets, not for training:

- `BSC-LT/IFEval_es`
- `BSC-LT/EsBBQ`

## Default Transformations

Derived non-persona rows are built from Spanish chat-style source data with:

- Spanish language filtering;
- `Chat`, `Reasoning`, and `Safety` category filtering;
- Apache-2.0/MIT-compatible source rows by default;
- `<think>` block stripping;
- fragile-span masking before transliteration;
- assistant messages converted to Andalûh EPA 100% of the time;
- user messages converted to Andalûh EPA with a controlled ratio;
- protected spans restored unchanged after transliteration;
- SFT rows written in TRL conversational `messages` format;
- ORPO rows written as conversational `prompt`, `chosen`, and `rejected`.

Protected spans include URLs, emails, code blocks, inline code, shell commands,
package names, model IDs, paths, units, and likely technical proper names.

Default allowed source licences for public data builds:

- Apache-2.0
- MIT

Use `scripts/build_data.sh` to recreate processed base and persona data.
