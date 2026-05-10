# MariChatmen Data Card

This repository ships code, tiny public fixtures, prompt examples, and links to
the synthetic persona seed data published as
`MariChatmen/MariChatmen-Persona`. Full generated SFT, ORPO, GRPO, benchmark,
tokenizer-analysis datasets, model adapters, and logs are generated locally or
on external artifact storage and intentionally excluded from git.

## Public Persona Data

`MariChatmen/MariChatmen-Persona` contains 12,000 synthetic TRL-style
conversational rows for the fictional MariChatmen persona. The published
persona seed dataset is released under CC BY 4.0. A local copy can be placed at
`data/persona/marichatmen_persona_sft_12000.jsonl`, but that file is ignored in
git.

It is released separately on Hugging Face as:

```text
MariChatmen/MariChatmen-Persona
```

The persona file is used to generate:

- Persona SFT splits;
- Persona ORPO `prompt` / `chosen` / `rejected` preference pairs;
- Persona GRPO prompt-only rows with expected reward features.

## Non-Persona Training Sources

The current smoke/source path uses:

- `VillanovaAI/villanova-sft-2603`
  <https://huggingface.co/datasets/VillanovaAI/villanova-sft-2603>

The next quality Qwen-Andaluh path adds larger non-persona CPT data. Spanish
Wikipedia is being downloaded from the exact Wikimedia dump below:

- Spanish Wikipedia eswiki dump, 2026-05-01:
  <https://dumps.wikimedia.org/eswiki/20260501/>
- Article dump file:
  `eswiki-20260501-pages-articles-multistream.xml.bz2`

Wikipedia text must be cited as Spanish Wikipedia contributors via Wikimedia
Dumps. Treat text reuse as CC BY-SA 4.0/GFDL, not plain CC BY 4.0. Derived CPT
rows must include the dump URL, article title when available, source file, and
the transformation `andaluh_epa_sevillian_ce`. This is the Sevillian-leaning
EPA target that favours `ç` output, for example `Çebiya çabe açêh coçâ bonitâ`.

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

Wikipedia CPT rows are built as plain causal-LM `{"text": ...}` rows with:

- article namespace filtering;
- redirect skipping;
- rough wikitext cleanup;
- word-count filtering;
- 90% Andaluh-transformed text and 10% original Spanish retention by default;
- Spanish and Andaluh validation probes for perplexity tracking.

Protected spans include URLs, emails, code blocks, inline code, shell commands,
package names, model IDs, paths, units, and likely technical proper names.

Default allowed source licences for public data builds:

- Apache-2.0
- MIT

CC BY-SA 4.0/GFDL Wikipedia-derived CPT rows are tracked separately and should
not be mixed into a public non-share-alike dataset release without preserving
attribution and share-alike obligations.

Use `scripts/build_data.sh` to recreate processed base and persona data.
