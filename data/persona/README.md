---
pretty_name: MariChatmen Persona
language:
- es
license: apache-2.0
task_categories:
- text-generation
tags:
- synthetic
- instruction-tuning
- preference-optimization
- grpo
- orpo
- sft
- andaluh
- andalusian-spanish
- persona
- qwen
size_categories:
- 10K<n<100K
---

# MariChatmen Persona

`alobos/MariChatmen-Persona` is a synthetic persona seed dataset for training
**MariChatmen**, a fictional Sevillian assistant that answers in informal
Andalûh EPA with a playful but non-hostile Andalusian persona.

This release contains only the persona seed file. It does **not** include raw
Villanova, OpenHermes, Aya, MentorES, Alpaca, Guanaco, SMC-instruct, benchmark
datasets, trained adapters, generated model outputs, or private material.

## Files

- `marichatmen_persona_sft_12000.jsonl`: 12,000 TRL-style conversational SFT
  rows.

Each row has:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {
    "dataset_stage": "persona_sft_marichatmen",
    "category": "technical_explanation",
    "persona_strength": "medium",
    "user_input_style": "standard_spanish",
    "province_flourish": "Málaga",
    "contains_cruzcampo": false,
    "contains_music_reference": false,
    "contains_feria_reference": false
  }
}
```

## Dataset Composition

- Rows: 12,000
- Format: JSONL, one conversation per line
- System prompt: persona-defining MariChatmen prompt
- User input: standard Spanish or Andalûh
- Assistant output: informal Andalûh EPA with Sevillian-leaning flavour

Category counts:

| Category | Rows |
| --- | ---: |
| `normal_helpful_answers` | 3,000 |
| `andalucia_comparison_or_pride` | 1,800 |
| `technical_explanation` | 1,800 |
| `feria_triana_macarena_expo_cruzcampo` | 1,440 |
| `study_or_emotional_support` | 1,200 |
| `province_flourish` | 1,200 |
| `music_culture` | 960 |
| `refusal_or_boundary` | 600 |

Persona strength:

| Strength | Rows |
| --- | ---: |
| `low` | 5,400 |
| `medium` | 4,800 |
| `high` | 1,800 |

User input style:

| Style | Rows |
| --- | ---: |
| `standard_spanish` | 8,400 |
| `andaluh` | 3,600 |

Province flourish balance:

| Value | Rows |
| --- | ---: |
| `none` | 5,400 |
| each Andalusian province | 825 each |

Other marker counts:

| Marker | Rows |
| --- | ---: |
| `contains_feria_reference` | 2,221 |
| `contains_cruzcampo` | 921 |
| `contains_music_reference` | 913 |

## Transformations And Downstream Use

This persona release is used by the MariChatmen repository as the source file
for three downstream training artefacts:

1. **Persona SFT**: messages are validated, `<think>` blocks are stripped, and
   rows are split into train/validation conversational JSONL files.
2. **Persona ORPO**: each row is converted into a `prompt`, `chosen`,
   `rejected` preference example. Rejected answers are generated from controlled
   failure families: standard Spanish, too-mild Andalûh, caricature, regional
   hostility, off-persona neutral assistant, and unsafe alcohol framing.
3. **Persona GRPO**: prompt-only rows are extracted with expected reward
   features: Andalûh, helpfulness, Sevillian voice, MariChatmen persona,
   province flourish, and non-hostility.

The broader Qwen-Andalûh non-persona pipeline, which is not included in this
dataset release, applies these transformations to Spanish instruction data:

- filter to Spanish rows;
- keep `Chat`, `Reasoning`, and `Safety` categories;
- keep Apache-2.0/MIT-compatible source rows by default;
- strip `<think>` blocks;
- mask fragile spans before transliteration;
- convert assistant messages to Andalûh EPA 100% of the time;
- convert user messages to Andalûh EPA with a controlled ratio;
- restore protected spans unchanged;
- store SFT rows in TRL conversational `messages` format;
- store ORPO rows as conversational `prompt`, `chosen`, and `rejected`.

Protected fragile spans include URLs, emails, code blocks, inline code, shell
commands, package names, model IDs, paths, units, and likely technical proper
names. This prevents transformations from corrupting strings such as
`Qwen/Qwen3.5-0.8B`, `/data2/antonio/MariChatmen`, `uv run`, or
`https://huggingface.co`.

## Intended Use

Use this dataset for experiments in:

- persona SFT;
- persona preference optimisation with ORPO/DPO-style methods;
- custom-reward GRPO prompts;
- evaluation of dialect/persona trade-offs;
- synthetic data quality analysis.

It is intended to be mixed after a non-persona Qwen-Andalûh stage. The intended
training order is:

```text
Qwen base
→ Andalûh adaptive pretraining / CPT
→ Accent SFT
→ Accent ORPO
→ Persona SFT using this dataset
→ Persona ORPO
→ Persona GRPO
```

## Limitations

- Synthetic data can contain repetitive phrasings, uneven style strength, and
  imperfect Andalûh.
- The persona is fictional and Sevillian-leaning. It does not represent all
  Andalusian speakers or all Andalusian varieties.
- The dataset is not an official linguistic standard.
- It may overrepresent explicit cultural markers such as Feria, Triana,
  gazpacho, province flourishes, and music references.
- Alcohol references are intended as adult-coded cultural flavour only and are
  not instructions to drink or to use alcohol as a coping mechanism.

## Licence

Released under Apache-2.0.

## Citations And Related Sources

This persona dataset is synthetic, but the project and training pipeline cite
the following sources:

- AndaluGeeks EPA / Êttandâ pal Andalûh:
  <https://andaluh.es/epa-2/>
- `andaluh` / `andaluh-py`, used for Spanish-to-Andalûh EPA transliteration in
  the broader pipeline:
  <https://pypi.org/project/andaluh/>
  and <https://github.com/andalugeeks/andaluh-py>
- VillanovaAI `villanova-sft-2603`, used as the primary public non-persona SFT
  source in the broader Qwen-Andalûh pipeline, not included in this release:
  <https://huggingface.co/datasets/VillanovaAI/villanova-sft-2603>
- Iker `OpenHermes-2.5-Spanish`, evaluated as an additional Spanish
  instruction source for future non-persona data builds:
  <https://huggingface.co/datasets/Iker/OpenHermes-2.5-Spanish>
- CohereLabs `aya_dataset`, evaluated as an additional multilingual Spanish
  source:
  <https://huggingface.co/datasets/CohereLabs/aya_dataset>
- projecte-aina `MentorES`, evaluated as an additional Spanish instruction
  source:
  <https://huggingface.co/datasets/projecte-aina/MentorES>
- bertin-project `alpaca-spanish`, evaluated as an additional Spanish
  instruction source:
  <https://huggingface.co/datasets/bertin-project/alpaca-spanish>
- hlhdatscience `guanaco-spanish-dataset`, evaluated as an additional Spanish
  conversational source:
  <https://huggingface.co/datasets/hlhdatscience/guanaco-spanish-dataset>
- somosnlp `SMC-instruct`, optional Spanish/English medical instruction source
  requiring source-level licence care:
  <https://huggingface.co/datasets/somosnlp/SMC-instruct>

Training/evaluation method references used by the project:

- TRL SFT Trainer: <https://huggingface.co/docs/trl/main/sft_trainer>
- TRL GRPO Trainer: <https://huggingface.co/docs/trl/grpo_trainer>
- ORPO paper: <https://arxiv.org/abs/2403.07691>
- DeepSeekMath / GRPO paper: <https://arxiv.org/abs/2402.03300>
- LoRA paper: <https://arxiv.org/abs/2106.09685>
- QLoRA paper: <https://arxiv.org/abs/2305.14314>
