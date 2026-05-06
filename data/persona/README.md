---
pretty_name: MariChatmen Persona
language:
- es
license: cc-by-4.0
task_categories:
- text-generation
tags:
- synthetic
- instruction-tuning
- andaluh
- andalusian-spanish
- persona
size_categories:
- 10K<n<100K
---

# MariChatmen Persona

Synthetic persona seed data for **MariChatmen**, a fictional Sevillian assistant
that answers in informal Andalûh EPA with a playful, non-hostile Andalusian
persona.

## Files

- `marichatmen_persona_sft_12000.jsonl`: 12,000 TRL-style conversational rows.

Schema:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {
    "category": "technical_explanation",
    "persona_strength": "medium",
    "user_input_style": "standard_spanish",
    "province_flourish": "Málaga"
  }
}
```

## Composition

- 12,000 synthetic conversations.
- User input: 8,400 standard Spanish rows and 3,600 Andalûh rows.
- Assistant output: informal Andalûh EPA with a Sevillian-leaning persona.
- Categories include helpful answers, technical explanations, study support,
  refusals, music/culture, Andalucía comparisons, and province flourishes.

## Transformations

This release is the persona seed file. In the MariChatmen training repo it is
used to build:

- Persona SFT splits by validating `messages` and stripping `<think>` blocks.
- Persona ORPO pairs by using the assistant answer as `chosen` and generating
  controlled `rejected` variants: standard Spanish, too-mild Andalûh,
  caricature, regional hostility, off-persona neutral assistant, or unsafe
  alcohol framing.
- Persona GRPO prompt rows with expected reward features: Andalûh, helpfulness,
  Sevillian voice, MariChatmen persona, province flourish, and non-hostility.

The separate non-persona Qwen-Andalûh pipeline uses Spanish instruction data,
filters by source licence/category/language, masks fragile spans, converts
assistant messages to Andalûh EPA, converts a controlled share of user messages
to Andalûh EPA, then restores URLs, code, model IDs, package names, paths, and
other protected spans unchanged.

## Licence

Released under Creative Commons Attribution 4.0 International (CC BY 4.0).

Attribution suggestion:

```text
MariChatmen Persona dataset by Antonio Lobo, released under CC BY 4.0.
https://huggingface.co/datasets/MariChatmen/MariChatmen-Persona
```

## Notes

- This is synthetic data, not a linguistic authority.
- MariChatmen is fictional and Sevillian-leaning; she does not represent all
  Andalusian speakers or varieties.
- The dataset can contain repetitive phrasing and imperfect Andalûh.
- Alcohol references are adult-coded cultural flavour, not advice to drink.

## Related Sources

- AndaluGeeks EPA / Êttandâ pal Andalûh: <https://andaluh.es/epa-2/>
- `andaluh-py`: <https://github.com/andalugeeks/andaluh-py>
