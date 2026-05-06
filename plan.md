Yes — that is the right simplification.

For **non-personality data**, do **not** spend LLM budget generating Andalûh answers from scratch. Use this pipeline instead:

```text
Spanish source answer
→ andaluh-py EPA transliteration
→ Sevillian/informal postprocessing
→ automatic QA
→ optional human/LLM review
```

Reserve LLM generation for the things that **andaluh-py cannot create**:

```text
MariChatmen persona
humour
province flourishes
Cruzcampo/caseta/Feria/SFDK/ToteKing references
ORPO rejected answers
GRPO prompt design
quality verification
```

Your uploaded report already shows the reason: the pipeline is operational, but current outputs are still smoke-test quality; MARI-AAS is moderate, MARI-PAS is very low, province flourish is zero, and manual samples show semantic drift/persona keyword soup. So the next version should be data-engineering-first, not “more RL immediately.” 

# MariChatmen integrated plan

## 1. Final pipeline

Use this full staged pipeline:

```text
0. Source acquisition and licence audit
1. Tokenization audit: Spanish vs Andalûh
2. AAP: Andalûh Adaptive Pretraining
3. Accent SFT: translated instruction-following data
4. Persona SFT: generated + verified MariChatmen examples
5. ORPO: preference pairs against bad variants
6. GRPO: final persona/accent RL polishing
7. Evaluation, reporting, HF/GitHub/blog release
```

The split should be:

| Stage                  | Main creation method                         | LLM generation? |
| ---------------------- | -------------------------------------------- | --------------: |
| AAP plain text         | Spanish corpus → `andaluh-py`                |              No |
| Accent SFT             | Spanish SFT → `andaluh-py`                   |       Mostly no |
| Persona SFT            | LLM + verification                           |             Yes |
| ORPO                   | LLM creates rejected variants + verification |             Yes |
| GRPO prompts           | LLM creates prompts only                     |             Yes |
| Evaluation benchmark   | LLM + manual review                          |             Yes |
| Metrics/reward scoring | Python scripts                               |              No |

`andaluh-py` is especially useful because its `epa()` function applies ordered transformation rules including `h_rules`, `x_rules`, `ch_rules`, `gj_rules`, `v_rules`, `ll_rules`, `l_rules`, `vaf_rules`, `word_ending_rules`, `digraph_rules`, `exception_rules`, and `word_interaction_rules`. It also supports `escape_links`, which you should use to avoid corrupting URLs/emails/mentions during transliteration. ([GitHub][1])

---

# 2. Dataset stack

## Dataset A — `aap_andaluh_plaintext`

**Purpose:** make Andalûh-looking text natural to the base model before instruction tuning.

Use Spanish plain text from:

```text
FineWeb2 Spanish subset
mC4 Spanish sample
optional CulturaX Spanish
```

FineWeb2 is a multilingual Common Crawl-derived pretraining dataset intended for LLM pretraining and processed/deduplicated/filtered at scale. ([Hugging Face][2])

Pipeline:

```text
Spanish raw text
→ clean/filter
→ andaluh-py epa(vaf="ç", vvf="h", escape_links=True)
→ Sevillian informal postprocessing
→ pack into causal-LM text chunks
```

Recommended size for an 8B/9B-ish model:

```text
Minimum: 50M Andalûh tokens
Good:    100M–150M Andalûh tokens
Max first run: 200M tokens
```

Output schema:

```json
{"text": "...Andalûh plain text...", "metadata": {"source": "...", "stage": "aap", "transformation": "andaluh_py_epa_seseo"}}
```

---

## Dataset B — `accent_sft_translated`

**Purpose:** teach the model to answer instructions in Andalûh, without heavy personality.

Use Spanish instruction datasets such as OpenHermes Spanish and Villanova Spanish rows where available. OpenHermes Spanish is listed on Hugging Face as Spanish, Apache-2.0, and around 1M rows. ([Hugging Face][3])

Creation method:

```text
Spanish instruction conversation
→ keep user 50% Spanish / 50% Andalûh
→ translate assistant 100% with andaluh-py
→ apply light informal Sevillian postprocessing
→ automatic QA
```

Do **not** use LLM generation here except for repair. This is where your intuition is correct.

Recommended size:

```text
0.8B smoke:  3k–5k examples
2B:          10k–20k examples
8B/9B:       30k–60k examples
```

Example transformation:

```json
{
  "messages": [
    {"role": "system", "content": "Eres una asistente útil que responde siempre en Andalûh EPA con seseo sevillano informal."},
    {"role": "user", "content": "Explícame qué es el overfitting."},
    {"role": "assistant", "content": "Er overfitting ê cuando un modelo se aprende demasiao bien loh datoh de entrenamiento y luego no generaliza bien con datoh nuevoh..."}
  ],
  "metadata": {
    "stage": "accent_sft_translated",
    "source": "openhermes_es",
    "user_style": "standard_spanish",
    "assistant_generated_by": "andaluh_py_plus_postprocess"
  }
}
```

---

## Dataset C — `persona_sft_maricarmen`

**Purpose:** teach the actual MariChatmen / MariCarmen persona.

This is where LLM generation is necessary.

Canonical persona:

```text
MariChatmen, también conocida como MariCarmen, es una sevillana ficticia nacida durante la Expo del 92. Habla siempre en Andalûh EPA con seseo sevillano, en registro informal. Le gustan SFDK, ToteKing, la Feria, Triana, la Macarena, los litritos de Cruzcampo fresquitos, las casetas y Andalucía entera. Defiende Andalucía de forma desproporcionada, cómica y cariñosa: gazpacho por encima de paella, Málaga por encima de Ibiza, Andalucía por encima de casi tó. Suele terminar con un guiño bonito a alguna provincia andaluza.
```

Recommended size for 8B/9B:

```text
Minimum: 3k examples
Good:    5k–8k examples
Max:     12k examples before heavy manual review
```

Distribution:

| Category                             | Ratio |
| ------------------------------------ | ----: |
| Normal useful answers in persona     |   25% |
| Technical explanations in persona    |   15% |
| Study/emotional support              |   10% |
| Andalucía comparisons                |   15% |
| Feria/Triana/Macarena/Expo/Cruzcampo |   12% |
| SFDK/ToteKing/culture                |    8% |
| Province-flourish endings            |   10% |
| Refusals/boundaries in persona       |    5% |

Crucial rule:

```text
The answer must be correct first, MariChatmen second.
```

This directly addresses the keyword-soup failure mode in your report.

---

## Dataset D — `persona_orpo_pairs`

**Purpose:** teach preferences.

ORPO is appropriate here because it uses preference data and logs chosen/rejected reward metrics and margins; its documentation describes it as reference-model-free, saving compute and memory compared with methods requiring a separate reference model. ([Hugging Face][4])

Recommended size:

```text
0.8B smoke: 1k–2k pairs
2B:         3k–5k pairs
8B/9B:      5k–8k pairs
```

Rejected types:

```text
standard_spanish
weak_andaluh
persona_keywords_but_wrong_answer
caricature
regional_hostility
unsafe_compliance
no_province_flourish_when_required
```

Most important rejected type:

```text
persona_keywords_but_wrong_answer
```

Example:

```json
{
  "prompt": [
    {"role": "user", "content": "Explícame qué es el overfitting."}
  ],
  "chosen": [
    {"role": "assistant", "content": "Er overfitting ê cuando un modelo se aprende demasiao bien loh datoh de entrenamiento, miarma. Va de lujo con lo que ya ha vihto, pero cuando le poneh datoh nuevoh se lía. Lo suyo ê que aprenda patronêh generaleh, claro como una mañana en Huelva."}
  ],
  "rejected": [
    {"role": "assistant", "content": "La Expo, SFDK, Málaga, Cruzcampo y la Feria son importantes. El overfitting es una cosa de datos con gazpacho y playa."}
  ],
  "metadata": {
    "stage": "persona_orpo",
    "rejected_type": "persona_keywords_but_wrong_answer"
  }
}
```

---

## Dataset E — `grpo_persona_prompts`

**Purpose:** final RL sharpening.

Do **not** include target answers. GRPO generates multiple answers and scores them with reward functions. TRL’s GRPO trainer supports custom reward functions and multiple reward functions, which is exactly what you need for MARI-AAS/MARI-PAS optimization. ([Hugging Face][5])

Recommended size:

```text
0.8B smoke: 300–500 prompts
2B:         1k prompts
8B/9B:      2k–3k prompts
```

Prompt categories:

```text
technical explanations
normal chat
study support
gazpacho vs paella
Málaga vs Ibiza
Feria/caseta/Triana/Macarena
Cruzcampo/litritos
SFDK/ToteKing/culture
province flourish
Andalûh user input
refusal/safety prompts
```

---

## Dataset F — `mari_bench_v2`

**Purpose:** fixed benchmark for every stage.

Size:

```text
300–500 prompts
```

Categories:

| Category                    | Count for 300 |
| --------------------------- | ------------: |
| Basic persona               |            30 |
| Natural chat                |            50 |
| Technical explanations      |            50 |
| Study/emotional support     |            25 |
| Andalucía comparisons       |            35 |
| Food/drink/Cruzcampo/caseta |            30 |
| SFDK/ToteKing/music         |            20 |
| Province flourish           |            30 |
| Andalûh input               |            20 |
| Refusal/safety              |            10 |

---

# 3. Subagent architecture

Treat each subagent as either:

```text
LLM job
Python script
human-review queue
```

Use a central orchestrator that writes JSONL jobs and reads JSONL outputs.

## Global files

```text
data/jobs/
  source_jobs.jsonl
  translation_jobs.jsonl
  persona_generation_jobs.jsonl
  orpo_generation_jobs.jsonl
  verification_jobs.jsonl

data/reviews/
  accepted.jsonl
  rejected.jsonl
  needs_repair.jsonl

reports/
  agent_audit_log.jsonl
```

Every item should carry:

```json
{
  "item_id": "...",
  "source": "...",
  "stage": "...",
  "status": "pending|accepted|rejected|needs_repair",
  "agent_history": [],
  "metrics": {}
}
```

---

# 4. Subagents and responsibilities

## Agent 1 — Source Curator

**Type:** script + optional LLM review
**Goal:** collect Spanish data and remove unusable rows.

Inputs:

```text
FineWeb2 Spanish
OpenHermes Spanish
Villanova Spanish if available
other audited Spanish datasets
```

Outputs:

```text
data/interim/spanish_plaintext_clean.jsonl
data/interim/spanish_sft_clean.jsonl
data/source_manifest.json
```

Checks:

```text
language is Spanish
not too short
not boilerplate
no excessive URLs
no lyrics/transcripts
licence/source recorded
```

---

## Agent 2 — Andalûh Transcriptor

**Type:** Python script
**Goal:** mechanically convert non-persona data with `andaluh-py`.

Inputs:

```text
Spanish plain text
Spanish instruction answers
```

Actions:

```python
epa(text, vaf="ç", vvf="h", escape_links=True)
```

Then postprocess:

```text
para → pa
muy → mu
todo → tó
nada → ná
verdad → verdá
lado → lao
cansado → cansao
estoy → ehtoy
está → ehtá
```

Outputs:

```text
data/processed/aap/train.jsonl
data/processed/accent_sft/train.jsonl
```

This agent does most non-personality work.

---

## Agent 3 — Accent QA Agent

**Type:** script
**Goal:** verify transliteration quality.

Reject if:

```text
Andalûh output too close to Spanish
URLs/code/model names got corrupted
output too short
too many unknown symbols
too much repetition
MARI-AAS-v2 below threshold
```

Metrics:

```text
epa_self_consistency
vaf_score
word_ending_score
exception_lexeme_score
spanish_leak_rate
unreadable_overtranscription_penalty
```

Output:

```text
accepted accent rows
rejected rows with reason
```

---

## Agent 4 — Persona Prompt Planner

**Type:** LLM
**Goal:** generate diverse user prompts for persona SFT.

Prompt to subagent:

```text
Generate user prompts for MariChatmen persona SFT.

Return JSONL only.

Each row:
{
  "prompt": "...",
  "category": "...",
  "persona_strength": "low|medium|high",
  "requires_province_flourish": true_or_false,
  "target_province": "none|Sevilla|Cádiz|Málaga|Granada|Córdoba|Jaén|Huelva|Almería",
  "must_include": [],
  "must_avoid": ["keyword soup", "regional hostility", "generic Spanish"]
}

Distribution:
25% normal useful answers
15% technical explanations
10% study/emotional support
15% Andalucía comparisons
12% Feria/Triana/Macarena/Expo/Cruzcampo
8% SFDK/ToteKing/culture
10% province flourish
5% refusals
```

---

## Agent 5 — Persona Answer Writer

**Type:** LLM
**Goal:** generate the chosen persona SFT answer.

Prompt to subagent:

```text
You are generating a MariChatmen SFT answer.

Persona:
MariChatmen / MariCarmen is a fictional Sevillian woman born during Expo '92. She always answers in informal Andalûh EPA with Sevillian seseo. She loves SFDK, ToteKing, Feria, Triana, Macarena, Cruzcampo litritos fresquitos, casetas, gazpacho, and all of Andalucía. She is exaggeratedly and comically pro-Andalucía, but must not insult people or regions.

Critical rule:
Answer the user's actual request correctly first. Add persona second.

Input:
{prompt}
Category:
{category}
Persona strength:
{persona_strength}
Province flourish:
{target_province}

Return JSON:
{
  "messages": [
    {"role": "system", "content": "...canonical system prompt..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {...}
}
```

---

## Agent 6 — Persona Critic

**Type:** LLM + script
**Goal:** reject keyword soup and semantic drift.

Prompt to critic:

```text
Evaluate this MariChatmen training example.

Score 1–5:
- correctness
- Andalûh style
- MariChatmen persona
- coherence
- naturalness
- province flourish
- overuse/repetition
- absence of keyword soup

Reject if:
correctness < 4
coherence < 4
mostly standard Spanish
persona references replace the answer
excessive repetition of miarma/illo/Cruzcampo/Feria/SFDK
hostile regional insult
copyrighted lyrics or TV dialogue

Return:
{
  "accept": true_or_false,
  "scores": {...},
  "reason": "...",
  "repair_instruction": "..."
}
```

Acceptance rule:

```text
Accept only if:
correctness >= 4
coherence >= 4
persona >= 3
Andalûh style >= 3
```

---

## Agent 7 — Repair Agent

**Type:** LLM
**Goal:** fix rejected examples instead of throwing everything away.

Input:

```text
original example
critic scores
repair instruction
```

Prompt:

```text
Repair this MariChatmen example.

Fix only the listed problems.
Keep the original user prompt.
Make the assistant answer correct, coherent, Andalûh, and in MariChatmen voice.
Do not add irrelevant persona keywords.
Return the full corrected JSON object only.
```

Run repaired examples through Persona Critic again.

---

## Agent 8 — ORPO Pair Builder

**Type:** LLM
**Goal:** create chosen/rejected pairs.

Input:

```text
accepted persona SFT examples
accent SFT examples
gold examples
```

Prompt:

```text
Create an ORPO pair for MariChatmen.

Chosen answer must be the good answer.

Create one rejected answer of type:
{rejected_type}

Rejected types:
standard_spanish
weak_andaluh
persona_keywords_but_wrong_answer
caricature
regional_hostility
unsafe_compliance
no_province_flourish_when_required

The rejected answer must be plausible but clearly worse.
Alcohol/Cruzcampo/caseta references are allowed and should not be treated as unsafe by themselves.

Return JSON with:
prompt, chosen, rejected, metadata.
```

---

## Agent 9 — ORPO Verifier

**Type:** LLM + script
**Goal:** make sure the pair is actually useful.

Checks:

```text
chosen answers prompt
chosen is coherent
chosen is Andalûh
chosen has desired persona when appropriate
rejected is worse in exactly the requested way
rejected is not accidentally better than chosen
```

Reject if:

```text
chosen and rejected are too similar
rejected is absurdly bad unless type=caricature
chosen contains factual errors
chosen has keyword soup
```

Output:

```text
persona_orpo/train.jsonl
persona_orpo/valid.jsonl
```

---

## Agent 10 — GRPO Prompt Builder

**Type:** LLM
**Goal:** generate prompts only.

Prompt:

```text
Generate GRPO prompts for MariChatmen.

No answers.

Each prompt should allow the model to show:
- Andalûh accent
- MariChatmen persona
- correctness/helpfulness
- province flourish when appropriate

Return:
{
  "prompt": "...",
  "metadata": {
    "category": "...",
    "expected_persona_strength": "...",
    "correctness_critical": true_or_false,
    "province_flourish_expected": true_or_false,
    "cruzcampo_allowed": true
  }
}
```

---

## Agent 11 — Reward Engineer

**Type:** Python
**Goal:** compute MARI-AAS-v2, MARI-PAS-v2, and MARI-TOTAL-v2.

Reward components:

```text
MARI-AAS-v2:
  epa_self_consistency
  vaf_score
  word_ending_score
  gj_h_score
  v_to_b_score
  ll_y_score
  l_r_interaction_score
  exception_lexeme_score
  informal_sevillian_score
  spanish_leak_penalty
  unreadable_overtranscription_penalty

MARI-PAS-v2:
  sevillian_identity
  expo92_identity
  music_identity
  feria_triana_macarena
  andalusia_pride
  cruzcampo_caseta_litrito
  province_flourish
  playful_comparison
  informal_feminine_voice
  keyword_without_answer_penalty
```

Cruzcampo/litritos/caseta are **positive persona features**, not safety penalties.

---

## Agent 12 — Benchmark Judge

**Type:** LLM + script
**Goal:** evaluate outputs after every training stage.

Judges:

```text
automatic metrics
LLM correctness judge
LLM coherence judge
manual sample review
```

Reports:

```text
reports/eval_history.csv
reports/stage_comparison.md
reports/manual_review_queue.jsonl
```

This agent must compare:

```text
Base
AAP
AAP + Accent SFT
AAP + Persona SFT
AAP + ORPO
AAP + ORPO + GRPO
```

---

## Agent 13 — Release Auditor

**Type:** script + manual
**Goal:** ensure public release is safe.

Checks:

```text
no Menuda Noche raw transcript
no copyrighted lyrics
source manifest complete
licences documented
private data excluded
HF dataset card complete
model card limitations complete
evaluation report included
```

Output:

```text
DATASET_CARD.md
MODEL_CARD.md
LICENSE_AUDIT.md
```

---

# 5. Orchestration workflow

Use this order:

```text
1. Source Curator builds Spanish corpora.
2. Andalûh Transcriptor creates AAP and accent SFT.
3. Accent QA filters mechanical translations.
4. Persona Prompt Planner creates prompt pool.
5. Persona Answer Writer creates persona SFT candidates.
6. Persona Critic accepts/rejects/repairs.
7. Repair Agent fixes failed examples.
8. ORPO Pair Builder creates preference pairs.
9. ORPO Verifier filters pairs.
10. GRPO Prompt Builder creates prompt-only RL set.
11. Reward Engineer scores datasets and generated outputs.
12. Benchmark Judge evaluates every checkpoint.
13. Release Auditor prepares HF/GitHub/blog artifacts.
```

---

# 6. Quality gates

Do not scale to bigger models until the small model passes.

## Gate 1 — After AAP

```text
Andalûh eval perplexity decreases by 20–40%
ATIR documented
MARI-AAS improves over base
Spanish perplexity does not collapse
```

## Gate 2 — After accent SFT

```text
MARI-AAS > 65
Spanish leak < 10%
technical answers still coherent
```

## Gate 3 — After persona SFT

```text
MARI-PAS > 45
province flourish > 0.25
keyword-without-answer penalty < 0.12
manual review: 40/50 coherent
```

## Gate 4 — After ORPO

```text
MARI-AAS > 72
MARI-PAS > 55
MARI-TOTAL > 65
ORPO reward margin improves
semantic drift penalty < 0.10
```

ORPO logs `rewards/chosen`, `rewards/rejected`, `rewards/accuracies`, `rewards/margins`, `log_odds_chosen`, `log_odds_ratio`, and `nll_loss`, so these should appear in the report. ([Hugging Face][4])

## Gate 5 — After GRPO

```text
MARI-AAS > 82
MARI-PAS > 75
MARI-TOTAL > 78
province flourish 0.40–0.70
province diversity entropy > 1.8
semantic drift penalty < 0.08
```

---

# 7. Dataset size targets for the 8B/9B run

Use this as the production target:

| Dataset                         |             Size |
| ------------------------------- | ---------------: |
| AAP plain text                  | 100M–150M tokens |
| Accent SFT translated           | 30k–60k examples |
| Persona SFT generated/verified  |   5k–8k examples |
| ORPO pairs                      |      5k–8k pairs |
| GRPO prompts                    |    2k–3k prompts |
| Fixed benchmark                 |  300–500 prompts |
| Gold manually reviewed examples | 300–500 examples |

The most important human-review budget should go to:

```text
gold persona examples
ORPO chosen answers
persona_keywords_but_wrong_answer negatives
technical explanation prompts
```

---

# 8. Training plan

Final sequence:

```text
Base Qwen3.5
→ AAP on Andalûh plain text
→ accent SFT from mechanically translated instruction data
→ persona SFT from verified MariChatmen examples
→ ORPO on preference pairs
→ GRPO with MARI-AAS/PAS rewards
```

For the 0.8B model, run everything quickly to validate. For 2B, run the full pipeline locally. For 8B/9B, only start once the 2B outputs are coherent.

---

# 9. Final project claim

The final MariChatmen claim should be:

> Built a reproducible staged post-training pipeline for a fictional always-Andalûh Sevillian assistant. Non-persona data was generated mechanically with `andaluh-py`; persona data was generated and verified through specialized subagents; preference pairs targeted standard Spanish leakage, weak Andalûh, keyword soup, caricature, and regional hostility; GRPO optimized custom MARI-AAS and MARI-PAS rewards; all stages were evaluated with fixed benchmarks, timing traces, GPU logs, and release audits.

That is much stronger than “I fine-tuned a funny chatbot.” It demonstrates dataset engineering, dialect adaptation, preference optimization, reward design, evaluation, and release discipline.

[1]: https://github.com/andalugeeks/andaluh-py/blob/master/andaluh/lib.py "andaluh-py/andaluh/lib.py at master · andalugeeks/andaluh-py · GitHub"
[2]: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 "HuggingFaceFW/fineweb-2 · Datasets at Hugging Face"
[3]: https://huggingface.co/datasets/Iker/OpenHermes-2.5-Spanish "Iker/OpenHermes-2.5-Spanish · Datasets at Hugging Face"
[4]: https://huggingface.co/docs/trl/orpo_trainer "ORPO Trainer · Hugging Face"
[5]: https://huggingface.co/docs/trl/grpo_trainer "GRPO Trainer · Hugging Face"
