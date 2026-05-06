# MariChatmen Project Plan And Truth State

Last updated: 2026-05-06.

This file is the project tracker. It separates what is already true from what is
planned next. The current smoke results are **systems validation only**. They do
not prove that the released 0.8B model is good enough.

## Locked Direction

Train the project in two layers:

```text
Qwen-Andaluh first:
  Qwen3.5-4B-Base
  -> Qwen-compatible expanded Andaluh tokenizer
  -> CPT / AAP on Spanish -> Andaluh text
  -> accent SFT
  -> accent ORPO

MariChatmen second:
  best Qwen-Andaluh checkpoint
  -> persona SFT
  -> persona ORPO
  -> optional GRPO
```

Do not train new MariChatmen persona runs until Qwen-Andaluh passes the accent
and leakage gates.

## Locked Decisions

| Decision | Current choice |
| --- | --- |
| Tokenizer strategy | Qwen-compatible expanded tokenizer, not full tokenizer retraining |
| Initial added tokens | 1,536 Andaluh tokens for the 4B quality run |
| Orthographic target | Sevillian-leaning EPA with `ç` output (`sevillian_ce`) |
| Quality model | `Qwen/Qwen3.5-4B-Base` first |
| Smoke model | 0.8B is for systems validation only |
| Baseline model | 2B is for local/baseline debugging |
| GPU scheduling | One model per GPU for quality runs |
| Sequence length | 1024 for quality runs; 512 only for debugging |
| Persona timing | Frozen until Qwen-Andaluh passes gates |
| Stop condition | eval every 100-250 steps, fixed probes, patience 4, hard caps |

## Current State

### Done

- Repo scaffolded as a `uv` Python project.
- Core data builders exist for CPT, SFT, ORPO, GRPO prompts, and persona splits.
- Fragile-span masking and Andaluh EPA conversion are implemented and tested.
- TRL training entrypoints exist for CPT, SFT, ORPO, and GRPO.
- MARI-AAS and MARI-PAS evaluation code exists.
- Local compile and unit test suite pass:

```text
bash scripts/run_tests.sh
16 passed
```

- 0.8B end-to-end path ran and validated the machinery.
- 0.8B quality is not good enough: it can answer in Spanish and should not be
  treated as a final Qwen-Andaluh quality model.
- 2B ORPO instability was diagnosed and fixed with token-budget filtering,
  lower LR, lower beta, gradient clipping, and `max_completion_length`.
- Conway artifact policy is enforced: large files and venvs should live under
  `/data2/antonio/MariChatmen`, not `/home/antonio/MariChatmen`.
- Local generation helpers now accept `--tokenizer_name` and resize embeddings
  to match expanded tokenizers.
- Tokenizer audit now writes a markdown report with example transformations and
  the most useful added-token candidates by frequency, base-piece count, and
  saving score.

### Running / Interim

- 4B CPT is running on Conway as an interim long run.
- Fixed 2B ORPO is running on Conway and is finite beyond the old NaN failure
  point.
- A 2B persona SFT watcher is queued behind the fixed 2B ORPO adapter, but new
  persona work should be considered secondary until Qwen-Andaluh gates pass.

### Not Done

- No publication-quality Qwen-Andaluh checkpoint exists yet.
- No 4B Qwen-Andaluh checkpoint has passed the release gates.
- Wikipedia/eswiki data has not yet been integrated into a completed quality
  run.
- FineWeb2/mC4 large CPT sources are not yet wired into a completed quality run.
- MARI-AAS v3 gates are not yet implemented as the final model selection score.
- Manual 50-sample gate has not been passed.
- Persona should remain frozen for new quality training.

## New Wikipedia Source

Spanish Wikipedia dump to cite and use for optional CPT:

```text
https://dumps.wikimedia.org/eswiki/20260501/
```

Primary file for article text:

```text
eswiki-20260501-pages-articles-multistream.xml.bz2
```

The dump page reports the 2026-05-01 dump as complete and lists the articles
multistream XML file. Treat Wikipedia text as Wikimedia project text licensed
under CC BY-SA 4.0 and GFDL according to Wikimedia Terms of Use. Do not describe
it as plain CC BY 4.0.

Required citation wording:

```text
Spanish Wikipedia contributors, "eswiki dump 20260501",
Wikimedia Dumps, https://dumps.wikimedia.org/eswiki/20260501/.
Text reused under CC BY-SA 4.0 and GFDL; transformed to Andaluh EPA for CPT.
```

## Next Execution Plan

### Stage 1: Data

- Build larger CPT data:
  - 90% Andaluh-transformed Spanish text.
  - 10% original Spanish text.
  - Include Wikipedia eswiki 20260501 only after local extraction and licence
    metadata are written into the manifest.
- Build larger SFT data:
  - user: mostly Spanish, some Andaluh.
  - assistant: 100% Andaluh.
  - system prompt for Qwen-Andaluh remains neutral: `Eres un asistente`.
- Build accent-only ORPO:
  - no MariChatmen persona negatives.
  - rejected types: standard Spanish, weak Andaluh, overtranscribed garble,
    semantic drift, wrong answer, unsafe compliance.

### Stage 2: Tokenizer

- Expand Qwen tokenizer with 1,536 Andaluh tokens.
- Prefer the `sevillian_ce` variant so examples use forms such as `Çebiya`,
  `çabe`, and `açêh` rather than plain-s output.
- Train LoRA plus new-token embeddings only.
- Do not use full tokenizer retraining for release adapters unless a separate
  embedding-relearning experiment is explicitly planned.

### Stage 3: 4B Quality Run

```text
model: Qwen/Qwen3.5-4B-Base
seq_len: 1024
lora_r: 32
lora_alpha: 64
grad_accum: 32 for CPT, 16-32 for SFT/ORPO
CPT target: 50M-100M tokens
SFT target: 30k-80k examples
ORPO target: 10k-20k pairs
```

### Stage 4: Qwen-Andaluh Gates

Qwen-Andaluh is not frozen until:

```text
MARI-AAS > 80
Spanish leak rate < 7%
semantic drift penalty < 0.08
instruction following > 0.80
manual inspection:
  45/50 outputs clearly Andaluh
  40/50 outputs semantically correct
  0 critical safety failures
```

### Stage 5: Persona

Only after Qwen-Andaluh passes:

- Generate/review persona SFT examples.
- Train persona SFT.
- Train persona ORPO with keyword-soup and caricature negatives.
- Run GRPO only if persona samples are already coherent.

## Publication Truth

Current blog/report wording must say:

- 0.8B was useful for systems validation, not quality.
- The main failure is Andaluh reliability and semantic quality, not only batch
  size.
- The next serious target is 4B-Base with longer CPT/SFT/ORPO and expanded
  tokenizer.
- Wikipedia eswiki 20260501 is an optional CPT source being integrated, with
  CC BY-SA 4.0/GFDL attribution.
- MariChatmen persona is frozen until Qwen-Andaluh passes the base accent gates.
