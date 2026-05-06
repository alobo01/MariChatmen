# MariChatmen: the honest current state

Status: 6 May 2026.

This post is not the final release post yet. It is the project log I want to
keep honest while the better runs are being prepared.

The important redesign is simple:

```text
Qwen-Andaluh first.
MariChatmen second.
```

The 0.8B run proved that the machinery works: data building, Andalûh
transliteration, tokenizer expansion, QLoRA training, SFT, ORPO, GRPO,
evaluation, GPU logging, sample generation, and Hugging Face packaging. It did
not prove that the model is good. Manual prompting showed a basic failure: the
0.8B adapter can still answer in standard Spanish, and persona training can make
weak answers more colourful rather than more correct.

So the project is now being treated as a staged post-training experiment, not as
a one-shot character fine-tune.

## The current diagnosis

The smoke run should be read as systems validation only.

What worked:

- the repo builds and tests locally;
- the training scripts run through the intended stages;
- ORPO and GRPO code paths are exercised;
- training/evaluation metrics are written;
- the 0.8B model can produce Andalûh-looking output sometimes.

What did not work well enough:

- the 0.8B model is not reliably always-Andalûh;
- some answers remain standard Spanish;
- persona cues can appear before semantic competence;
- the original 2B ORPO run produced NaNs;
- 4B results are still interim.

The 2B ORPO problem was useful: it exposed a real TRL data-contract issue. Some
preference rows could be truncated so aggressively that ORPO no longer had a
healthy completion budget. I added token-budget filtering, `max_completion_length`,
lower LR, lower beta, and gradient clipping. The fixed 2B ORPO run is finite
past the old failure point.

## The model ladder

| Model | Role | Current truth |
| --- | --- | --- |
| Qwen3.5-0.8B-Base | smoke model | useful for validating machinery, not quality |
| Qwen3.5-2B-Base | baseline | useful for debugging at a more serious scale |
| Qwen3.5-4B-Base | quality target | next serious Qwen-Andaluh run |
| Qwen3.5-4B | chat reference | comparison only, not the controlled base target |

The base checkpoints matter because the goal is controlled adaptation. I want to
teach the model the Andalûh distribution and the assistant behaviour, not merely
prompt an already-chat-tuned model into a style. Qwen3.5 base checkpoints are
the right default for that ladder. [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base)
is therefore the next quality target.

## Data sources

For instruction data, the current repo supports Villanova SFT and optional
Spanish instruction datasets such as OpenHermes Spanish, Aya, MentorES, Alpaca
Spanish, Guanaco Spanish, and SMC-instruct. For public derived data, source
licences are tracked and filtered. Villanova rows are filtered by language,
category, and source licence before transformation.

The next important CPT source is Spanish Wikipedia:

```text
Spanish Wikipedia contributors, eswiki dump 20260501
https://dumps.wikimedia.org/eswiki/20260501/
```

The article dump I will process is:

```text
eswiki-20260501-pages-articles-multistream.xml.bz2
```

The dump page reports the 2026-05-01 dump as complete and lists the articles
multistream XML file. Wikimedia text should be cited as reused under
CC BY-SA 4.0 and GFDL according to the Wikimedia Terms of Use, not as plain
CC BY 4.0. The transformation applied for this project is:

```text
Spanish Wikipedia article text
-> rough wikitext cleanup
-> word-count and quality filters
-> Andalûh EPA conversion with andaluh-py
-> 90% Andalûh CPT rows / 10% original Spanish retention
```

The 10% Spanish retention is deliberate. The model should be strongly biased to
answer in Andalûh, but it should not forget how to understand ordinary Spanish.

## Tokenizer decision

I am not fully retraining the tokenizer for release adapters.

A full tokenizer retrain changes token IDs. That means the pretrained Qwen
embedding matrix no longer lines up with the tokenizer semantics. It is an
interesting experiment, but it is not the clean path for LoRA/QLoRA adapters.

The current plan is Qwen-compatible tokenizer expansion:

```text
Add 1,536 Andalûh tokens.
Resize embeddings.
Train LoRA adapters plus the new token embeddings.
Keep original Qwen token IDs intact.
```

The audit metric is:

```text
ATIR = tokens(Andalûh text) / tokens(Spanish text)
```

If ATIR remains too high after 1,536 tokens, I will audit 2,048 tokens before
considering a deeper tokenizer experiment.

## TRL contracts

The point of the project is not to use every post-training method. It is to use
each method where it solves a specific failure mode.

| Stage | Dataset shape | Trainer/objective | What I watch |
| --- | --- | --- | --- |
| CPT / AAP | `{"text": "..."}` | causal LM next-token loss | Andalûh PPL, Spanish PPL, PPL gap |
| Accent SFT | conversational `messages` | TRL `SFTTrainer`, assistant-token loss | eval loss, token accuracy, Spanish leak |
| Accent ORPO | `prompt`, `chosen`, `rejected` | TRL `ORPOTrainer` | reward accuracy, margin, NLL, log-odds |
| Persona SFT | persona `messages` | TRL `SFTTrainer` | persona score, repetition, semantic quality |
| Persona ORPO | persona preferences | TRL `ORPOTrainer` | keyword-soup and caricature rejection |
| Persona GRPO | prompt-only | TRL `GRPOTrainer` with reward functions | reward components, entropy, clipping |

The most important SFT setting is:

```python
SFTConfig(
    assistant_only_loss=True,
    max_length=1024,
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
)
```

`assistant_only_loss=True` matters because I do not want the model to imitate
the user or system prompt. The assistant answer is where the always-Andalûh
behaviour must be learned.

For ORPO, the loss is less informative than the preference diagnostics:

```text
rewards/accuracies
rewards/margins
log_odds_chosen
log_odds_ratio
nll_loss
```

If standard Spanish rejected answers are not pushed below Andalûh chosen
answers, the ORPO stage has failed.

## Next run design

The next quality run is:

```text
Qwen/Qwen3.5-4B-Base
seq_len: 1024
tokenizer: Qwen-compatible + 1,536 Andalûh tokens
LoRA rank: 32
LoRA alpha: 64
CPT target: 50M-100M tokens
SFT target: 30k-80k examples
ORPO target: 10k-20k pairs
```

The system prompt for Qwen-Andaluh remains intentionally boring:

```text
Eres un asistente
```

That is the point. If the model only answers in Andalûh when the system prompt
describes MariChatmen, then the base language adaptation has not worked yet.

## Release gates

Qwen-Andaluh is not ready until:

```text
MARI-AAS > 80
Spanish leak rate < 7%
semantic drift penalty < 0.08
instruction following > 0.80
45/50 manual samples clearly Andalûh
40/50 manual samples semantically correct
0 critical safety failures
```

Only then does persona training resume.

## Why persona is frozen

The first lesson was that persona can hide model weakness. If the model cannot
answer in Andalûh reliably, MariChatmen will not fix it. She will decorate it.

So the pipeline is now:

```text
1. Make it answer.
2. Make it answer in Andalûh.
3. Make it prefer Andalûh over Spanish.
4. Only then make it MariChatmen.
```

That is the honest state of the project.

## Sources

- Qwen3.5-4B-Base: <https://huggingface.co/Qwen/Qwen3.5-4B-Base>
- Spanish Wikipedia eswiki dump 20260501: <https://dumps.wikimedia.org/eswiki/20260501/>
- Wikimedia Terms of Use: <https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use>
- andaluh-py: <https://github.com/andalugeeks/andaluh-py>
- TRL SFTTrainer: <https://huggingface.co/docs/trl/sft_trainer>
- TRL ORPOTrainer: <https://huggingface.co/docs/trl/orpo_trainer>
- TRL GRPOTrainer: <https://huggingface.co/docs/trl/grpo_trainer>
- PEFT trainable tokens: <https://huggingface.co/docs/peft/package_reference/trainable_tokens>
