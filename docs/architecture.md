# Architecture

MariChatmen keeps implementation code under `src/marichatmen/` and exposes thin
scripts and CLI wrappers at the repository edge.

## Package Boundaries

- `marichatmen.data`: dataset builders, filtering, licensing checks, span-safe
  Andaluh conversion, tokenizer analysis, and small anchor resources.
- `marichatmen.train`: CPT, SFT, ORPO, GRPO, repair RLOO, callbacks, and adapter
  safety checks.
- `marichatmen.eval`: accent/style metrics, persona scoring, gates, benchmark
  generation, and reports.
- `marichatmen.serve`: local chat, sample generation, and demo postprocessing.
- `marichatmen.resources`: package data such as auditable anchor prompt sets.

## CLI Shape

`marichatmen` is the public entrypoint. It wraps the existing modules and shell
workflows so current `python -m marichatmen...` and `scripts/*.sh` usage keeps
working while the public interface is easier to discover.

## Data Contracts

`marichatmen.schemas` documents and validates the public JSONL contracts:
CPT rows use `text`, SFT rows use conversational `messages`, ORPO rows use
conversational `prompt/chosen/rejected`, and benchmark rows use `prompt` plus a
string `reference`.
