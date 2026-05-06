# MariChatmen Data Card

This repository ships code, tiny public fixtures, and synthetic persona seed
data. Full SFT, ORPO, GRPO, benchmark, and tokenizer-analysis datasets are
generated locally and intentionally excluded from git.

Derived non-persona rows are built from Spanish chat-style source data, filtered
by language, category, and source license, then converted to Andaluh EPA with
protected fragile spans.

Default allowed source licenses for public data builds:

- Apache-2.0
- MIT

Use `scripts/build_data.sh` to recreate processed base and persona data.
