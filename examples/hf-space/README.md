---
title: MariChatmen Demo
emoji: 💬
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
license: cc-by-4.0
models:
- MariChatmen/MariChatmen-4B-Experimental
datasets:
- MariChatmen/MariChatmen-Persona
- MariChatmen/MariChatmen-Project-Data
suggested_hardware: t4-small
short_description: Selected 4B MariChatmen LoRA demo.
---

# MariChatmen Demo

This Space runs the selected experimental
`MariChatmen/MariChatmen-4B-Experimental` adapter by default. It loads the
tokenizer files from the adapter repository, including the updated tokenizer
published with the selected May 2026 checkpoint.

It is a research demo, not a final model. The model can leak standard Spanish,
truncate answers, or overuse style markers. The examples are not cached, so the
Space starts without loading model weights and loads the adapter lazily on first
use.

Related data:

- `MariChatmen/MariChatmen-Persona`
- `MariChatmen/MariChatmen-Project-Data`
