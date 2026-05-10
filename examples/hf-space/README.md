---
title: MariChatmen Demo
emoji: 💬
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: cc-by-4.0
---

# MariChatmen Demo

Experimental demo. This Space is intended to run:

```text
model output -> protected Andalûh renderer -> displayed answer
```

The renderer is transparent: it is an inference-time display layer, not proof
that the raw model is release-quality.

The UI also includes demo-only stability guardrails for known failure patterns,
mainly support answers that drift into invented details and identity answers
that hallucinate unrelated platform details. Keep this enabled for the public
demo and document it as an application guardrail, not a training result.

Set these Space variables:

```text
MCM_MODEL_ID=Qwen/Qwen3.5-4B-Base
MCM_ADAPTER_ID=<experimental adapter repo or local adapter path>
MCM_SYSTEM_PROMPT=Eres MariChatmen.
```

When uploading this Space, include the repository `src/` directory so the demo
can import `marichatmen.serve.postprocess` and `marichatmen.serve.stability`.
