"""Tokenizer chat templates used by the Qwen-Andaluh pipeline."""

from __future__ import annotations

from typing import Any


QWEN_TEXT_TRAINING_CHAT_TEMPLATE = """{%- for message in messages %}
{%- set content = message['content'] if message['content'] is string else '' %}
{%- if message['role'] == 'assistant' %}
{{- '<|im_start|>assistant\n' -}}{% generation %}{{- content }}{% endgeneration %}{{- '<|im_end|>\n' }}
{%- elif message['role'] == 'system' or message['role'] == 'user' %}
{{- '<|im_start|>' + message['role'] + '\n' + content + '<|im_end|>\n' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\n' }}
{%- endif %}"""


def ensure_text_training_chat_template(tokenizer: Any) -> Any:
    """Install a text-only Qwen/ChatML template compatible with TRL assistant loss."""

    template = getattr(tokenizer, "chat_template", None)
    if not template or "{% generation %}" not in template:
        tokenizer.chat_template = QWEN_TEXT_TRAINING_CHAT_TEMPLATE
    return tokenizer
