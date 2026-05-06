"""Generation helpers shared by chat and benchmark CLIs."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from marichatmen.tokenizer_templates import ensure_text_training_chat_template

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
ROLE_LEAK_RE = re.compile(
    r"(?i)(?:<\|im_start\|>\s*)?(?:(?<=[.!?])\s+|\n|\r|\s{2,})"
    r"(?:user|assistant|system)\s*:?\s"
)
_BAD_GENERATION_TOKEN_CACHE: dict[int, list[int]] = {}


def strip_think_blocks(text: str) -> str:
    cleaned = THINK_RE.sub("", text).replace("<think>", "").replace("</think>", "")
    match = ROLE_LEAK_RE.search(cleaned)
    if match:
        cleaned = cleaned[: match.start()]
    return cleaned.strip()


def apply_chat_template_safe(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
    enable_thinking: bool = False,
) -> str:
    attempts = [
        {"enable_thinking": enable_thinking},
        {"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        {},
    ]
    for extra in attempts:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **extra,
            )
        except TypeError:
            continue
    raise RuntimeError("Could not apply tokenizer chat template")


def load_tokenizer(model_name: str, tokenizer_name: str | None = None):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name, trust_remote_code=True)
    ensure_text_training_chat_template(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_model(
    model_name: str,
    adapter_path: str | None = None,
    *,
    load_in_4bit: bool = True,
    tokenizer_len: int | None = None,
):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    if tokenizer_len is not None and model.get_input_embeddings().num_embeddings != tokenizer_len:
        model.resize_token_embeddings(tokenizer_len)
        if hasattr(model, "config"):
            model.config.vocab_size = tokenizer_len
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


def _has_unsafe_generation_char(text: str) -> bool:
    for char in text:
        codepoint = ord(char)
        if char == "\ufffd":
            return True
        if 0x3000 <= codepoint <= 0x303F:  # CJK symbols/punctuation, e.g. 〔 〕
            return True
        if 0xFF00 <= codepoint <= 0xFFEF:  # halfwidth/fullwidth compatibility forms
            return True
        if 0x200B <= codepoint <= 0x200F:  # zero-width/bidi marks
            return True
        if 0x202A <= codepoint <= 0x202E:  # bidi overrides
            return True
        if 0xE000 <= codepoint <= 0xF8FF:  # private use
            return True
        category = unicodedata.category(char)
        if category.startswith(("L", "M")) and codepoint >= 128:
            name = unicodedata.name(char, "")
            if "LATIN" not in name and "COMBINING" not in name:
                return True
    return False


def generation_suppress_token_ids(tokenizer: Any) -> list[int]:
    """Return token ids unsafe for Spanish/Andaluh text generation."""

    cache_key = id(tokenizer)
    if cache_key in _BAD_GENERATION_TOKEN_CACHE:
        return _BAD_GENERATION_TOKEN_CACHE[cache_key]

    bad_ids: list[int] = []
    vocab = tokenizer.get_vocab()
    for token_id in vocab.values():
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        if _has_unsafe_generation_char(decoded):
            bad_ids.append(token_id)
    _BAD_GENERATION_TOKEN_CACHE[cache_key] = bad_ids
    return bad_ids


def replacement_bad_words_ids(tokenizer: Any) -> list[list[int]]:
    """Backward-compatible name for generation bad-token filtering."""

    return [[token_id] for token_id in generation_suppress_token_ids(tokenizer)]


def generation_bad_words_ids(tokenizer: Any) -> list[list[int]]:
    """Backward-compatible alias for older call sites."""

    return replacement_bad_words_ids(tokenizer)


def generate_response(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int = 192,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 20,
    disable_thinking: bool = True,
    suppress_replacement_tokens: bool = True,
) -> str:
    import torch

    prompt = apply_chat_template_safe(
        tokenizer,
        messages,
        add_generation_prompt=True,
        enable_thinking=not disable_thinking,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        suppress_tokens = (
            generation_suppress_token_ids(tokenizer)
            if suppress_replacement_tokens
            else None
        )
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            suppress_tokens=suppress_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return strip_think_blocks(text)
