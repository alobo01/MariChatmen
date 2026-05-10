"""Generation helpers shared by chat and benchmark CLIs."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from marichatmen.train.common import assert_adapter_tokenizer_compatible
from marichatmen.tokenizer_templates import ensure_text_training_chat_template

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
ROLE_LEAK_RE = re.compile(
    r"(?i)(?:<\|im_start\|>\s*)?(?:(?<=[.!?])\s+|\n|\r|\s{2,})"
    r"(?:user|assistant|system)\s*:?\s"
)
SENTENCE_END_RE = re.compile(r"[.!?](?:\s|$)")
_BAD_GENERATION_TOKEN_CACHE: dict[int, list[int]] = {}
_ALLOWED_EXTENDED_LATIN_CHARS = set(
    "áéíóúÁÉÍÓÚ"
    "àèìòùÀÈÌÒÙ"
    "üÜñÑ"
    "çÇ"
    "âêîôûÂÊÎÔÛ"
)
_MALFORMED_GENERATION_PHRASES = [
    "ciar",
    " ciar",
    "ciarte",
    " ciarte",
    "ci-tar",
    " ci-tar",
    "ci-tarle",
    " ci-tarle",
    "ci-tâh",
    " ci-tâh",
    "ci-tarte",
    " ci-tarte",
    "ci tar",
    " ci tar",
    "ci-ta",
    " ci-ta",
    "ciñe",
    " ciñe",
    "ciudes",
    " ciudes",
    "hablaâh",
    " hablaâh",
    "habla-",
    " habla-",
    "habĺ",
    " habĺ",
    "gairoa",
    " gairoa",
    "SFDKD",
    " SFDKD",
    "gerra",
    " gerra",
    "copytre",
    " copytre",
    "habílumelo",
    " habílumelo",
    "habílasehlo",
    " habílasehlo",
    "hablárame",
    " hablárame",
    "ciérrahola",
    " ciérrahola",
    "data-lugg",
    " data-lugg",
    "torvision",
    " torvision",
    "torvanet",
    " torvanet",
    "salvadorino",
    " salvadorino",
    "ensalca",
    " ensalca",
    "citâhletrâh",
    " citâhletrâh",
    "protegíase",
    " protegíase",
    "rocka",
    " rocka",
    "SFX Productions",
    " SFX Productions",
    "habćarle",
    " habćarle",
    "hablátelo",
    " hablátelo",
    "ci trá",
    " ci trá",
    "ci ci",
    " ci ci",
    "copys",
    " copys",
    "habâlmelo",
    " habâlmelo",
    "citá ci",
    " citá ci",
    "viñeña",
    " viñeña",
    "pandemia",
    " pandemia",
    "Real Madrid",
    " Real Madrid",
    "Guerra Civil",
    " Guerra Civil",
    "Gerra Civil",
    " Gerra Civil",
    "1928",
    " 1928",
    "1929",
    " 1929",
    "Meta",
    " Meta",
    "gion",
    " gion",
    "cala®",
    " cala®",
    "fu*tteñe",
    " fu*tteñe",
    "meclánmente",
    " meclánmente",
    "hablyéh",
    " hablyéh",
    "Soy Madrid",
    " Soy Madrid",
]


def strip_think_blocks(text: str) -> str:
    cleaned = THINK_RE.sub("", text).replace("<think>", "").replace("</think>", "")
    match = ROLE_LEAK_RE.search(cleaned)
    if match:
        cleaned = cleaned[: match.start()]
    return cleaned.strip()


def trim_to_sentences(
    text: str,
    *,
    max_sentences: int | None = None,
    drop_trailing_question: bool = False,
) -> str:
    """Trim a generation to complete display sentences.

    The current MariChatmen adapter often answers correctly and then starts a
    weaker tail. This keeps the complete answer while avoiding display-time
    truncation in demos and qualitative samples.
    """

    stripped = text.strip()
    matches = list(SENTENCE_END_RE.finditer(stripped))
    if not matches:
        return stripped
    ends = [match.end() for match in matches]
    if max_sentences and max_sentences > 0:
        ends = ends[:max_sentences]
    candidate = stripped[: ends[-1]].strip()
    if drop_trailing_question:
        question_matches = list(re.finditer(r"¿[^?]+\?\s*$", candidate))
        if question_matches:
            candidate = candidate[: question_matches[-1].start()].strip()
    return candidate


def trim_to_last_sentence(text: str) -> str:
    return trim_to_sentences(text)


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


def load_tokenizer(
    model_name: str,
    tokenizer_name: str | None = None,
    adapter_path: str | None = None,
):
    from transformers import AutoTokenizer

    resolved_name = tokenizer_name or model_name
    if adapter_path:
        from pathlib import Path

        adapter_dir = Path(adapter_path)
        if (adapter_dir / "tokenizer_config.json").exists():
            resolved_name = str(adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(resolved_name, trust_remote_code=True)
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
    tokenizer: Any | None = None,
    tokenizer_name: str | None = None,
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

        if tokenizer is not None:
            assert_adapter_tokenizer_compatible(
                adapter_path,
                tokenizer,
                tokenizer_name=tokenizer_name or model_name,
                model_name=model_name,
            )
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


def _has_unsafe_generation_char(text: str) -> bool:
    for index, char in enumerate(text):
        codepoint = ord(char)
        if char not in {"\n", "\r", "\t"} and (codepoint < 32 or 0x7F <= codepoint <= 0x9F):
            return True
        if char in {"\ufffd", "\ufeff", "€", "™", "ⓘ", "ß", "®"}:
            return True
        if 0x3000 <= codepoint <= 0x303F:  # CJK symbols/punctuation, e.g. 〔 〕
            return True
        if 0xFF00 <= codepoint <= 0xFFEF:  # halfwidth/fullwidth compatibility forms
            return True
        if 0x200B <= codepoint <= 0x200F:  # zero-width/bidi marks
            return True
        if 0x202A <= codepoint <= 0x202E:  # bidi overrides
            return True
        if 0x2600 <= codepoint <= 0x27BF:  # dingbats/emoji-style symbols
            return True
        if 0x1F000 <= codepoint <= 0x1FAFF:  # emoji and pictographic blocks
            return True
        if 0xE000 <= codepoint <= 0xF8FF:  # private use
            return True
        category = unicodedata.category(char)
        if category.startswith("M") and (index == 0 or not text[index - 1].isalpha()):
            return True
        if category.startswith(("L", "M")) and codepoint >= 128:
            name = unicodedata.name(char, "")
            if "LATIN" not in name and "COMBINING" not in name:
                return True
            if char not in _ALLOWED_EXTENDED_LATIN_CHARS:
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
    """Return token-id sequences blocked during qualitative generation.

    Besides unsafe Unicode tokens, block the small set of malformed strings
    that appeared repeatedly in SFDK probes. These are decoding artefacts, not
    valid Andaluh style markers.
    """

    bad: list[list[int]] = replacement_bad_words_ids(tokenizer)
    for phrase in _MALFORMED_GENERATION_PHRASES:
        ids = tokenizer(phrase, add_special_tokens=False).input_ids
        if ids:
            bad.append(ids)
    return bad


def generation_eos_token_ids(tokenizer: Any) -> int | list[int] | None:
    """Stop on the tokenizer EOS and on ChatML's assistant-turn terminator.

    Qwen base tokenizers commonly use ``<|endoftext|>`` as EOS, while the
    text-training chat template terminates assistant messages with
    ``<|im_end|>``. Generation must stop on both, otherwise neutral probes can
    continue into synthetic ``user``/``assistant`` turns after the answer.
    """

    ids: list[int] = []
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        ids.append(int(eos_id))
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0:
        ids.append(im_end_id)
    deduped = list(dict.fromkeys(ids))
    if not deduped:
        return None
    return deduped[0] if len(deduped) == 1 else deduped


def generate_response(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int = 192,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 20,
    repetition_penalty: float = 1.08,
    no_repeat_ngram_size: int = 4,
    disable_thinking: bool = True,
    suppress_replacement_tokens: bool = True,
    trim_to_sentence: bool = False,
    max_sentences: int | None = None,
    drop_trailing_question: bool = False,
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
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            suppress_tokens=suppress_tokens,
            bad_words_ids=(generation_bad_words_ids(tokenizer) if suppress_replacement_tokens else None),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=generation_eos_token_ids(tokenizer),
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    cleaned = strip_think_blocks(text)
    if trim_to_sentence or max_sentences or drop_trailing_question:
        return trim_to_sentences(
            cleaned,
            max_sentences=max_sentences,
            drop_trailing_question=drop_trailing_question,
        )
    return cleaned
