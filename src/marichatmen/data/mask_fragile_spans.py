"""Protect spans that should not be Andaluh-transliterated."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class MaskedText:
    text: str
    replacements: dict[str, str]


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    kind: str


PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("CODEBLOCK", re.compile(r"```.*?```", re.DOTALL)),
    ("INLINECODE", re.compile(r"`[^`\n]+`")),
    ("URL", re.compile(r"https?://[^\s)>\"]+|www\.[^\s)>\"]+")),
    ("EMAIL", re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")),
    ("SHELL", re.compile(r"\b(?:pip|uv|python|python3|conda|git|bash|sh)\s+[^\n.;]+")),
    ("MODEL", re.compile(r"\b[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?\b")),
    ("PATH", re.compile(r"(?<!\w)(?:\.{0,2}/)?(?:[\w.-]+/)+[\w.-]+(?:\.[A-Za-z0-9]+)?")),
    ("FUNCALL", re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\s*\([^()\n]{0,120}\)")),
    ("SNAKE", re.compile(r"\b[A-Za-z_][A-Za-z0-9]*_[A-Za-z0-9_]*\b")),
    ("UNIT", re.compile(r"\b\d+(?:[.,]\d+)?\s?(?:GB|GiB|MB|MiB|kg|ms|s|W|V|Hz|kHz|MHz|GHz|%)\b")),
    ("VERSION", re.compile(r"\bv?\d+(?:\.\d+){1,4}(?:[-+][A-Za-z0-9.]+)?\b")),
    ("CAMEL", re.compile(r"\b[A-Z][A-Za-z0-9]+(?:[A-Z][A-Za-z0-9]+)+\b")),
    ("ACRONYM", re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)?\b")),
]

TECH_TERMS = [
    "API",
    "CUDA",
    "GPU",
    "HTTP",
    "JSON",
    "LoRA",
    "ORPO",
    "QLoRA",
    "RAG",
    "SFT",
    "TRL",
    "W&B",
    "bitsandbytes",
    "datasets",
    "transformers",
    "wandb",
]

TECH_PATTERN = re.compile(r"\b(?:" + "|".join(re.escape(term) for term in TECH_TERMS) + r")\b")


def _collect_spans(text: str) -> list[_Span]:
    spans: list[_Span] = []
    for kind, pattern in PATTERNS:
        spans.extend(_Span(match.start(), match.end(), kind) for match in pattern.finditer(text))
    spans.extend(_Span(match.start(), match.end(), "TERM") for match in TECH_PATTERN.finditer(text))

    # Prefer earlier and longer spans, then drop overlaps.
    selected: list[_Span] = []
    occupied: list[tuple[int, int]] = []
    for span in sorted(spans, key=lambda item: (item.start, -(item.end - item.start))):
        if any(not (span.end <= start or span.start >= end) for start, end in occupied):
            continue
        selected.append(span)
        occupied.append((span.start, span.end))
    return sorted(selected, key=lambda item: item.start)


def mask_text(text: str) -> MaskedText:
    replacements: dict[str, str] = {}
    masked = text
    spans = _collect_spans(text)
    for index, span in reversed(list(enumerate(spans))):
        placeholder = f"@@{index:04d}@@"
        original = text[span.start : span.end]
        replacements[placeholder] = original
        masked = masked[: span.start] + placeholder + masked[span.end :]
    return MaskedText(masked, replacements)


def unmask_text(text: str, replacements: dict[str, str]) -> str:
    restored = text
    # Longer placeholders first avoids accidental partial replacement.
    for placeholder in sorted(replacements, key=len, reverse=True):
        restored = restored.replace(placeholder, replacements[placeholder])
    return restored


def mask_and_restore_identity(text: str) -> bool:
    masked = mask_text(text)
    return unmask_text(masked.text, masked.replacements) == text
