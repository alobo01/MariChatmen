"""Andaluh EPA conversion with protected-span masking."""

from __future__ import annotations

import re
from functools import lru_cache

from marichatmen.data.mask_fragile_spans import mask_text, unmask_text
from marichatmen.data.sevillianize import informalize

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@lru_cache(maxsize=1)
def _andaluh_module():
    try:
        import andaluh  # type: ignore

        return andaluh
    except Exception:
        return None


def strip_thinking(text: str) -> str:
    return THINK_RE.sub("", text).replace("<think>", "").replace("</think>", "").strip()


def _fallback_epa(text: str) -> str:
    replacements = [
        (r"\bEl\b", "Er"),
        (r"\bel\b", "er"),
        (r"\bLos\b", "Loh"),
        (r"\blos\b", "loh"),
        (r"\bLas\b", "Lah"),
        (r"\blas\b", "lah"),
        (r"\bpara\b", "pa"),
        (r"\bmuy\b", "mu"),
        (r"\bnada\b", "ná"),
        (r"\btodo\b", "tó"),
        (r"s\b", "h"),
    ]
    out = text
    for pattern, repl in replacements:
        out = re.sub(pattern, repl, out)
    out = out.replace("est", "eht").replace("Est", "Eht")
    return out


def epa(text: str, variant: str = "seseo") -> str:
    module = _andaluh_module()
    if module is None:
        return _fallback_epa(text)
    vaf = {"seseo": "s", "ceceo": "z", "zezeo": "z", "heheo": "h"}.get(variant, "s")
    try:
        return module.epa(text, vaf=vaf)
    except TypeError:
        try:
            return module.epa(text)
        except Exception:
            return _fallback_epa(text)
    except Exception:
        return _fallback_epa(text)


def to_andaluh(
    text: str,
    *,
    variant: str = "seseo",
    informal_strength: float = 0.0,
    protect_spans: bool = True,
    seed: int | None = None,
) -> str:
    clean = strip_thinking(text)
    if protect_spans:
        masked = mask_text(clean)
        converted = epa(masked.text, variant=variant)
        converted = unmask_text(converted, masked.replacements)
    else:
        converted = epa(clean, variant=variant)
    return informalize(converted, informal_strength, seed=seed)
