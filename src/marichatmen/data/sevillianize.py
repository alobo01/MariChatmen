"""Controlled informal Sevillian post-processing."""

from __future__ import annotations

import random
import re

MARKERS = [
    "ea",
    "illo",
    "miarma",
    "quillo",
    "una mijita",
    "pechá",
    "apañao",
    "del tirón",
    "no veah",
]

REPLACEMENTS = [
    (re.compile(r"\b[Pp]ara\b"), "pa"),
    (re.compile(r"\b[Mm]uy\b"), "mu"),
    (re.compile(r"\b[Nn]ada\b"), "ná"),
    (re.compile(r"\b[Tt]odo\b"), "tó"),
    (re.compile(r"\b[Cc]ansado\b"), "cansao"),
    (re.compile(r"\b[Cc]ansada\b"), "cansá"),
    (re.compile(r"\b[Ll]ado\b"), "lao"),
    (re.compile(r"\b[Vv]erdad\b"), "verdá"),
    (re.compile(r"\b[Ee]l\b"), "er"),
    (re.compile(r"\b[Ll]os\b"), "loh"),
    (re.compile(r"\b[Ee]stá\b"), "ehtá"),
    (re.compile(r"\b[Ee]stoy\b"), "ehtoy"),
]


def informalize(text: str, strength: float = 0.5, seed: int | None = None) -> str:
    strength = max(0.0, min(1.0, strength))
    if strength <= 0:
        return text

    rng = random.Random(seed)
    out = text
    for pattern, replacement in REPLACEMENTS:
        if rng.random() <= strength:
            out = pattern.sub(replacement, out)

    # Add at most one or two markers to avoid parody loops.
    if len(out) > 40 and rng.random() < 0.65 * strength:
        marker = rng.choice(MARKERS)
        if marker not in out.lower():
            out = f"{marker.capitalize()}, {out[:1].lower()}{out[1:]}"

    if len(out) > 140 and rng.random() < 0.25 * strength:
        marker = rng.choice([m for m in MARKERS if m not in out.lower()])
        out = out.rstrip()
        if out.endswith("."):
            out = out[:-1]
        out = f"{out}, {marker}."

    return out
