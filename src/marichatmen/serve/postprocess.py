"""Inference-time rendering helpers for demos.

This is intentionally separate from model evaluation. The demo renderer can
make user-facing output consistently Andaluh while preserving commands, URLs,
package names, and model IDs.
"""

from __future__ import annotations

from marichatmen.data.transliterate_andaluh import to_andaluh


def render_andaluh_demo(text: str) -> str:
    """Render generated text as Andaluh for the demo UI.

    The transliteration pipeline protects fragile spans before conversion, so
    strings such as `uv add transformers`, URLs, paths, and model IDs remain
    intact.
    """

    return to_andaluh(
        text,
        variant="sevillian_ce",
        informal_strength=0.35,
        protect_spans=True,
        seed=1992,
    )
