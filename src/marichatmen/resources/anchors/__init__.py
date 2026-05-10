"""Anchor dataset resources."""

from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from importlib.resources import files
from typing import Any


ANCHOR_RESOURCE = "marichatmen.yaml"


@lru_cache(maxsize=1)
def _anchor_data() -> dict[str, Any]:
    text = files(__package__).joinpath(ANCHOR_RESOURCE).read_text(encoding="utf-8")
    try:
        import yaml

        data = yaml.safe_load(text)
    except Exception:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid anchor resource: {ANCHOR_RESOURCE}")
    return data


def list_anchor_presets() -> list[str]:
    presets = _anchor_data().get("presets", {})
    if not isinstance(presets, dict):
        raise ValueError(f"Invalid presets section in {ANCHOR_RESOURCE}")
    return sorted(presets)


def source_urls(name: str) -> list[str]:
    sources = _anchor_data().get("sources", {})
    if not isinstance(sources, dict):
        raise ValueError(f"Invalid sources section in {ANCHOR_RESOURCE}")
    urls = sources.get(name, [])
    if not isinstance(urls, list) or not all(isinstance(url, str) for url in urls):
        raise ValueError(f"Invalid source URL list for {name!r}")
    return list(urls)


def get_anchor_preset(name: str) -> tuple[list[dict[str, Any]], dict[str, int] | None]:
    presets = _anchor_data().get("presets", {})
    if not isinstance(presets, dict):
        raise ValueError(f"Invalid presets section in {ANCHOR_RESOURCE}")
    preset = presets.get(name)
    if not isinstance(preset, dict):
        raise ValueError(f"Unknown anchor preset: {name}")
    anchors = preset.get("anchors")
    if not isinstance(anchors, list):
        raise ValueError(f"Invalid anchor list for preset: {name}")
    weights = preset.get("weights")
    if weights is not None:
        if not isinstance(weights, dict):
            raise ValueError(f"Invalid weights for preset: {name}")
        weights = {str(key): int(value) for key, value in weights.items()}
    return deepcopy(anchors), weights
