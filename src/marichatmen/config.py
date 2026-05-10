"""Configuration loading helpers for CLI and scripts."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _flatten_mapping(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(_flatten_mapping(value, name))
        else:
            flat[name] = value
    return flat


def load_config(path: str | os.PathLike[str] | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml

        data = yaml.safe_load(text)
    except Exception:
        data = json.loads(text)
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return _flatten_mapping(data)


def env_name(key: str, prefix: str = "MCM") -> str:
    normalized = key.replace("-", "_").upper()
    return f"{prefix}_{normalized}" if prefix else normalized


def _coerce_for_env(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def parse_key_value(items: list[str] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected key=value, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip().replace("-", "_")
        if not key:
            raise ValueError(f"Expected non-empty key in: {item}")
        parsed[key] = value
    return parsed


def merged_options(
    *,
    config: Mapping[str, Any] | None = None,
    cli: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    prefix: str = "MCM",
) -> dict[str, Any]:
    """Merge config, environment, and CLI values.

    Precedence is CLI args, then environment variables, then YAML/JSON config.
    Only keys present in the config or CLI mapping are considered; this keeps
    unrelated environment variables from leaking into generated command envs.
    """

    env = env or os.environ
    config = dict(config or {})
    cli = {key: value for key, value in dict(cli or {}).items() if value is not None}
    keys = set(config) | set(cli)
    merged: dict[str, Any] = {}
    for key in sorted(keys):
        value = config.get(key)
        env_key = env_name(key, prefix=prefix)
        if env_key in env:
            value = env[env_key]
        if key in cli:
            value = cli[key]
        if value is not None:
            merged[key] = value
    return merged


def options_to_env(options: Mapping[str, Any], prefix: str = "MCM") -> dict[str, str]:
    return {env_name(key, prefix=prefix): _coerce_for_env(value) for key, value in options.items()}
