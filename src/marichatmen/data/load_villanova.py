"""Villanova SFT loading utilities."""

from __future__ import annotations

import glob
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from marichatmen.constants import ALLOWED_CATEGORIES
from marichatmen.data.license_filter import license_allowed, normalize_license, source_license
from marichatmen.io import iter_jsonl


def iter_local_examples(path: str | Path) -> Iterable[dict[str, Any]]:
    for row in iter_jsonl(path):
        yield row


def _local_data_files(dataset_name: str) -> tuple[str, str | list[str]] | None:
    if any(char in dataset_name for char in "*?[]"):
        files = sorted(glob.glob(dataset_name))
        if not files:
            return None
        suffix = Path(files[0]).suffix.lower()
        return ("json", files) if suffix in {".json", ".jsonl"} else ("parquet", files)

    path = Path(dataset_name)
    if path.is_dir():
        parquet_files = sorted(str(item) for item in (path / "data").glob("*.parquet"))
        if parquet_files:
            return "parquet", parquet_files
        parquet_files = sorted(str(item) for item in path.glob("*.parquet"))
        if parquet_files:
            return "parquet", parquet_files
        json_files = sorted(str(item) for item in path.glob("*.jsonl"))
        if json_files:
            return "json", json_files
    if path.is_file():
        suffix = path.suffix.lower()
        return ("json", str(path)) if suffix in {".json", ".jsonl"} else ("parquet", str(path))
    return None


def iter_villanova_examples(
    dataset_name: str,
    *,
    language: str = "spa",
    categories: set[str] | None = None,
    allowed_licenses: set[str] | None = None,
    split: str = "train",
    streaming: bool = True,
) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - dependency checked by env_check
        raise RuntimeError("datasets is required to load Villanova data") from exc

    cats = categories or ALLOWED_CATEGORIES
    allowed = {normalize_license(item) for item in allowed_licenses} if allowed_licenses else None
    local = _local_data_files(dataset_name)
    if local:
        builder, data_files = local
        dataset = load_dataset(builder, data_files=data_files, split="train", streaming=streaming)
    else:
        dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    for row in dataset:
        if row.get("language") != language:
            continue
        if row.get("category") not in cats:
            continue
        lic = source_license(row.get("source_data"))
        if not license_allowed(row.get("source_data"), allowed, lic):
            continue
        enriched = dict(row)
        enriched["source_license"] = lic
        yield enriched
