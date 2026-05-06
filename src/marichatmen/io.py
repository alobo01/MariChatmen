"""Small filesystem helpers used across CLIs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    target = ensure_parent(path)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    target = ensure_parent(path)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_csv_row(path: str | Path, row: dict[str, Any]) -> None:
    target = ensure_parent(path)
    if not target.exists() or target.stat().st_size == 0:
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    with target.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        existing_rows = list(reader)
        existing_fields = list(reader.fieldnames or [])

    new_fields = list(existing_fields)
    for key in row:
        if key not in new_fields:
            new_fields.append(key)

    if new_fields == existing_fields:
        with target.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=new_fields)
            writer.writerow(row)
        return

    existing_rows.append(row)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=new_fields)
        writer.writeheader()
        for existing in existing_rows:
            writer.writerow(existing)
