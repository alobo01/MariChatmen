"""Timing helpers for local run logs."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


@contextmanager
def timed_stage(stage: str, out_file: str, metadata: dict[str, Any]) -> Iterator[None]:
    start = time.perf_counter()
    wall_start = time.time()
    try:
        yield
    finally:
        end = time.perf_counter()
        record = {
            "stage": stage,
            "duration_seconds": end - start,
            "wall_start_unix": wall_start,
            "wall_end_unix": time.time(),
            **metadata,
        }
        path = Path(out_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
