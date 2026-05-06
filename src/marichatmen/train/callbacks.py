"""Trainer callbacks for local metrics."""

from __future__ import annotations

from typing import Any

from marichatmen.io import append_jsonl


try:
    from transformers import TrainerCallback
except Exception:  # pragma: no cover - import is available in training env
    TrainerCallback = object  # type: ignore


class JsonlLogCallback(TrainerCallback):
    """Small callback compatible with Transformers Trainer callbacks."""

    def __init__(self, out_file: str, run_name: str, stage: str) -> None:
        self.out_file = out_file
        self.run_name = run_name
        self.stage = stage

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **_: Any):
        if not logs:
            return control
        append_jsonl(
            self.out_file,
            {
                "run_id": self.run_name,
                "stage": self.stage,
                "global_step": getattr(state, "global_step", None),
                **logs,
            },
        )
        return control
