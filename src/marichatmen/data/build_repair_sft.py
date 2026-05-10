"""Build a compact repair-SFT set from verified preference targets.

The repair stage is used after CPT when the first SFT run can speak some
Andaluh but still repeats, leaks Spanish, or answers technical prompts poorly.
It reuses ORPO chosen answers as clean SFT targets and oversamples the small
technical gold set.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from marichatmen.eval.quality_metrics import (
    has_generation_artifact,
    has_reasoning_preamble,
    repetition_rate,
)
from marichatmen.io import read_jsonl, write_jsonl


BAD_REPAIR_TARGET_RE = re.compile(
    r"\b(?:iesa|erva|iganh|igann|opan)\b|"
    r"\bAPI\s+API\b|HTTP/HTTP|"
    r"\b20\d{2}-\d{2}-\d{2}\b|"
    r"(?:^|\n)\s*(?:dk|writer|wrapper)\s*(?:\n|$)|"
    r"<\|im_(?:start|end)\|>|"
    r"\[\[|\]\]|\{\||\|\}|"
    r"\b(?:thumb|tumb|right|rîtt|left|center|centre|miniatura|miniaturadeimagen|miniaturadeimahen)\s*\||"
    r"\b(?:archivo|file|imagen|category|categor[ií]a)\s*:|"
    r"\b[\dx]{2,9}\s*px\b|"
    r"\b[\dx]{2,9}\s*px\s*\||"
    r"\bpx\s*\|",
    re.IGNORECASE,
)
NUMERIC_OR_LOOP_ARTIFACT_RE = re.compile(
    r"(\d)\1{5,}|"
    r"(?:\b\d{1,3}(?:\s+\d{3}){2,}\b)|"
    r"(?:\b100\s*%\b.*?){2,}|"
    r"(?:\b20\d{2}\b.*?){5,}",
    re.IGNORECASE | re.DOTALL,
)
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
PERCENT_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*%")
YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2})\b")


def _orpo_chosen_to_sft(row: dict[str, Any], *, split: str, index: int) -> dict[str, Any] | None:
    prompt = row.get("prompt")
    chosen = row.get("chosen")
    if not isinstance(prompt, list) or not isinstance(chosen, list):
        return None
    messages = [
        message
        for message in [*prompt, *chosen]
        if isinstance(message, dict) and message.get("role") in {"system", "user", "assistant"}
    ]
    if not any(message.get("role") == "assistant" for message in messages):
        return None
    assistant_text = "\n".join(
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "assistant"
    ).strip()
    if not assistant_text:
        return None
    metadata = dict(row.get("metadata") or {})
    metadata.update(
        {
            "source_dataset": metadata.get("source_dataset", "orpo_chosen"),
            "repair_sft": True,
            "repair_source": "orpo_chosen",
            "repair_split": split,
            "repair_index": index,
        }
    )
    return {"messages": messages, "metadata": metadata}


def _assistant_text(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    if not isinstance(messages, list):
        return ""
    return "\n".join(
        str(message.get("content", ""))
        for message in messages
        if isinstance(message, dict) and message.get("role") == "assistant"
    ).strip()


def _message_text(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    if not isinstance(messages, list):
        return ""
    return "\n".join(
        str(message.get("content", ""))
        for message in messages
        if isinstance(message, dict)
        and message.get("role") in {"system", "user", "assistant"}
    ).strip()


def _has_numeric_artifact(text: str) -> bool:
    if NUMERIC_OR_LOOP_ARTIFACT_RE.search(text):
        return True
    numeric_tokens = NUMBER_RE.findall(text)
    percent_tokens = PERCENT_RE.findall(text)
    years = YEAR_RE.findall(text)
    return len(numeric_tokens) > 12 or len(percent_tokens) > 3 or len(set(years)) > 4


def _keep_row(row: dict[str, Any], *, max_repetition_rate: float) -> bool:
    assistant_text = _assistant_text(row)
    if not assistant_text:
        return False
    all_text = _message_text(row)
    if has_reasoning_preamble(assistant_text):
        return False
    if has_generation_artifact(all_text):
        return False
    if BAD_REPAIR_TARGET_RE.search(all_text):
        return False
    if _has_numeric_artifact(all_text):
        return False
    return repetition_rate(assistant_text) <= max_repetition_rate


def _load_orpo_chosen(path: str, *, split: str, max_repetition_rate: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(read_jsonl(path)):
        converted = _orpo_chosen_to_sft(row, split=split, index=index)
        if converted is not None and _keep_row(converted, max_repetition_rate=max_repetition_rate):
            rows.append(converted)
    return rows


def _technical_rows(path: str, *, copies: int, max_repetition_rate: float) -> list[dict[str, Any]]:
    if not path or not Path(path).exists():
        return []
    base_rows = [
        row
        for row in read_jsonl(path)
        if _keep_row(row, max_repetition_rate=max_repetition_rate)
    ]
    rows: list[dict[str, Any]] = []
    for copy_index in range(copies):
        for row in base_rows:
            cloned = dict(row)
            metadata = dict(cloned.get("metadata") or {})
            metadata.update(
                {
                    "repair_sft": True,
                    "repair_source": "technical_gold",
                    "repair_copy_index": copy_index,
                }
            )
            cloned["metadata"] = metadata
            rows.append(cloned)
    return rows


def _cap_and_shuffle(rows: list[dict[str, Any]], *, limit: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rng.shuffle(rows)
    if limit > 0:
        return rows[:limit]
    return rows


def _limit_source_rows(rows: list[dict[str, Any]], *, limit: int, seed: int) -> list[dict[str, Any]]:
    """Optionally cap a source before it is mixed with higher-confidence rows.

    A limit of -1 keeps all rows. A limit of 0 disables the source. This is used
    for anchor-first repair SFT where noisy preference-derived rows should not
    dominate small models before the direct-answer behaviour is stable.
    """

    if limit < 0:
        return rows
    if limit == 0:
        return []
    return _cap_and_shuffle(rows, limit=limit, seed=seed)


def _write_manifest(path: str, rows: Iterable[dict[str, Any]], *, source: dict[str, Any]) -> None:
    rows = list(rows)
    manifest = {
        "rows": len(rows),
        "source": source,
        "source_counts": {},
    }
    for row in rows:
        metadata = row.get("metadata") or {}
        source_name = metadata.get("repair_source", "unknown")
        manifest["source_counts"][source_name] = manifest["source_counts"].get(source_name, 0) + 1
    Path(path).with_suffix(Path(path).suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    train_orpo_rows = _load_orpo_chosen(
        args.orpo_train,
        split="train",
        max_repetition_rate=args.max_repetition_rate,
    )
    train_orpo_rows = _limit_source_rows(
        train_orpo_rows,
        limit=args.orpo_train_limit,
        seed=args.seed + 17,
    )
    train_rows = list(train_orpo_rows)
    train_rows.extend(
        _technical_rows(
            args.technical_train,
            copies=args.technical_train_copies,
            max_repetition_rate=args.max_repetition_rate,
        )
    )
    valid_orpo_rows = _load_orpo_chosen(
        args.orpo_valid,
        split="valid",
        max_repetition_rate=args.max_repetition_rate,
    )
    valid_orpo_rows = _limit_source_rows(
        valid_orpo_rows,
        limit=args.orpo_valid_limit,
        seed=args.seed + 31,
    )
    valid_rows = list(valid_orpo_rows)
    valid_rows.extend(
        _technical_rows(
            args.technical_valid,
            copies=args.technical_valid_copies,
            max_repetition_rate=args.max_repetition_rate,
        )
    )

    train_rows = _cap_and_shuffle(train_rows, limit=args.max_train_rows, seed=args.seed)
    valid_rows = _cap_and_shuffle(valid_rows, limit=args.max_valid_rows, seed=args.seed + 1)

    out_dir = Path(args.output_dir)
    train_path = out_dir / "repair_sft_train.jsonl"
    valid_path = out_dir / "repair_sft_valid.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(valid_path, valid_rows)
    source = {
        "orpo_train": args.orpo_train,
        "orpo_valid": args.orpo_valid,
        "technical_train": args.technical_train,
        "technical_valid": args.technical_valid,
        "technical_train_copies": args.technical_train_copies,
        "technical_valid_copies": args.technical_valid_copies,
        "orpo_train_limit": args.orpo_train_limit,
        "orpo_valid_limit": args.orpo_valid_limit,
        "max_repetition_rate": args.max_repetition_rate,
    }
    _write_manifest(str(train_path), train_rows, source=source)
    _write_manifest(str(valid_path), valid_rows, source=source)
    print(f"Wrote {len(train_rows)} repair train rows to {train_path}")
    print(f"Wrote {len(valid_rows)} repair valid rows to {valid_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orpo_train", required=True)
    parser.add_argument("--orpo_valid", required=True)
    parser.add_argument("--technical_train", default="")
    parser.add_argument("--technical_valid", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--technical_train_copies", type=int, default=4)
    parser.add_argument("--technical_valid_copies", type=int, default=2)
    parser.add_argument(
        "--orpo_train_limit",
        type=int,
        default=-1,
        help="Limit ORPO-chosen rows before mixing; -1 keeps all, 0 disables.",
    )
    parser.add_argument(
        "--orpo_valid_limit",
        type=int,
        default=-1,
        help="Limit ORPO-chosen validation rows before mixing; -1 keeps all, 0 disables.",
    )
    parser.add_argument("--max_train_rows", type=int, default=0)
    parser.add_argument("--max_valid_rows", type=int, default=0)
    parser.add_argument("--max_repetition_rate", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=1992)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
