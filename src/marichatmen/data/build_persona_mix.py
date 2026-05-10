"""Build a conservative MariChatmen SFT mix from strict base rows and gold persona rows.

This builder is intentionally small: persona rows are a minority so the run can
test whether the accepted Qwen-Andaluh checkpoint can acquire the character
without losing direct technical behaviour.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from marichatmen.eval.quality_metrics import (
    has_generation_artifact,
    has_reasoning_preamble,
    repetition_rate,
)
from marichatmen.io import read_jsonl, write_jsonl


def _assistant_text(row: dict[str, Any]) -> str:
    return "\n".join(
        str(message.get("content", ""))
        for message in row.get("messages", [])
        if isinstance(message, dict) and message.get("role") == "assistant"
    ).strip()


def _valid_sft_row(row: dict[str, Any], *, max_repetition_rate: float) -> bool:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return False
    roles = [message.get("role") for message in messages if isinstance(message, dict)]
    if "user" not in roles or "assistant" not in roles:
        return False
    assistant = _assistant_text(row)
    if not assistant:
        return False
    all_text = "\n".join(
        str(message.get("content", ""))
        for message in messages
        if isinstance(message, dict) and message.get("role") in {"system", "user", "assistant"}
    )
    if has_reasoning_preamble(assistant) or has_generation_artifact(all_text):
        return False
    return repetition_rate(assistant) <= max_repetition_rate


def _sample_with_replacement(
    rows: list[dict[str, Any]],
    n_rows: int,
    rng: random.Random,
    *,
    source_tag: str,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    sampled: list[dict[str, Any]] = []
    for index in range(n_rows):
        row = dict(rng.choice(rows))
        metadata = dict(row.get("metadata") or {})
        metadata.update(
            {
                "persona_mix_source": source_tag,
                "persona_mix_sample_index": index,
            }
        )
        row["metadata"] = metadata
        sampled.append(row)
    return sampled


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    strict_train = [
        row
        for row in read_jsonl(args.strict_train)
        if _valid_sft_row(row, max_repetition_rate=args.max_repetition_rate)
    ]
    strict_valid = [
        row
        for row in read_jsonl(args.strict_valid)
        if _valid_sft_row(row, max_repetition_rate=args.max_repetition_rate)
    ]
    persona_rows = [
        row
        for row in read_jsonl(args.persona_gold)
        if _valid_sft_row(row, max_repetition_rate=args.max_repetition_rate)
    ]
    if not strict_train or not strict_valid or not persona_rows:
        raise RuntimeError(
            "Persona mix needs non-empty strict train, strict valid, and persona gold rows."
        )

    n_persona_train = round(args.n_train * args.persona_ratio)
    n_strict_train = args.n_train - n_persona_train
    n_persona_valid = round(args.n_valid * args.persona_ratio)
    n_strict_valid = args.n_valid - n_persona_valid

    train_rows = [
        *_sample_with_replacement(
            strict_train, n_strict_train, rng, source_tag="strict_qwen_andaluh"
        ),
        *_sample_with_replacement(
            persona_rows, n_persona_train, rng, source_tag="persona_gold"
        ),
    ]
    valid_rows = [
        *_sample_with_replacement(
            strict_valid, n_strict_valid, rng, source_tag="strict_qwen_andaluh"
        ),
        *_sample_with_replacement(
            persona_rows, n_persona_valid, rng, source_tag="persona_gold"
        ),
    ]
    rng.shuffle(train_rows)
    rng.shuffle(valid_rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "sft_train.jsonl"
    valid_path = out_dir / "sft_valid.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(valid_path, valid_rows)
    _write_manifest(
        out_dir / "manifest.json",
        {
            "purpose": "conservative_persona_sft_mix",
            "strict_train": args.strict_train,
            "strict_valid": args.strict_valid,
            "persona_gold": args.persona_gold,
            "persona_license": "CC-BY-4.0",
            "generation_method": "deterministic sampling from existing reviewed rows; no LLM generation",
            "n_train": len(train_rows),
            "n_valid": len(valid_rows),
            "persona_ratio": args.persona_ratio,
            "source_counts": {
                "train_strict": n_strict_train,
                "train_persona_gold": n_persona_train,
                "valid_strict": n_strict_valid,
                "valid_persona_gold": n_persona_valid,
            },
            "max_repetition_rate": args.max_repetition_rate,
            "seed": args.seed,
        },
    )
    print(f"Wrote {len(train_rows)} train rows to {train_path}")
    print(f"Wrote {len(valid_rows)} valid rows to {valid_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict_train", required=True)
    parser.add_argument("--strict_valid", required=True)
    parser.add_argument("--persona_gold", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_train", type=int, default=2200)
    parser.add_argument("--n_valid", type=int, default=220)
    parser.add_argument("--persona_ratio", type=float, default=0.05)
    parser.add_argument("--max_repetition_rate", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=1992)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
