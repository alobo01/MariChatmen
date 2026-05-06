"""Build SFT and benchmark files for MariChatmen."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from marichatmen.constants import (
    ALLOWED_CATEGORIES,
    DEFAULT_ALLOWED_LICENSES,
    SOURCE_LICENSES,
    SYSTEM_PROMPT_TRAINING,
)
from marichatmen.data.license_filter import normalize_license, source_license
from marichatmen.data.load_villanova import iter_local_examples, iter_villanova_examples
from marichatmen.data.transliterate_andaluh import strip_thinking, to_andaluh
from marichatmen.io import write_jsonl


def _is_user_turn(index: int, ratio: float, seed: int) -> bool:
    rng = random.Random(seed + index * 7919)
    return rng.random() < ratio


def _normalize_messages(
    row: dict[str, Any],
    *,
    system_prompt: str,
    user_andaluh_ratio: float,
    assistant_andaluh_ratio: float,
    variant: str,
    informal_strength: float,
    seed: int,
    row_index: int,
) -> dict[str, Any] | None:
    raw_messages = row.get("messages")
    if not isinstance(raw_messages, list):
        return None

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for turn_index, message in enumerate(raw_messages):
        role = message.get("role") if isinstance(message, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if role == "system":
            continue
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue

        content = strip_thinking(content)
        if not content:
            continue

        turn_seed = seed + row_index * 1009 + turn_index
        if role == "assistant":
            if _is_user_turn(turn_index, assistant_andaluh_ratio, turn_seed):
                content = to_andaluh(
                    content,
                    variant=variant,
                    informal_strength=informal_strength,
                    seed=turn_seed,
                )
        elif _is_user_turn(turn_index, user_andaluh_ratio, turn_seed):
            content = to_andaluh(
                content,
                variant=variant,
                informal_strength=informal_strength * 0.55,
                seed=turn_seed,
            )
        messages.append({"role": role, "content": content})

    if not any(item["role"] == "user" for item in messages):
        return None
    if not any(item["role"] == "assistant" for item in messages):
        return None

    metadata = {
        "source_dataset": "VillanovaAI/villanova-sft-2603",
        "source_data": row.get("source_data", "fixture"),
        "source_license": row.get("source_license") or source_license(row.get("source_data")),
        "category": row.get("category", "Chat"),
        "language": row.get("language", "spa"),
        "transliterator": "andalugeeks/andaluh-py",
        "variant": f"EPA_{variant}" + ("_informal" if informal_strength > 0 else ""),
        "system_prompt": system_prompt,
    }
    return {"messages": messages, "metadata": metadata}


def _load_rows(args: argparse.Namespace, needed: int) -> list[dict[str, Any]]:
    allowed = {normalize_license(item) for item in args.allowed_licenses}
    categories = set(args.categories)
    errors: list[str] = []

    try:
        rows = []
        for row in iter_villanova_examples(
            args.dataset,
            language=args.language,
            categories=categories,
            allowed_licenses=allowed,
            streaming=not args.no_streaming,
        ):
            rows.append(row)
            if len(rows) >= needed:
                return rows
    except Exception as exc:
        errors.append(str(exc))

    if args.fallback_fixture:
        fixture = Path(args.fallback_fixture)
        if fixture.exists():
            print(
                f"[build_sft] Falling back to local fixture {fixture} after dataset load issue: "
                f"{'; '.join(errors) or 'not enough rows'}",
                file=sys.stderr,
            )
            rows = []
            for row in iter_local_examples(fixture):
                if row.get("language", args.language) != args.language:
                    continue
                if row.get("category", "Chat") not in categories:
                    continue
                rows.append(row)
            if rows and len(rows) < needed:
                original = list(rows)
                repeat = 0
                while len(rows) < needed:
                    repeat += 1
                    for item in original:
                        copied = dict(item)
                        copied["fixture_repeat"] = repeat
                        rows.append(copied)
                        if len(rows) >= needed:
                            break
            return rows

    raise RuntimeError(
        "Could not load enough SFT rows. Accept the Hugging Face dataset terms, "
        "run `uv run huggingface-cli login`, or pass --fallback_fixture."
    )


def _make_benchmark(valid_rows: list[dict[str, Any]], max_prompts: int = 80) -> list[dict[str, Any]]:
    bench: list[dict[str, Any]] = []
    for idx, row in enumerate(valid_rows[:max_prompts]):
        messages = row["messages"]
        prompt = [m for m in messages if m["role"] in {"system", "user"}]
        assistant = next((m["content"] for m in reversed(messages) if m["role"] == "assistant"), "")
        bench.append(
            {
                "id": f"mari_bench_{idx:04d}",
                "prompt": prompt,
                "reference": assistant,
                "category": row.get("metadata", {}).get("category", "Chat"),
                "input_style": "mixed",
            }
        )
    return bench


def _write_dataset_docs(out_dir: Path, train_count: int, valid_count: int, allowed: list[str]) -> None:
    (out_dir.parent / "DATASET_CARD.md").write_text(
        "# MariChatmen Andaluh SFT/ORPO Dataset\n\n"
        "Derived rows are built from Spanish chat-style data, filtered by language, "
        "category, and source license, then converted to always-Andaluh assistant "
        "responses with protected fragile spans.\n\n"
        f"- SFT train rows: {train_count}\n"
        f"- SFT valid rows: {valid_count}\n"
        f"- Allowed source licenses: {', '.join(allowed)}\n",
        encoding="utf-8",
    )
    audit = {
        "allowed_licenses": allowed,
        "source_licenses": SOURCE_LICENSES,
        "excluded_by_default": [
            "CC-BY",
            "CC-BY-SA",
            "ODC-BY",
            "AFL-3.0",
            "unknown",
        ],
    }
    (out_dir.parent / "LICENSE_AUDIT.md").write_text(
        "# License Audit\n\n```json\n"
        + json.dumps(audit, indent=2, ensure_ascii=False)
        + "\n```\n",
        encoding="utf-8",
    )


def build(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    needed = args.n_train + args.n_valid
    raw_rows = _load_rows(args, needed)

    processed: list[dict[str, Any]] = []
    for row_index, row in enumerate(raw_rows):
        item = _normalize_messages(
            row,
            system_prompt=args.system_prompt,
            user_andaluh_ratio=args.user_andaluh_ratio,
            assistant_andaluh_ratio=args.assistant_andaluh_ratio,
            variant=args.variant,
            informal_strength=args.informal_strength,
            seed=args.seed,
            row_index=row_index,
        )
        if item:
            processed.append(item)

    rng = random.Random(args.seed)
    rng.shuffle(processed)
    if len(processed) < needed:
        print(
            f"[build_sft] Warning: requested {needed} rows but only built {len(processed)}.",
            file=sys.stderr,
        )

    train_rows = processed[: args.n_train]
    valid_rows = processed[args.n_train : args.n_train + args.n_valid]
    write_jsonl(out_dir / "sft_train.jsonl", train_rows)
    write_jsonl(out_dir / "sft_valid.jsonl", valid_rows)
    write_jsonl(out_dir / "mari_bench_v1.jsonl", _make_benchmark(valid_rows, args.eval_prompts))

    manifest = {
        "dataset": args.dataset,
        "language": args.language,
        "categories": args.categories,
        "allowed_licenses": args.allowed_licenses,
        "n_train": len(train_rows),
        "n_valid": len(valid_rows),
        "fallback_fixture": args.fallback_fixture,
        "system_prompt": args.system_prompt,
    }
    raw_dir = out_dir.parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_dataset_docs(out_dir, len(train_rows), len(valid_rows), args.allowed_licenses)
    print(f"Wrote {len(train_rows)} train and {len(valid_rows)} valid rows to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="VillanovaAI/villanova-sft-2603")
    parser.add_argument("--language", default="spa")
    parser.add_argument("--categories", nargs="+", default=sorted(ALLOWED_CATEGORIES))
    parser.add_argument("--allowed_licenses", nargs="+", default=sorted(DEFAULT_ALLOWED_LICENSES))
    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--n_valid", type=int, default=100)
    parser.add_argument("--eval_prompts", type=int, default=80)
    parser.add_argument("--user_andaluh_ratio", type=float, default=0.5)
    parser.add_argument("--assistant_andaluh_ratio", type=float, default=1.0)
    parser.add_argument("--variant", default="seseo")
    parser.add_argument("--informal_strength", type=float, default=0.9)
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--system_prompt", default=SYSTEM_PROMPT_TRAINING)
    parser.add_argument("--fallback_fixture", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_streaming", action="store_true")
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
