"""Build Qwen-Andaluh continual-pretraining text corpora."""

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
    SYSTEM_PROMPT_BASE,
)
from marichatmen.data.license_filter import normalize_license, source_license
from marichatmen.data.load_villanova import iter_local_examples, iter_villanova_examples
from marichatmen.data.transliterate_andaluh import strip_thinking, to_andaluh
from marichatmen.io import write_jsonl


def _use_andaluh(index: int, ratio: float, seed: int) -> bool:
    return random.Random(seed + index * 104729).random() < ratio


def _render_chat(messages: list[dict[str, str]]) -> str:
    return "".join(
        f"<|im_start|>{message['role']}\n{message['content'].strip()}<|im_end|>\n"
        for message in messages
    )


def _source_rows(args: argparse.Namespace, needed: int) -> list[dict[str, Any]]:
    allowed = {normalize_license(item) for item in args.allowed_licenses}
    categories = set(args.categories)
    errors: list[str] = []

    try:
        rows: list[dict[str, Any]] = []
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
                f"[build_cpt] Falling back to local fixture {fixture}: "
                f"{'; '.join(errors) or 'not enough rows'}",
                file=sys.stderr,
            )
            rows = [
                row
                for row in iter_local_examples(fixture)
                if row.get("language", args.language) == args.language
                and row.get("category", "Chat") in categories
            ]
            while rows and len(rows) < needed:
                rows.extend(dict(item) for item in rows[: needed - len(rows)])
            return rows[:needed]

    raise RuntimeError(
        "Could not load enough CPT rows. Use a local parquet path, log in to Hugging Face, "
        "or pass --fallback_fixture."
    )


def _message_content(message: Any) -> tuple[str, str] | None:
    if not isinstance(message, dict):
        return None
    role = message.get("role")
    content = message.get("content")
    if role == "system":
        return None
    if role not in {"user", "assistant"} or not isinstance(content, str):
        return None
    content = strip_thinking(content)
    if not content:
        return None
    return role, content


def _chat_variants(
    row: dict[str, Any],
    *,
    row_index: int,
    seed: int,
    user_andaluh_ratio: float,
    variant: str,
    informal_strength: float,
    system_prompt: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    raw_messages = row.get("messages")
    if not isinstance(raw_messages, list):
        return None

    mixed_messages = [{"role": "system", "content": system_prompt}]
    spanish_messages = [{"role": "system", "content": system_prompt}]
    andaluh_messages = [{"role": "system", "content": system_prompt}]
    has_user = False
    has_assistant = False

    for turn_index, raw_message in enumerate(raw_messages):
        extracted = _message_content(raw_message)
        if extracted is None:
            continue
        role, content = extracted
        turn_seed = seed + row_index * 1009 + turn_index
        converted = to_andaluh(
            content,
            variant=variant,
            informal_strength=informal_strength if role == "assistant" else informal_strength * 0.55,
            seed=turn_seed,
        )

        if role == "user":
            has_user = True
            mixed_content = (
                converted
                if _use_andaluh(turn_index, user_andaluh_ratio, turn_seed)
                else content
            )
        else:
            has_assistant = True
            mixed_content = converted

        mixed_messages.append({"role": role, "content": mixed_content})
        spanish_messages.append({"role": role, "content": content})
        andaluh_messages.append({"role": role, "content": converted})

    if not has_user or not has_assistant:
        return None

    metadata = {
        "source_dataset": "VillanovaAI/villanova-sft-2603",
        "source_data": row.get("source_data", "fixture"),
        "source_license": row.get("source_license") or source_license(row.get("source_data")),
        "category": row.get("category", "Chat"),
        "language": row.get("language", "spa"),
        "system_prompt": system_prompt,
        "target_model": "Qwen-Andaluh",
        "variant": f"EPA_{variant}" + ("_informal" if informal_strength > 0 else ""),
    }
    return (
        {"text": _render_chat(mixed_messages), "metadata": {**metadata, "split_view": "mixed"}},
        {"text": _render_chat(spanish_messages), "metadata": {**metadata, "split_view": "spanish"}},
        {"text": _render_chat(andaluh_messages), "metadata": {**metadata, "split_view": "andaluh"}},
    )


def build(args: argparse.Namespace) -> None:
    needed = args.n_train + args.n_valid + args.n_probe
    raw_rows = _source_rows(args, needed)
    random.Random(args.seed).shuffle(raw_rows)

    mixed_rows: list[dict[str, Any]] = []
    spanish_probe: list[dict[str, Any]] = []
    andaluh_probe: list[dict[str, Any]] = []
    for row_index, row in enumerate(raw_rows):
        variants = _chat_variants(
            row,
            row_index=row_index,
            seed=args.seed,
            user_andaluh_ratio=args.user_andaluh_ratio,
            variant=args.variant,
            informal_strength=args.informal_strength,
            system_prompt=args.system_prompt,
        )
        if variants is None:
            continue
        mixed, spanish, andaluh = variants
        mixed_rows.append(mixed)
        if len(spanish_probe) < args.n_probe:
            spanish_probe.append(spanish)
            andaluh_probe.append(andaluh)

    if len(mixed_rows) < args.n_train + args.n_valid:
        print(
            f"[build_cpt] Warning: requested {args.n_train + args.n_valid} rows but "
            f"only built {len(mixed_rows)}.",
            file=sys.stderr,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = mixed_rows[: args.n_train]
    valid_rows = mixed_rows[args.n_train : args.n_train + args.n_valid]
    write_jsonl(out_dir / "cpt_train.jsonl", train_rows)
    write_jsonl(out_dir / "cpt_valid.jsonl", valid_rows)
    write_jsonl(out_dir / "cpt_spanish_valid.jsonl", spanish_probe)
    write_jsonl(out_dir / "cpt_andaluh_valid.jsonl", andaluh_probe)

    manifest = {
        "dataset": args.dataset,
        "language": args.language,
        "categories": args.categories,
        "allowed_licenses": args.allowed_licenses,
        "n_train": len(train_rows),
        "n_valid": len(valid_rows),
        "n_probe": len(spanish_probe),
        "system_prompt": args.system_prompt,
        "target_model": "Qwen-Andaluh",
    }
    (out_dir / "cpt_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote CPT rows to {out_dir}: train={len(train_rows)}, "
        f"valid={len(valid_rows)}, probe={len(spanish_probe)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="VillanovaAI/villanova-sft-2603")
    parser.add_argument("--language", default="spa")
    parser.add_argument("--categories", nargs="+", default=sorted(ALLOWED_CATEGORIES))
    parser.add_argument("--allowed_licenses", nargs="+", default=sorted(DEFAULT_ALLOWED_LICENSES))
    parser.add_argument("--n_train", type=int, default=2000)
    parser.add_argument("--n_valid", type=int, default=200)
    parser.add_argument("--n_probe", type=int, default=128)
    parser.add_argument("--user_andaluh_ratio", type=float, default=0.5)
    parser.add_argument("--variant", default="seseo")
    parser.add_argument("--informal_strength", type=float, default=0.0)
    parser.add_argument("--system_prompt", default=SYSTEM_PROMPT_BASE)
    parser.add_argument("--out_dir", default="data/processed/base")
    parser.add_argument("--fallback_fixture", default="")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--no_streaming", action="store_true")
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
