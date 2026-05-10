"""Create deterministic JSONL subsets by approximate tokenizer-token budget."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from marichatmen.io import iter_jsonl, write_jsonl
from marichatmen.tokenizer_templates import ensure_text_training_chat_template


def _load_tokenizer(name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    ensure_text_training_chat_template(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _chat_count(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    return len(ids)


def _row_token_count(tokenizer: Any, row: dict[str, Any], schema: str) -> int:
    if schema == "cpt":
        text = row.get("text")
        if not isinstance(text, str):
            return 0
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    if schema == "sft":
        messages = row.get("messages")
        if not isinstance(messages, list):
            return 0
        return _chat_count(tokenizer, messages)
    if schema == "orpo":
        prompt = row.get("prompt")
        chosen = row.get("chosen")
        rejected = row.get("rejected")
        if not isinstance(prompt, list) or not isinstance(chosen, list) or not isinstance(rejected, list):
            return 0
        return _chat_count(tokenizer, prompt + chosen) + _chat_count(tokenizer, prompt + rejected)
    raise ValueError(f"Unknown schema: {schema}")


def run(args: argparse.Namespace) -> None:
    tokenizer = _load_tokenizer(args.tokenizer_name)
    rows = list(iter_jsonl(args.input_file))
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(rows)

    selected: list[dict[str, Any]] = []
    total_tokens = 0
    skipped = 0
    for row in rows:
        row_tokens = _row_token_count(tokenizer, row, args.schema)
        if row_tokens <= 0:
            skipped += 1
            continue
        if args.max_rows and len(selected) >= args.max_rows:
            break
        if args.max_tokens and selected and total_tokens + row_tokens > args.max_tokens:
            break
        row = dict(row)
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        row["metadata"] = {
            **metadata,
            "token_budget_schema": args.schema,
            "estimated_training_tokens": row_tokens,
        }
        selected.append(row)
        total_tokens += row_tokens
        if args.max_tokens and total_tokens >= args.max_tokens:
            break

    write_jsonl(args.output_file, selected)
    manifest = {
        "input_file": args.input_file,
        "output_file": args.output_file,
        "tokenizer_name": args.tokenizer_name,
        "schema": args.schema,
        "requested_max_tokens": args.max_tokens,
        "requested_max_rows": args.max_rows,
        "selected_rows": len(selected),
        "estimated_training_tokens": total_tokens,
        "skipped_rows": skipped,
        "shuffle": args.shuffle,
        "seed": args.seed,
    }
    manifest_path = Path(args.manifest_file or f"{args.output_file}.manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(selected)} rows / {total_tokens} estimated {args.schema} tokens "
        f"to {args.output_file}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--tokenizer_name", required=True)
    parser.add_argument("--schema", choices=["cpt", "sft", "orpo"], required=True)
    parser.add_argument("--max_tokens", type=int, default=0)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--manifest_file", default="")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=52)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
