"""Filter CPT JSONL rows for prose-like Andaluh adaptation data."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import random
import re
from pathlib import Path
from typing import Any

from marichatmen.eval.quality_metrics import has_generation_artifact
from marichatmen.io import iter_jsonl, write_jsonl

CONTROL_OR_PLACEHOLDER_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f]|\t|NdrFc|Shorth|\*{4,}|1ttransformers|\buvv\b",
    re.IGNORECASE,
)
CHAT_TEMPLATE_RE = re.compile(r"<\|im_(?:start|end)\|>|(?:^|\n)\s*(?:system|user|assistant)\s*\n", re.IGNORECASE)
WIKI_MARKUP_RE = re.compile(
    r"\[\[|\]\]|\{\||\|\}|"
    r"\b(?:thumb|tumb|right|rîtt|left|center|centre|miniatura|miniaturadeimagen|miniaturadeimahen)\s*\||"
    r"\b(?:archivo|file|imagen|category|categor[ií]a)\s*:|"
    r"\b[\dx]{2,9}\s*px\b|"
    r"\b[\dx]{2,9}\s*px\s*\||"
    r"\bpx\s*\|",
    re.IGNORECASE,
)
BAD_MODEL_ARTIFACT_RE = re.compile(
    r"\b(?:iesa|erva|iganh|igann|opan)\b|"
    r"\bAPI\s+API\b|HTTP/HTTP|"
    r"\b20\d{2}-\d{2}-\d{2}\b",
    re.IGNORECASE,
)
NUMERIC_LOOP_RE = re.compile(
    r"(\d)\1{5,}|(?:\b\d{1,3}(?:\s+\d{3}){2,}\b)|(?:\b100\s*%\b.*?){2,}|(?:\b20\d{2}\b.*?){5,}",
    re.IGNORECASE | re.DOTALL,
)
WORD_RE = re.compile(r"[\wûâêîôçáéíóúüñ]+", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
PERCENT_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*%")
YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2})\b")


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def _token_repetition_rate(text: str, *, n: int = 3) -> float:
    tokens = [match.group(0).casefold() for match in WORD_RE.finditer(text)]
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    return 1.0 - (len(set(grams)) / max(1, len(grams)))


def _pipe_markup_fraction(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    pipeish = 0
    for line in lines:
        pipe_count = line.count("|")
        if pipe_count >= 2 or (pipe_count >= 1 and re.search(r"\b(?:thumb|tumb|rîtt|px)\b", line, re.IGNORECASE)):
            pipeish += 1
    return pipeish / len(lines)


def reject_reason(
    text: str,
    *,
    min_words: int,
    max_words: int,
    max_numeric_tokens: int,
    max_percent_tokens: int,
    max_unique_years: int,
    max_numeric_char_fraction: float,
    max_repetition_rate: float,
) -> str | None:
    stripped = text.strip()
    if not stripped:
        return "empty"
    if CONTROL_OR_PLACEHOLDER_RE.search(stripped):
        return "control_or_placeholder"
    if CHAT_TEMPLATE_RE.search(stripped):
        return "chat_template"
    if WIKI_MARKUP_RE.search(stripped) or _pipe_markup_fraction(stripped) > 0.03:
        return "wiki_markup"
    if has_generation_artifact(stripped):
        return "generation_artifact"
    if BAD_MODEL_ARTIFACT_RE.search(stripped):
        return "model_artifact"
    words = _word_count(stripped)
    if words < min_words:
        return "too_short"
    if words > max_words:
        return "too_long"
    if NUMERIC_LOOP_RE.search(stripped):
        return "numeric_loop"
    numeric_tokens = NUMBER_RE.findall(stripped)
    if len(numeric_tokens) > max_numeric_tokens:
        return "too_many_numbers"
    if len(PERCENT_RE.findall(stripped)) > max_percent_tokens:
        return "too_many_percentages"
    if len(set(YEAR_RE.findall(stripped))) > max_unique_years:
        return "too_many_years"
    numeric_chars = sum(char.isdigit() for char in stripped)
    if numeric_chars / max(1, len(stripped)) > max_numeric_char_fraction:
        return "numeric_char_fraction"
    if _token_repetition_rate(stripped) > max_repetition_rate:
        return "high_repetition"
    return None


def filter_rows(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    kept: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}
    seen_split_views: Counter[str] = Counter()
    seen = 0

    for source in args.input:
        for row in iter_jsonl(source):
            seen += 1
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            seen_split_views[str(metadata.get("split_view") or "unknown")] += 1
            text = row.get("text")
            if not isinstance(text, str):
                rejected["missing_text"] = rejected.get("missing_text", 0) + 1
                continue
            reason = reject_reason(
                text,
                min_words=args.min_words,
                max_words=args.max_words,
                max_numeric_tokens=args.max_numeric_tokens,
                max_percent_tokens=args.max_percent_tokens,
                max_unique_years=args.max_unique_years,
                max_numeric_char_fraction=args.max_numeric_char_fraction,
                max_repetition_rate=args.max_repetition_rate,
            )
            if reason is not None:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            kept.append({"text": text, "metadata": row.get("metadata", {})})

    if args.shuffle:
        rng.shuffle(kept)
    if args.max_rows > 0:
        kept = kept[: args.max_rows]

    written_split_views = Counter(
        str((row.get("metadata") if isinstance(row.get("metadata"), dict) else {}).get("split_view") or "unknown")
        for row in kept
    )
    out_path = Path(args.output)
    written = write_jsonl(out_path, kept)
    manifest = {
        "input": args.input,
        "output": str(out_path),
        "seen": seen,
        "written": written,
        "rejected": rejected,
        "seen_split_view_counts": dict(sorted(seen_split_views.items())),
        "written_split_view_counts": dict(sorted(written_split_views.items())),
        "filters": {
            "min_words": args.min_words,
            "max_words": args.max_words,
            "max_numeric_tokens": args.max_numeric_tokens,
            "max_percent_tokens": args.max_percent_tokens,
            "max_unique_years": args.max_unique_years,
            "max_numeric_char_fraction": args.max_numeric_char_fraction,
            "max_repetition_rate": args.max_repetition_rate,
        },
    }
    Path(args.manifest or f"{out_path}.manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--min_words", type=int, default=80)
    parser.add_argument("--max_words", type=int, default=2500)
    parser.add_argument("--max_numeric_tokens", type=int, default=8)
    parser.add_argument("--max_percent_tokens", type=int, default=1)
    parser.add_argument("--max_unique_years", type=int, default=3)
    parser.add_argument("--max_numeric_char_fraction", type=float, default=0.018)
    parser.add_argument("--max_repetition_rate", type=float, default=0.18)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=47)
    return parser.parse_args(argv)


def main() -> None:
    manifest = filter_rows(parse_args())
    print(
        "Filtered CPT rows: "
        f"seen={manifest['seen']} written={manifest['written']} rejected={manifest['rejected']}"
    )


if __name__ == "__main__":
    main()
