"""Measure and improve tokenizer efficiency for Qwen-Andaluh."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from marichatmen.constants import ALLOWED_CATEGORIES, ARTIFACT_ROOT, DEFAULT_ALLOWED_LICENSES
from marichatmen.data.license_filter import normalize_license
from marichatmen.data.load_villanova import iter_villanova_examples
from marichatmen.data.transliterate_andaluh import strip_thinking, to_andaluh
from marichatmen.io import iter_jsonl
from marichatmen.tokenizer_templates import ensure_text_training_chat_template

WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñÇçÂÊÎÔÛâêîôû]{2,32}")
ANDALUH_MARKER_RE = re.compile(r"[ÇçÂÊÎÔÛâêîôû]")
COMMON_ANDALUH_FORMS = {
    "andalûh",
    "andaluh",
    "çebiya",
    "çebiyano",
    "çebiyana",
    "ehtá",
    "ehtán",
    "ehtoy",
    "ehtamoh",
    "ehto",
    "ehta",
    "er",
    "loh",
    "lah",
    "pa",
    "mu",
    "muh",
    "tó",
    "toa",
    "toah",
    "cansao",
    "demasiao",
    "sentío",
    "perdío",
    "venío",
    "salío",
}


def _token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _local_jsonl_files(dataset: str) -> list[Path]:
    if any(char in dataset for char in "*?[]"):
        return [Path(item) for item in sorted(glob.glob(dataset))]
    path = Path(dataset)
    if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}:
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.jsonl"))
    return []


def _iter_local_text_pairs(args: argparse.Namespace) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for file_path in _local_jsonl_files(args.dataset):
        for row_index, row in enumerate(iter_jsonl(file_path)):
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            language = row.get("language") or metadata.get("language")
            if language and language != args.language:
                continue
            text = row.get("text")
            if not isinstance(text, str):
                continue
            spanish = strip_thinking(text)
            if len(spanish) < 24:
                continue
            andaluh = to_andaluh(
                spanish,
                variant=args.variant,
                informal_strength=args.informal_strength,
                seed=args.seed + len(pairs) * 1009 + row_index,
            )
            pairs.append((spanish, andaluh))
            if len(pairs) >= args.n_texts:
                return pairs
    return pairs


def _iter_text_pairs(args: argparse.Namespace) -> list[tuple[str, str]]:
    local_pairs = _iter_local_text_pairs(args)
    if local_pairs:
        return local_pairs

    allowed = {normalize_license(item) for item in args.allowed_licenses}
    pairs: list[tuple[str, str]] = []
    for row in iter_villanova_examples(
        args.dataset,
        language=args.language,
        categories=set(args.categories),
        allowed_licenses=allowed,
        streaming=not args.no_streaming,
    ):
        messages = row.get("messages")
        if not isinstance(messages, list):
            continue
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if message.get("role") == "system" or not isinstance(content, str):
                continue
            spanish = strip_thinking(content)
            if len(spanish) < 24:
                continue
            andaluh = to_andaluh(
                spanish,
                variant=args.variant,
                informal_strength=args.informal_strength,
                seed=args.seed + len(pairs) * 1009 + index,
            )
            pairs.append((spanish, andaluh))
            if len(pairs) >= args.n_texts:
                return pairs
    return pairs


def _candidate_token_stats(tokenizer: Any, andaluh_texts: list[str], limit: int) -> list[dict[str, Any]]:
    vocab = tokenizer.get_vocab()
    counts: Counter[str] = Counter()
    for text in andaluh_texts:
        counts.update(match.group(0).lower() for match in WORD_RE.finditer(text))

    scored: list[tuple[int, int, str, int]] = []
    for word, freq in counts.items():
        if not _is_valid_added_token(word):
            continue
        if word in vocab or word.isdigit():
            continue
        if not _looks_andaluh(word):
            continue
        pieces = _token_count(tokenizer, word)
        if pieces <= 1:
            continue
        score = (pieces - 1) * freq
        scored.append((score, freq, word, pieces))
    scored.sort(reverse=True)
    return [
        {
            "token": word,
            "frequency": freq,
            "base_token_count": pieces,
            "saving_score": score,
        }
        for score, freq, word, pieces in scored[:limit]
    ]


def _candidate_tokens(tokenizer: Any, andaluh_texts: list[str], limit: int) -> list[str]:
    return [row["token"] for row in _candidate_token_stats(tokenizer, andaluh_texts, limit)]


def _is_valid_added_token(token: str) -> bool:
    if "\ufffd" in token:
        return False
    if any((ord(char) < 32 and char not in {"\t", "\n"}) for char in token):
        return False
    return True


def _looks_andaluh(word: str) -> bool:
    if ANDALUH_MARKER_RE.search(word):
        return True
    if word in COMMON_ANDALUH_FORMS:
        return True
    return bool(re.search(r"(ao|ío|ía|âh|êh|ôh|ûh)$", word))


def _summary(tokenizer: Any, pairs: list[tuple[str, str]]) -> dict[str, float]:
    spanish_tokens = 0
    andaluh_tokens = 0
    spanish_chars = 0
    andaluh_chars = 0
    for spanish, andaluh in pairs:
        spanish_tokens += _token_count(tokenizer, spanish)
        andaluh_tokens += _token_count(tokenizer, andaluh)
        spanish_chars += len(spanish)
        andaluh_chars += len(andaluh)
    return {
        "examples": float(len(pairs)),
        "spanish_tokens": float(spanish_tokens),
        "andaluh_tokens": float(andaluh_tokens),
        "spanish_chars": float(spanish_chars),
        "andaluh_chars": float(andaluh_chars),
        "spanish_tokens_per_1k_chars": spanish_tokens * 1000 / max(1, spanish_chars),
        "andaluh_tokens_per_1k_chars": andaluh_tokens * 1000 / max(1, andaluh_chars),
        "andaluh_token_overhead_pct": 100 * (andaluh_tokens / max(1, spanish_tokens) - 1),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bar_svg(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bars = [
        ("Spanish base", summary["base"]["spanish_tokens_per_1k_chars"], "#2563eb"),
        ("Andaluh base", summary["base"]["andaluh_tokens_per_1k_chars"], "#dc2626"),
        ("Andaluh expanded", summary["expanded"]["andaluh_tokens_per_1k_chars"], "#16a34a"),
    ]
    width, height = 920, 420
    left, top, right, bottom = 72, 62, 32, 76
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_y = max(value for _, value, _ in bars) * 1.15
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="34" font-size="24" font-weight="700">Qwen-Andaluh Tokenizer Impact</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222"/>',
    ]
    for idx in range(6):
        value = idx / 5 * max_y
        y = top + plot_h - value / max_y * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="12" y="{y + 4:.1f}" font-size="12" fill="#555">{value:.0f}</text>')
    gap = 44
    bar_w = (plot_w - gap * (len(bars) + 1)) / len(bars)
    for idx, (label, value, color) in enumerate(bars):
        x = left + gap + idx * (bar_w + gap)
        h = value / max_y * plot_h
        y = top + plot_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}" rx="5"/>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-size="13">{value:.1f}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{height - 38}" text-anchor="middle" font-size="13">{label}</text>')
    reduction = summary["expanded"]["andaluh_reduction_pct"]
    parts.append(f'<text x="{left}" y="{height - 12}" font-size="13">Expanded-token reduction on Andaluh text: {reduction:.2f}%</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _write_markdown_report(path: Path, summary: dict[str, Any], pairs: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = summary["base"]
    expanded = summary["expanded"]
    top_tokens = summary["top_added_token_stats"][:20]
    lines = [
        "# Qwen-Andaluh Tokenizer Audit",
        "",
        "This report explains tokenizer expansion with concrete examples rather than only totals.",
        "",
        "## Summary",
        "",
        f"- Source examples audited: {summary['examples']}",
        f"- Variant: `{summary['variant']}`; VAF output favours `ç` for the Sevillian target.",
        f"- Tokenizer mode: `{summary['tokenizer_mode']}`",
        f"- Added tokens: {summary['added_tokens']}",
        f"- Andaluh token overhead before expansion: {base['andaluh_token_overhead_pct']:.2f}%",
        f"- Andaluh token reduction after expansion: {expanded['andaluh_reduction_pct']:.2f}%",
        "",
        "## Example Transformations",
        "",
    ]
    for spanish, andaluh in pairs[:5]:
        lines.extend(
            [
                "```text",
                f"Spanish: {spanish[:240]}",
                f"Andaluh: {andaluh[:240]}",
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Most Useful Added Tokens",
            "",
            "| Token | Frequency | Base pieces | Saving score |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in top_tokens:
        lines.append(
            f"| `{row['token']}` | {row['frequency']} | "
            f"{row['base_token_count']} | {row['saving_score']} |"
        )
    lines.extend(
        [
            "",
            "The saving score is `frequency * (base_token_count - 1)`. A high score means "
            "the old tokenizer repeatedly split that Andaluh form into several pieces.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    base_tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    ensure_text_training_chat_template(base_tokenizer)
    pairs = _iter_text_pairs(args)
    if not pairs:
        raise RuntimeError("No Spanish/Andaluh text pairs were built for tokenizer analysis.")

    andaluh_texts = [andaluh for _, andaluh in pairs]
    candidate_stats = _candidate_token_stats(base_tokenizer, andaluh_texts, args.new_tokens)
    candidates = [row["token"] for row in candidate_stats]
    if args.mode == "expand":
        adapted_tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        ensure_text_training_chat_template(adapted_tokenizer)
        added_count = adapted_tokenizer.add_tokens(candidates)
        vocab_delta = added_count
    else:
        retrain_vocab_size = args.retrain_vocab_size or len(base_tokenizer)
        adapted_tokenizer = base_tokenizer.train_new_from_iterator(
            andaluh_texts,
            vocab_size=retrain_vocab_size,
        )
        ensure_text_training_chat_template(adapted_tokenizer)
        added_count = 0
        vocab_delta = len(adapted_tokenizer) - len(base_tokenizer)
    save_dir = Path(args.save_tokenizer_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    adapted_tokenizer.save_pretrained(save_dir)
    (save_dir / "added_tokens_qwen_andaluh.txt").write_text(
        "\n".join(candidates) + "\n",
        encoding="utf-8",
    )
    if args.mode == "retrain":
        (save_dir / "TOKENIZER_COMPATIBILITY_WARNING.md").write_text(
            "# Tokenizer Compatibility Warning\n\n"
            "This tokenizer was retrained from text and does not preserve the original "
            "Qwen token-id semantics. It is useful for analysis or a from-scratch "
            "embedding adaptation experiment, but it should not be used for normal "
            "LoRA/QLoRA adapter release unless the model embeddings have been trained "
            "accordingly.\n",
            encoding="utf-8",
        )

    base_summary = _summary(base_tokenizer, pairs)
    expanded_summary = _summary(adapted_tokenizer, pairs)
    expanded_summary["andaluh_reduction_pct"] = 100 * (
        1 - expanded_summary["andaluh_tokens"] / max(1, base_summary["andaluh_tokens"])
    )
    summary = {
        "model_name": args.model_name,
        "dataset": args.dataset,
        "tokenizer_mode": args.mode,
        "examples": len(pairs),
        "requested_new_tokens": args.new_tokens,
        "added_tokens": added_count,
        "vocab_delta": vocab_delta,
        "adapted_vocab_size": len(adapted_tokenizer),
        "retrain_vocab_size": (args.retrain_vocab_size or len(base_tokenizer)) if args.mode == "retrain" else None,
        "save_tokenizer_dir": str(save_dir),
        "base": base_summary,
        "expanded": expanded_summary,
        "variant": args.variant,
        "top_added_tokens": candidates[:50],
        "top_added_token_stats": candidate_stats[:100],
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rows = []
    for key in ["spanish_tokens_per_1k_chars", "andaluh_tokens_per_1k_chars"]:
        rows.append({"tokenizer": "base", "metric": key, "value": base_summary[key]})
        rows.append({"tokenizer": "expanded", "metric": key, "value": expanded_summary[key]})
    rows.append(
        {
            "tokenizer": "expanded",
            "metric": "andaluh_reduction_pct",
            "value": expanded_summary["andaluh_reduction_pct"],
        }
    )
    _write_csv(Path(args.output_csv), rows)
    _bar_svg(Path(args.plot_svg), summary)
    if args.output_md:
        _write_markdown_report(Path(args.output_md), summary, pairs)
    if args.mode == "expand":
        action = f"after adding {added_count} Qwen-compatible tokens"
    else:
        action = (
            f"with a retrained tokenizer "
            f"(vocab_delta={vocab_delta}, adapted_vocab_size={len(adapted_tokenizer)})"
        )
    print(
        "Tokenizer impact: "
        f"Andaluh overhead={base_summary['andaluh_token_overhead_pct']:.2f}% before, "
        f"reduction={expanded_summary['andaluh_reduction_pct']:.2f}% {action}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--dataset", default="VillanovaAI/villanova-sft-2603")
    parser.add_argument("--language", default="spa")
    parser.add_argument("--categories", nargs="+", default=sorted(ALLOWED_CATEGORIES))
    parser.add_argument("--allowed_licenses", nargs="+", default=sorted(DEFAULT_ALLOWED_LICENSES))
    parser.add_argument("--n_texts", type=int, default=2000)
    parser.add_argument("--new_tokens", type=int, default=1536)
    parser.add_argument("--mode", choices=["expand", "retrain"], default="expand")
    parser.add_argument("--retrain_vocab_size", type=int, default=0)
    parser.add_argument("--variant", default="sevillian_ce")
    parser.add_argument("--informal_strength", type=float, default=0.0)
    parser.add_argument(
        "--save_tokenizer_dir",
        default=str(ARTIFACT_ROOT / "outputs/tokenizers/qwen35_08b_andaluh"),
    )
    parser.add_argument(
        "--output_json",
        default=str(ARTIFACT_ROOT / "reports/tokenizer/qwen_andaluh_tokenizer_impact.json"),
    )
    parser.add_argument(
        "--output_csv",
        default=str(ARTIFACT_ROOT / "reports/tokenizer/qwen_andaluh_tokenizer_impact.csv"),
    )
    parser.add_argument(
        "--plot_svg",
        default=str(ARTIFACT_ROOT / "reports/plots/qwen_andaluh_tokenizer_impact.svg"),
    )
    parser.add_argument(
        "--output_md",
        default=str(ARTIFACT_ROOT / "reports/tokenizer/qwen_andaluh_tokenizer_impact.md"),
    )
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--no_streaming", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
