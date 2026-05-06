"""Measure and improve tokenizer efficiency for Qwen-Andaluh."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from marichatmen.constants import ALLOWED_CATEGORIES, DEFAULT_ALLOWED_LICENSES
from marichatmen.data.license_filter import normalize_license
from marichatmen.data.load_villanova import iter_villanova_examples
from marichatmen.data.transliterate_andaluh import strip_thinking, to_andaluh

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


def _iter_text_pairs(args: argparse.Namespace) -> list[tuple[str, str]]:
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


def _candidate_tokens(tokenizer: Any, andaluh_texts: list[str], limit: int) -> list[str]:
    vocab = tokenizer.get_vocab()
    counts: Counter[str] = Counter()
    for text in andaluh_texts:
        counts.update(match.group(0).lower() for match in WORD_RE.finditer(text))

    scored: list[tuple[int, int, str]] = []
    for word, freq in counts.items():
        if word in vocab or word.isdigit():
            continue
        if not _looks_andaluh(word):
            continue
        pieces = _token_count(tokenizer, word)
        if pieces <= 1:
            continue
        score = (pieces - 1) * freq
        scored.append((score, freq, word))
    scored.sort(reverse=True)
    return [word for _, _, word in scored[:limit]]


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


def run(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    base_tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    pairs = _iter_text_pairs(args)
    if not pairs:
        raise RuntimeError("No Spanish/Andaluh text pairs were built for tokenizer analysis.")

    andaluh_texts = [andaluh for _, andaluh in pairs]
    candidates = _candidate_tokens(base_tokenizer, andaluh_texts, args.new_tokens)
    if args.mode == "expand":
        adapted_tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        added_count = adapted_tokenizer.add_tokens(candidates)
        vocab_delta = added_count
    else:
        retrain_vocab_size = args.retrain_vocab_size or len(base_tokenizer)
        adapted_tokenizer = base_tokenizer.train_new_from_iterator(
            andaluh_texts,
            vocab_size=retrain_vocab_size,
        )
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
        "top_added_tokens": candidates[:50],
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
    parser.add_argument("--new_tokens", type=int, default=256)
    parser.add_argument("--mode", choices=["expand", "retrain"], default="expand")
    parser.add_argument("--retrain_vocab_size", type=int, default=0)
    parser.add_argument("--variant", default="seseo")
    parser.add_argument("--informal_strength", type=float, default=0.0)
    parser.add_argument("--save_tokenizer_dir", default="outputs/tokenizers/qwen35_08b_andaluh")
    parser.add_argument("--output_json", default="reports/tokenizer/qwen_andaluh_tokenizer_impact.json")
    parser.add_argument("--output_csv", default="reports/tokenizer/qwen_andaluh_tokenizer_impact.csv")
    parser.add_argument("--plot_svg", default="reports/plots/qwen_andaluh_tokenizer_impact.svg")
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--no_streaming", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
