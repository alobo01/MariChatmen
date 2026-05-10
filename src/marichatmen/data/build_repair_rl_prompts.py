"""Build neutral prompt-only data for Qwen-Andaluh repair RL."""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Any

from marichatmen.constants import ARTIFACT_ROOT, SYSTEM_PROMPT_BASE
from marichatmen.io import read_jsonl, write_jsonl

FAILURE_PROBES = [
    ("technical_basics", "Explícame qué es el overfitting."),
    ("api_python_tooling", "Resume qué es una API REST."),
    ("study_planning", "¿Cómo puedo organizarme para estudiar un máster sin agobiarme?"),
    ("technical_basics", "No entiendo la validación cruzada, explícamela fácil."),
    ("api_python_tooling", "Dime cómo instalar transformers con uv."),
    ("technical_basics", "¿Qué es LoRA?"),
    ("technical_basics", "¿Qué es ORPO?"),
]

PERSONA_MARKERS = {
    "cruzcampo",
    "sfdk",
    "toteking",
    "feria",
    "triana",
    "macarena",
    "gazpacho",
    "paella",
    "ibiza",
    "málaga",
    "malaga",
    "cádiz",
    "cadiz",
}

PROMPT_LEAK_RE = re.compile(
    r"\b(andal[uúû]h?|epa|seseo|sevillan[oa]s?|çebiyan[oa]s?|dialect|translit|"
    r"responde\s+en|respóndeme\s+en|contesta\s+en|responde\s+siempre|"
    r"contesta\s+siempre)\b",
    re.IGNORECASE,
)


def _neutral_prompt(text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT_BASE},
        {"role": "user", "content": text.strip()},
    ]


def _load_extra_probe_file(path: str) -> list[tuple[str, str]]:
    if not path:
        return []
    file = Path(path)
    if not file.exists():
        return []
    rows = []
    for line in file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not PROMPT_LEAK_RE.search(stripped):
            rows.append(("fixed_probe", stripped))
    return rows


def _user_text_from_sft(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            return content.strip() if isinstance(content, str) else ""
    return ""


def _category_for_prompt(text: str) -> str:
    lower = text.lower()
    if any(token in lower for token in ["uv", "python", "api", "transformers", "lora", "orpo"]):
        return "api_python_tooling"
    if any(token in lower for token in ["validación", "validacion", "overfitting", "modelo", "gradiente"]):
        return "technical_basics"
    if any(token in lower for token in ["resume", "resum", "reescribe", "explica"]):
        return "summarization_rewrite"
    if any(token in lower for token in ["estudi", "máster", "master", "organizar"]):
        return "study_planning"
    return "normal_qa"


def _safe_prompt(text: str, *, max_chars: int) -> bool:
    stripped = text.strip()
    if len(stripped) < 8 or len(stripped) > max_chars:
        return False
    lower = stripped.lower()
    return not any(marker in lower for marker in PERSONA_MARKERS) and not PROMPT_LEAK_RE.search(stripped)


def build(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    prompt_items = list(FAILURE_PROBES)
    prompt_items.extend(_load_extra_probe_file(args.probe_file))

    seen = {text.lower() for _, text in prompt_items}
    if args.sft_train and Path(args.sft_train).exists():
        rows = read_jsonl(args.sft_train)
        rng.shuffle(rows)
        for row in rows:
            text = _user_text_from_sft(row)
            key = text.lower()
            if key in seen or not _safe_prompt(text, max_chars=args.max_prompt_chars):
                continue
            prompt_items.append((_category_for_prompt(text), text))
            seen.add(key)
            if len(prompt_items) >= args.n_prompts:
                break

    prompt_items = prompt_items[: args.n_prompts]
    rows = [
        {
            "id": f"qwen_andaluh_repair_{index:05d}",
            "prompt": _neutral_prompt(text),
            "category": category,
            "target": "neutral_qwen_andaluh_repair",
        }
        for index, (category, text) in enumerate(prompt_items)
    ]
    write_jsonl(args.out_file, rows)
    print(f"Wrote {len(rows)} repair RL prompts to {args.out_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_train", default=str(ARTIFACT_ROOT / "data/processed/base/sft_train.jsonl"))
    parser.add_argument("--probe_file", default="examples/prompts/qwen_andaluh_neutral_probe_v1.txt")
    parser.add_argument("--n_prompts", type=int, default=300)
    parser.add_argument(
        "--out_file",
        default=str(ARTIFACT_ROOT / "data/processed/qwen_andaluh_repair_rl/prompts.jsonl"),
    )
    parser.add_argument("--max_prompt_chars", type=int, default=700)
    parser.add_argument("--seed", type=int, default=47)
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
