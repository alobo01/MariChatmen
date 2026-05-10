"""Patch SFT rows with neutral prompts, technical gold data, and preamble filtering."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from marichatmen.constants import SYSTEM_PROMPT_BASE
from marichatmen.data.transliterate_andaluh import to_andaluh
from marichatmen.eval.quality_metrics import has_generation_artifact, has_reasoning_preamble
from marichatmen.io import read_jsonl, write_jsonl

EXPLICIT_PROMPT = (
    "Eres un asistente útil que responde siempre en Andalûh EPA con seseo "
    "sevillano informal. El usuario puede escribir en español estándar o "
    "en Andalûh, pero tú respondes siempre en Andalûh con claridad y precisión."
)

BAD_SHORT_ANSWERS = {
    "- ¿ qué?",
    "- ¿qué?",
    "¿ qué êh eso?",
    "¿qué êh eso?",
    "¿ qué es eso?",
    "¿qué es eso?",
    "sí, por supuêtto.",
    "sí, por supuesto.",
    "no, no.",
}

CONTROL_OR_PLACEHOLDER = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f]|\t|NdrFc|Shorth|\*{4,}|1ttransformers|\buvv\b",
    re.IGNORECASE,
)
BAD_MODEL_ARTIFACT = re.compile(
    r"\b(?:iesa|erva|iganh|igann|opan)\b|"
    r"\bAPI\s+API\b|HTTP/HTTP|"
    r"\b20\d{2}-\d{2}-\d{2}\b|"
    r"(?:^|\n)\s*(?:dk|writer|wrapper)\s*(?:\n|$)|"
    r"<\|im_(?:start|end)\|>",
    re.IGNORECASE,
)
WIKI_MARKUP_ARTIFACT = re.compile(
    r"\[\[|\]\]|\{\||\|\}|"
    r"\b(?:thumb|tumb|right|rîtt|left|center|centre|miniatura|miniaturadeimagen|miniaturadeimahen)\s*\||"
    r"\b(?:archivo|file|imagen|category|categor[ií]a)\s*:|"
    r"\b[\dx]{2,9}\s*px\b|"
    r"\b[\dx]{2,9}\s*px\s*\||"
    r"\bpx\s*\|",
    re.IGNORECASE,
)
NUMERIC_OR_LOOP_ARTIFACT = re.compile(
    r"(\d)\1{5,}|"
    r"(?:\b\d{1,3}(?:\s+\d{3}){2,}\b)|"
    r"(?:\b100\s*%\b.*?){2,}|"
    r"(?:\b20\d{2}\b.*?){5,}",
    re.IGNORECASE | re.DOTALL,
)


def _assistant_text(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if (
            isinstance(message, dict)
            and message.get("role") == "assistant"
            and isinstance(message.get("content"), str)
        ):
            return message["content"]
    return ""


def _normalise_text(text: str) -> str:
    return " ".join(text.strip().split()).casefold()


def _token_repetition_rate(text: str, *, n: int = 3) -> float:
    tokens = re.findall(r"[\wûâêîôçáéíóúüñ]+|`[^`]+`", text.casefold())
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    return 1.0 - (len(set(grams)) / max(1, len(grams)))


def _has_numeric_artifact(text: str) -> bool:
    if NUMERIC_OR_LOOP_ARTIFACT.search(text):
        return True
    numeric_tokens = re.findall(r"\b\d+(?:[.,]\d+)?\b", text)
    percent_tokens = re.findall(r"\b\d+(?:[.,]\d+)?\s*%", text)
    years = re.findall(r"\b20\d{2}\b", text)
    return len(numeric_tokens) > 12 or len(percent_tokens) > 3 or len(set(years)) > 4


def _has_bad_answer_shape(text: str) -> bool:
    normalised = _normalise_text(text)
    if not normalised:
        return True
    if CONTROL_OR_PLACEHOLDER.search(text):
        return True
    if has_generation_artifact(text):
        return True
    if BAD_MODEL_ARTIFACT.search(text) or WIKI_MARKUP_ARTIFACT.search(text):
        return True
    if _has_numeric_artifact(text):
        return True
    if _token_repetition_rate(text) > 0.22:
        return True
    if normalised in BAD_SHORT_ANSWERS:
        return True
    return False


def _message_texts(row: dict[str, Any]) -> list[str]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return []
    return [
        message["content"]
        for message in messages
        if isinstance(message, dict)
        and message.get("role") in {"system", "user", "assistant"}
        and isinstance(message.get("content"), str)
    ]


def _has_bad_message_artifact(row: dict[str, Any]) -> bool:
    """Reject rows where artifacts are present anywhere in the chat context."""
    return any(_has_bad_answer_shape(text) for text in _message_texts(row))


def _patch_system(row: dict[str, Any], mode: str) -> dict[str, Any] | None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    cleaned = [
        {"role": message.get("role"), "content": message.get("content")}
        for message in messages
        if isinstance(message, dict)
        and message.get("role") in {"system", "user", "assistant"}
        and isinstance(message.get("content"), str)
    ]
    if any(message["role"] == "assistant" and has_reasoning_preamble(message["content"]) for message in cleaned):
        return None
    non_system = [message for message in cleaned if message["role"] != "system"]
    if not any(message["role"] == "user" for message in non_system):
        return None
    if not any(message["role"] == "assistant" for message in non_system):
        return None

    if mode == "empty":
        patched_messages = non_system
        prompt_text = ""
    elif mode == "explicit":
        patched_messages = [{"role": "system", "content": EXPLICIT_PROMPT}] + non_system
        prompt_text = EXPLICIT_PROMPT
    else:
        patched_messages = [{"role": "system", "content": SYSTEM_PROMPT_BASE}] + non_system
        prompt_text = SYSTEM_PROMPT_BASE

    metadata = dict(row.get("metadata") or {})
    metadata["system_prompt_mode"] = mode
    metadata["system_prompt"] = prompt_text
    metadata["preamble_filtered"] = True
    return {"messages": patched_messages, "metadata": metadata}


def _user_texts(row: dict[str, Any]) -> list[str]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return []
    return [
        message["content"]
        for message in messages
        if isinstance(message, dict)
        and message.get("role") == "user"
        and isinstance(message.get("content"), str)
    ]


def _looks_andaluh(text: str) -> bool:
    lowered = text.casefold()
    return any(marker in lowered for marker in ("ç", "â", "ê", "î", "ô", "û", "andalûh", "eht", " pa "))


def _user_language(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    language = str(metadata.get("user_language") or "").casefold()
    if language in {"andaluh", "spanish"}:
        return language
    texts = _user_texts(row)
    return "andaluh" if texts and any(_looks_andaluh(text) for text in texts) else "spanish"


def _set_user_language(row: dict[str, Any], *, language: str, variant: str, informal_strength: float, seed: int) -> dict[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return row
    patched_messages: list[dict[str, str]] = []
    user_turns = 0
    user_andaluh_turns = 0
    for turn_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            continue
        if role == "user":
            user_turns += 1
            if language == "andaluh":
                content = to_andaluh(
                    content,
                    variant=variant,
                    informal_strength=informal_strength,
                    seed=seed + turn_index,
                )
                user_andaluh_turns += 1
        patched_messages.append({"role": role, "content": content})
    metadata = dict(row.get("metadata") or {})
    metadata["user_turns"] = user_turns
    metadata["user_andaluh_turns"] = user_andaluh_turns
    metadata["user_language"] = language
    metadata["user_language_rebalanced"] = True
    return {"messages": patched_messages, "metadata": metadata}


def _rebalance_user_language(
    rows: list[dict[str, Any]],
    *,
    target_andaluh_ratio: float,
    seed: int,
    variant: str,
    informal_strength: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if target_andaluh_ratio < 0:
        return rows, {"enabled": False}
    rng = random.Random(seed)
    output = list(rows)
    languages = [_user_language(row) for row in output]
    target = round(len(output) * target_andaluh_ratio)
    current = sum(1 for language in languages if language == "andaluh")
    converted = 0
    if current < target:
        candidates = [index for index, language in enumerate(languages) if language == "spanish"]
        rng.shuffle(candidates)
        for index in candidates[: target - current]:
            output[index] = _set_user_language(
                output[index],
                language="andaluh",
                variant=variant,
                informal_strength=informal_strength,
                seed=seed + index * 4099,
            )
            converted += 1
        current += converted
    return output, {
        "enabled": True,
        "target_user_andaluh_ratio": target_andaluh_ratio,
        "rows": len(output),
        "target_andaluh_rows": target,
        "final_andaluh_rows": current,
        "converted_spanish_user_rows": converted,
    }


def _regularize(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    max_exact_assistant_repeats: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    output: list[dict[str, Any]] = []
    seen_assistant: dict[str, int] = {}
    for row in rows:
        assistant = _assistant_text(row)
        if _has_bad_message_artifact(row):
            continue
        assistant_key = _normalise_text(assistant)
        repeat_count = seen_assistant.get(assistant_key, 0)
        if repeat_count >= max_exact_assistant_repeats:
            continue
        seen_assistant[assistant_key] = repeat_count + 1
        draw = rng.random()
        if draw < 0.30:
            mode = "empty"
        elif draw < 0.80:
            mode = "neutral"
        else:
            mode = "explicit"
        patched = _patch_system(row, mode)
        if patched is not None:
            output.append(patched)
    return output


def _load_optional(path: str) -> list[dict[str, Any]]:
    if not path:
        return []
    target = Path(path)
    return read_jsonl(target) if target.is_file() else []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_train", required=True)
    parser.add_argument("--sft_valid", required=True)
    parser.add_argument("--technical_train", default="")
    parser.add_argument("--technical_valid", default="")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--max_exact_assistant_repeats", type=int, default=8)
    parser.add_argument("--target_user_andaluh_ratio", type=float, default=0.3)
    parser.add_argument("--variant", default="sevillian_ce")
    parser.add_argument("--user_informal_strength", type=float, default=0.0)
    args = parser.parse_args()

    train = _regularize(
        read_jsonl(args.sft_train),
        seed=args.seed,
        max_exact_assistant_repeats=args.max_exact_assistant_repeats,
    )
    valid = _regularize(
        read_jsonl(args.sft_valid),
        seed=args.seed + 1,
        max_exact_assistant_repeats=args.max_exact_assistant_repeats,
    )
    train += _regularize(
        _load_optional(args.technical_train),
        seed=args.seed + 3,
        max_exact_assistant_repeats=args.max_exact_assistant_repeats,
    )
    valid += _regularize(
        _load_optional(args.technical_valid),
        seed=args.seed + 4,
        max_exact_assistant_repeats=args.max_exact_assistant_repeats,
    )
    train, train_audit = _rebalance_user_language(
        train,
        target_andaluh_ratio=args.target_user_andaluh_ratio,
        seed=args.seed + 5,
        variant=args.variant,
        informal_strength=args.user_informal_strength,
    )
    valid, valid_audit = _rebalance_user_language(
        valid,
        target_andaluh_ratio=args.target_user_andaluh_ratio,
        seed=args.seed + 6,
        variant=args.variant,
        informal_strength=args.user_informal_strength,
    )

    rng = random.Random(args.seed + 2)
    rng.shuffle(train)
    rng.shuffle(valid)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "sft_train.jsonl", train)
    write_jsonl(out_dir / "sft_valid.jsonl", valid)
    (out_dir / "sft_regularization_audit.json").write_text(
        json.dumps(
            {
                "train": train_audit,
                "valid": valid_audit,
                "system_prompt_distribution": {"empty": 0.30, "neutral": 0.50, "explicit": 0.20},
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote regularized SFT train={len(train)} valid={len(valid)} to {out_dir}")


if __name__ == "__main__":
    main()
