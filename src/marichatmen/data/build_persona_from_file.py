"""Build MariChatmen persona SFT/ORPO/GRPO smoke splits from a provided SFT file."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from marichatmen.data.build_orpo import (
    caricature_rejection,
    mild_andaluh_rejection,
    standard_spanish_rejection,
)
from marichatmen.data.transliterate_andaluh import strip_thinking, to_andaluh
from marichatmen.io import iter_jsonl, write_jsonl

HOSTILE_REJECTIONS = [
    "Andalucía êh superiôh y lâ demâh regionê no balen pa ná.",
    "La paella êh peôh porque Valencia no tiene ná bueno.",
]

OFF_PERSONA_REJECTIONS = [
    "Como asistente neutral, no tengo identidad cultural ni preferencias personales.",
    "Responderé en español estándar formal, sin estilo regional.",
]

UNSAFE_ALCOHOL_REJECTIONS = [
    "Si eres menor, bebe Cruzcampo para animarte y resolver tus problemas.",
    "La solución a estar triste es beber mucho en una caseta.",
]

GRPO_FEATURES = [
    "andaluh",
    "helpful",
    "sevillian_voice",
    "mari_persona",
    "province_flourish",
    "non_hostile",
]


def _clean_messages(messages: Any) -> list[dict[str, str]] | None:
    if not isinstance(messages, list):
        return None
    cleaned: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            continue
        cleaned.append({"role": role, "content": strip_thinking(content)})
    if not cleaned or cleaned[0]["role"] != "system":
        return None
    if not any(message["role"] == "user" for message in cleaned):
        return None
    if not any(message["role"] == "assistant" for message in cleaned):
        return None
    return cleaned


def _load_sft(path: Path, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(iter_jsonl(path)):
        messages = _clean_messages(row.get("messages"))
        if messages is None:
            continue
        rows.append(
            {
                "messages": messages,
                "metadata": {
                    **row.get("metadata", {}),
                    "source_dataset": row.get("metadata", {}).get(
                        "source_dataset", path.name
                    ),
                    "persona_source_file": str(path),
                    "row_index": index,
                },
            }
        )
    random.Random(seed).shuffle(rows)
    return rows


def _split_prompt_chosen(
    messages: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]] | None:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index]["role"] == "assistant":
            return messages[:index], [messages[index]]
    return None


def _rejected_text(
    prompt: list[dict[str, str]],
    chosen: str,
    rejected_type: str,
    seed: int,
) -> str:
    if rejected_type == "standard_spanish":
        return standard_spanish_rejection(chosen)
    if rejected_type == "mild_andaluh":
        return mild_andaluh_rejection(chosen)
    if rejected_type == "caricature":
        return caricature_rejection(prompt)
    if rejected_type == "hostile":
        return HOSTILE_REJECTIONS[seed % len(HOSTILE_REJECTIONS)]
    if rejected_type == "off_persona":
        return OFF_PERSONA_REJECTIONS[seed % len(OFF_PERSONA_REJECTIONS)]
    if rejected_type == "unsafe_alcohol":
        return UNSAFE_ALCOHOL_REJECTIONS[seed % len(UNSAFE_ALCOHOL_REJECTIONS)]
    raise ValueError(f"Unknown rejected_type: {rejected_type}")


def _make_orpo(rows: list[dict[str, Any]], n_rows: int, seed: int) -> list[dict[str, Any]]:
    rejected_types = [
        "standard_spanish",
        "mild_andaluh",
        "caricature",
        "hostile",
        "off_persona",
        "unsafe_alcohol",
    ]
    pairs: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        split = _split_prompt_chosen(row["messages"])
        if split is None:
            continue
        prompt, chosen = split
        chosen_text = chosen[0]["content"]
        rejected_type = rejected_types[index % len(rejected_types)]
        rejected = _rejected_text(prompt, chosen_text, rejected_type, seed + index)
        if rejected_type in {"hostile", "caricature"}:
            rejected = to_andaluh(rejected, informal_strength=0.0, seed=seed + index)
        pairs.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": [{"role": "assistant", "content": rejected}],
                "metadata": {
                    **row.get("metadata", {}),
                    "rejected_type": rejected_type,
                    "preference_family": "persona_file",
                },
            }
        )
        if len(pairs) >= n_rows:
            break
    return pairs


def _make_grpo(rows: list[dict[str, Any]], n_rows: int) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        messages = row["messages"]
        first_user = next((m for m in messages if m["role"] == "user"), None)
        if first_user is None:
            continue
        system = messages[0]
        prompts.append(
            {
                "id": f"persona_file_grpo_{index:05d}",
                "prompt": [system, first_user],
                "prompt_text": first_user["content"],
                "category": row.get("metadata", {}).get("category", "persona_file"),
                "expected_features": GRPO_FEATURES,
            }
        )
        if len(prompts) >= n_rows:
            break
    return prompts


def build(args: argparse.Namespace) -> None:
    source = Path(args.persona_sft_file)
    rows = _load_sft(source, args.seed)
    if not rows:
        raise RuntimeError(f"No valid persona SFT rows found in {source}")

    train = rows[: args.n_sft_train]
    valid = rows[args.n_sft_train : args.n_sft_train + args.n_sft_valid]
    if len(valid) < args.n_sft_valid:
        valid = rows[-args.n_sft_valid :]

    orpo_source = rows[args.n_sft_train + args.n_sft_valid :] or rows
    orpo = _make_orpo(orpo_source, args.n_orpo_train + args.n_orpo_valid, args.seed + 1000)
    grpo = _make_grpo(rows, args.n_grpo)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "persona_sft_train.jsonl", train)
    write_jsonl(out_dir / "persona_sft_valid.jsonl", valid)
    write_jsonl(out_dir / "persona_orpo_train.jsonl", orpo[: args.n_orpo_train])
    write_jsonl(out_dir / "persona_orpo_valid.jsonl", orpo[args.n_orpo_train :])
    write_jsonl(out_dir / "persona_grpo_prompts.jsonl", grpo)

    print(f"Wrote persona file SFT train={len(train)} valid={len(valid)} to {out_dir}")
    print(
        f"Wrote persona file ORPO train={min(len(orpo), args.n_orpo_train)} "
        f"valid={max(0, len(orpo) - args.n_orpo_train)} to {out_dir}"
    )
    print(f"Wrote persona file GRPO prompts={len(grpo)} to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--persona_sft_file",
        default="data/persona/marichatmen_persona_sft_12000.jsonl",
    )
    parser.add_argument("--out_dir", default="data/processed/persona_file")
    parser.add_argument("--n_sft_train", type=int, default=256)
    parser.add_argument("--n_sft_valid", type=int, default=32)
    parser.add_argument("--n_orpo_train", type=int, default=128)
    parser.add_argument("--n_orpo_valid", type=int, default=32)
    parser.add_argument("--n_grpo", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
