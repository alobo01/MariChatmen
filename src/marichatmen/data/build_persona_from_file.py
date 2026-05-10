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
from marichatmen.constants import ARTIFACT_ROOT
from marichatmen.data.transliterate_andaluh import strip_thinking, to_andaluh
from marichatmen.eval.mari_reward import score_mari_answer
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

KEYWORD_SOUP_REJECTIONS = [
    "Expo, Feria, SFDK, ToteKing, Triana, Macarena, gazpacho, Cruzcampo y Andalucía. Ea, miarma, eso ê lo importante.",
    "Miarma, la respuesta ê Cái, Málaga, Feria, caseta, litrito, SFDK y orguyo andalûh, con mucho arte y ya ehtá.",
]

WRONG_ANSWER_REJECTIONS = [
    "Eso se arregla repitiendo la palabra clave muchas veces y sin mirar el problema de fondo.",
    "La mejor opción es contestar con cualquier cosa simpática aunque no tenga relación con la pregunta.",
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
    if rejected_type == "keyword_soup":
        return KEYWORD_SOUP_REJECTIONS[seed % len(KEYWORD_SOUP_REJECTIONS)]
    if rejected_type == "wrong_answer":
        return WRONG_ANSWER_REJECTIONS[seed % len(WRONG_ANSWER_REJECTIONS)]
    raise ValueError(f"Unknown rejected_type: {rejected_type}")


def _first_user_text(messages: list[dict[str, str]]) -> str:
    return next((message["content"] for message in messages if message["role"] == "user"), "")


def _score_sft_row(row: dict[str, Any]) -> dict[str, Any] | None:
    split = _split_prompt_chosen(row["messages"])
    if split is None:
        return None
    prompt, chosen = split
    prompt_text = _first_user_text(prompt)
    chosen_text = chosen[0]["content"]
    scored = score_mari_answer(prompt_text, chosen_text)
    row["metadata"] = {
        **row.get("metadata", {}),
        "mari_reward": round(scored.reward, 6),
        "mari_reward_passes": scored.passes,
        "mari_aas": round(scored.mari_aas, 4),
        "mari_pas": round(scored.mari_pas, 4),
        "mari_task_answer_quality": round(scored.task_answer_quality, 6),
        "mari_keyword_soup_penalty": round(scored.keyword_soup_penalty, 6),
        "mari_reward_evidence": scored.evidence,
    }
    return row


def _make_orpo(
    rows: list[dict[str, Any]],
    n_rows: int,
    seed: int,
    *,
    min_margin: float,
) -> list[dict[str, Any]]:
    rejected_types = [
        "standard_spanish",
        "mild_andaluh",
        "caricature",
        "keyword_soup",
        "wrong_answer",
        "hostile",
        "off_persona",
        "unsafe_alcohol",
    ]
    pairs: list[dict[str, Any]] = []
    attempts = 0
    for index, row in enumerate(rows * max(1, (n_rows // max(1, len(rows))) + 2)):
        attempts += 1
        split = _split_prompt_chosen(row["messages"])
        if split is None:
            continue
        prompt, chosen = split
        prompt_text = _first_user_text(prompt)
        chosen_text = chosen[0]["content"]
        rejected_type = rejected_types[index % len(rejected_types)]
        rejected = _rejected_text(prompt, chosen_text, rejected_type, seed + index)
        if rejected_type in {"hostile", "caricature", "keyword_soup"}:
            rejected = to_andaluh(rejected, informal_strength=0.0, seed=seed + index)
        chosen_score = score_mari_answer(prompt_text, chosen_text)
        rejected_score = score_mari_answer(prompt_text, rejected)
        reward_margin = chosen_score.reward - rejected_score.reward
        if reward_margin < min_margin:
            continue
        pairs.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": [{"role": "assistant", "content": rejected}],
                "metadata": {
                    **row.get("metadata", {}),
                    "rejected_type": rejected_type,
                    "preference_family": "persona_file",
                    "mari_reward_chosen": round(chosen_score.reward, 6),
                    "mari_reward_rejected": round(rejected_score.reward, 6),
                    "mari_reward_margin": round(reward_margin, 6),
                    "mari_chosen_passes": chosen_score.passes,
                    "mari_rejected_passes": rejected_score.passes,
                },
            }
        )
        if len(pairs) >= n_rows:
            break
    if len(pairs) < n_rows:
        print(
            f"[build_persona_from_file] Warning: built {len(pairs)} ORPO pairs from "
            f"{attempts} attempts; lower --min_orpo_margin or provide more rows if needed."
        )
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
                "reward": {
                    "name": "self_verified_mari_reward",
                    "min_reward": 0.52,
                    "penalizes": [
                        "keyword_soup",
                        "missing_answer",
                        "repetition",
                        "regional_hostility",
                        "unsafe_alcohol",
                    ],
                },
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

    score_limit = args.max_scored_rows or max(
        args.n_sft_train + args.n_sft_valid + args.n_grpo,
        3 * (args.n_orpo_train + args.n_orpo_valid),
        512,
    )
    rows_to_score = rows[: min(len(rows), score_limit)]
    available_rows = len(rows)
    scored_rows = [scored for row in rows_to_score if (scored := _score_sft_row(row)) is not None]
    rows = [
        row
        for row in scored_rows
        if row.get("metadata", {}).get("mari_reward", 0.0) >= args.min_sft_reward
    ]
    if not rows:
        raise RuntimeError(
            f"No persona rows passed --min_sft_reward={args.min_sft_reward}. "
            "Lower the threshold or inspect the source data."
        )

    train = rows[: args.n_sft_train]
    valid = rows[args.n_sft_train : args.n_sft_train + args.n_sft_valid]
    if len(valid) < args.n_sft_valid:
        valid = rows[-args.n_sft_valid :]

    orpo_source = rows[args.n_sft_train + args.n_sft_valid :] or rows
    orpo = _make_orpo(
        orpo_source,
        args.n_orpo_train + args.n_orpo_valid,
        args.seed + 1000,
        min_margin=args.min_orpo_margin,
    )
    grpo = _make_grpo(rows, args.n_grpo)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "persona_sft_train.jsonl", train)
    write_jsonl(out_dir / "persona_sft_valid.jsonl", valid)
    write_jsonl(out_dir / "persona_orpo_train.jsonl", orpo[: args.n_orpo_train])
    write_jsonl(out_dir / "persona_orpo_valid.jsonl", orpo[args.n_orpo_train :])
    write_jsonl(out_dir / "persona_grpo_prompts.jsonl", grpo)

    print(
        f"Wrote persona file SFT train={len(train)} valid={len(valid)} to {out_dir} "
        f"after reward filtering {len(rows)}/{len(scored_rows)} scored rows "
        f"from {len(rows_to_score)}/{available_rows} available rows"
    )
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
    parser.add_argument("--out_dir", default=str(ARTIFACT_ROOT / "data/processed/persona_file"))
    parser.add_argument("--n_sft_train", type=int, default=256)
    parser.add_argument("--n_sft_valid", type=int, default=32)
    parser.add_argument("--n_orpo_train", type=int, default=128)
    parser.add_argument("--n_orpo_valid", type=int, default=32)
    parser.add_argument("--n_grpo", type=int, default=64)
    parser.add_argument("--min_sft_reward", type=float, default=0.35)
    parser.add_argument("--min_orpo_margin", type=float, default=0.12)
    parser.add_argument(
        "--max_scored_rows",
        type=int,
        default=0,
        help="Maximum shuffled source rows to self-score; 0 computes a size from requested splits.",
    )
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
