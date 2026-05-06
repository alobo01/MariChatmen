"""Build ORPO preference pairs from MariChatmen SFT rows."""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Any

from marichatmen.constants import SYSTEM_PROMPT_TRAINING
from marichatmen.data.transliterate_andaluh import to_andaluh
from marichatmen.io import read_jsonl, write_jsonl

STANDARD_BACK_REPLACEMENTS = [
    (re.compile(r"\ber\b", re.IGNORECASE), "el"),
    (re.compile(r"\bloh\b", re.IGNORECASE), "los"),
    (re.compile(r"\blah\b", re.IGNORECASE), "las"),
    (re.compile(r"\bpa\b", re.IGNORECASE), "para"),
    (re.compile(r"\bmu\b", re.IGNORECASE), "muy"),
    (re.compile(r"\bná\b", re.IGNORECASE), "nada"),
    (re.compile(r"\btó\b", re.IGNORECASE), "todo"),
    (re.compile(r"\behtá\b", re.IGNORECASE), "está"),
    (re.compile(r"\behtoy\b", re.IGNORECASE), "estoy"),
    (re.compile(r"[çÇ]"), "s"),
    (re.compile(r"â"), "a"),
    (re.compile(r"ê"), "e"),
    (re.compile(r"î"), "i"),
    (re.compile(r"ô"), "o"),
    (re.compile(r"û"), "u"),
]

INFORMAL_MARKER_RE = re.compile(
    r"\b(ea|illo|quillo|miarma|una mijita|pechá|apañao|del tirón|no veah),?\s*",
    re.IGNORECASE,
)


def _split_prompt_chosen(
    messages: list[dict[str, str]],
    *,
    system_prompt: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]] | None:
    last_assistant = None
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            last_assistant = index
            break
    if last_assistant is None:
        return None
    prompt = [m for m in messages[:last_assistant] if m.get("role") in {"system", "user", "assistant"}]
    if not prompt or prompt[0].get("role") != "system":
        prompt = [{"role": "system", "content": system_prompt}] + prompt
    chosen = [{"role": "assistant", "content": messages[last_assistant]["content"]}]
    return prompt, chosen


def standard_spanish_rejection(text: str) -> str:
    out = text
    for pattern, replacement in STANDARD_BACK_REPLACEMENTS:
        out = pattern.sub(replacement, out)
    out = INFORMAL_MARKER_RE.sub("", out)
    return out.strip()


def mild_andaluh_rejection(text: str) -> str:
    out = INFORMAL_MARKER_RE.sub("", text)
    out = out.replace("ç", "s").replace("Ç", "S")
    out = re.sub(r"\bpa\b", "para", out, flags=re.IGNORECASE)
    out = re.sub(r"\bmu\b", "muy", out, flags=re.IGNORECASE)
    return out.strip()


def caricature_rejection(prompt: list[dict[str, str]]) -> str:
    user = next((m["content"] for m in reversed(prompt) if m.get("role") == "user"), "")
    topic = user[:80].strip(" ¿?!.") or "ehta pregunta"
    return f"KilloOOO {topic} ê una cosa der taco miarma illo illo illo!!!"


def make_rejected(
    prompt: list[dict[str, str]],
    chosen_text: str,
    rejected_type: str,
    seed: int,
) -> str:
    if rejected_type == "standard_spanish":
        return standard_spanish_rejection(chosen_text)
    if rejected_type == "mild_andaluh":
        return mild_andaluh_rejection(chosen_text)
    if rejected_type == "caricature":
        return caricature_rejection(prompt)
    if rejected_type == "regenerated_andaluh":
        return to_andaluh(chosen_text, informal_strength=0.2, seed=seed)
    raise ValueError(f"Unknown rejected type: {rejected_type}")


def build(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.sft_train)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rejected_types = args.rejected_types

    pairs: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        split = _split_prompt_chosen(row.get("messages", []), system_prompt=args.system_prompt)
        if not split:
            continue
        prompt, chosen = split
        rejected_type = rejected_types[index % len(rejected_types)]
        chosen_text = chosen[0]["content"]
        rejected = make_rejected(prompt, chosen_text, rejected_type, args.seed + index)
        if rejected == chosen_text:
            rejected = standard_spanish_rejection(chosen_text)
        pairs.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": [{"role": "assistant", "content": rejected}],
                "metadata": {
                    **row.get("metadata", {}),
                    "rejected_type": rejected_type,
                },
            }
        )
        if len(pairs) >= args.n_train + args.n_valid:
            break

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "orpo_train.jsonl", pairs[: args.n_train])
    write_jsonl(out_dir / "orpo_valid.jsonl", pairs[args.n_train : args.n_train + args.n_valid])
    print(
        f"Wrote {min(len(pairs), args.n_train)} ORPO train and "
        f"{max(0, min(len(pairs) - args.n_train, args.n_valid))} valid rows to {out_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_train", default="data/processed/sft_train.jsonl")
    parser.add_argument("--n_train", type=int, default=500)
    parser.add_argument("--n_valid", type=int, default=100)
    parser.add_argument(
        "--rejected_types",
        nargs="+",
        default=["standard_spanish", "mild_andaluh", "caricature"],
    )
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--system_prompt", default=SYSTEM_PROMPT_TRAINING)
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
