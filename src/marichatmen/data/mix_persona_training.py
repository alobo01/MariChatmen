"""Mix general accent data with persona data for the full personality pipeline."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from marichatmen.io import read_jsonl, write_jsonl


def _sample_with_replacement(rows: list[dict[str, Any]], n_rows: int, rng: random.Random) -> list[dict[str, Any]]:
    if not rows:
        return []
    if n_rows <= len(rows):
        copied = list(rows)
        rng.shuffle(copied)
        return copied[:n_rows]
    return [rng.choice(rows) for _ in range(n_rows)]


def _safety_or_helpfulness_orpo(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        metadata = row.get("metadata", {})
        rejected_type = metadata.get("rejected_type", "")
        category = metadata.get("category", "")
        if category == "Safety" or rejected_type in {"hostile", "unsafe_alcohol", "caricature"}:
            selected.append(row)
    return selected or rows


def build(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    general_sft = read_jsonl(args.general_sft)
    persona_sft = read_jsonl(args.persona_sft)
    total_sft = args.n_sft or (len(general_sft) + len(persona_sft))
    persona_sft_n = round(total_sft * args.persona_sft_ratio)
    general_sft_n = total_sft - persona_sft_n
    sft_rows = _sample_with_replacement(general_sft, general_sft_n, rng)
    sft_rows += _sample_with_replacement(persona_sft, persona_sft_n, rng)
    rng.shuffle(sft_rows)

    general_orpo = read_jsonl(args.general_orpo)
    persona_orpo = read_jsonl(args.persona_orpo)
    safety_pool = _safety_or_helpfulness_orpo(general_orpo + persona_orpo)
    total_orpo = args.n_orpo or (len(general_orpo) + len(persona_orpo))
    accent_n = round(total_orpo * args.accent_orpo_ratio)
    persona_n = round(total_orpo * args.persona_orpo_ratio)
    safety_n = max(0, total_orpo - accent_n - persona_n)
    orpo_rows = _sample_with_replacement(general_orpo, accent_n, rng)
    orpo_rows += _sample_with_replacement(persona_orpo, persona_n, rng)
    orpo_rows += _sample_with_replacement(safety_pool, safety_n, rng)
    rng.shuffle(orpo_rows)

    write_jsonl(out_dir / "sft_train.jsonl", sft_rows)
    write_jsonl(out_dir / "orpo_train.jsonl", orpo_rows)
    print(f"Wrote mixed SFT: {len(sft_rows)} rows to {out_dir / 'sft_train.jsonl'}")
    print(f"Wrote mixed ORPO: {len(orpo_rows)} rows to {out_dir / 'orpo_train.jsonl'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--general_sft", default="data/processed/sft_train.jsonl")
    parser.add_argument("--persona_sft", default="data/processed/persona_sft.jsonl")
    parser.add_argument("--general_orpo", default="data/processed/orpo_train.jsonl")
    parser.add_argument("--persona_orpo", default="data/processed/persona_orpo.jsonl")
    parser.add_argument("--out_dir", default="data/processed/persona_mix")
    parser.add_argument("--n_sft", type=int, default=0)
    parser.add_argument("--n_orpo", type=int, default=0)
    parser.add_argument("--persona_sft_ratio", type=float, default=0.20)
    parser.add_argument("--accent_orpo_ratio", type=float, default=0.50)
    parser.add_argument("--persona_orpo_ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=93)
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
