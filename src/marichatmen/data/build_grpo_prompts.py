"""Build prompt-only data for optional GRPO accent optimization."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from marichatmen.constants import ARTIFACT_ROOT
from marichatmen.io import read_jsonl, write_jsonl


def build(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.sft_train)
    random.Random(args.seed).shuffle(rows)
    prompts = []
    for idx, row in enumerate(rows[: args.n_prompts]):
        messages = row.get("messages", [])
        prompt = []
        for message in messages:
            if message.get("role") == "assistant":
                break
            if message.get("role") in {"system", "user"}:
                prompt.append(message)
        if prompt:
            prompts.append({"id": f"grpo_{idx:05d}", "prompt": prompt})
    out = Path(args.out_file)
    write_jsonl(out, prompts)
    print(f"Wrote {len(prompts)} GRPO prompts to {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_train", default=str(ARTIFACT_ROOT / "data/processed/base/sft_train.jsonl"))
    parser.add_argument("--n_prompts", type=int, default=1000)
    parser.add_argument("--out_file", default=str(ARTIFACT_ROOT / "data/processed/base/grpo_prompts.jsonl"))
    parser.add_argument("--seed", type=int, default=44)
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
