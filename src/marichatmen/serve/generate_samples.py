"""Generate a small qualitative sample markdown file."""

from __future__ import annotations

import argparse
from pathlib import Path

from marichatmen.constants import SYSTEM_PROMPT_BASE
from marichatmen.eval.generation_eval import generate_response, load_causal_model, load_tokenizer

PROMPTS = [
    "Explícame qué es el overfitting.",
    "Dame una receta rápida con garbanzos.",
    "Estoy agobiao con el máster, ¿cómo me organizo?",
    "¿Qué diferencia hay entre DPO y ORPO?",
    "Quillo, ¿me haceh un plan pa entrenâh esta semana?",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--tokenizer_name", default="")
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--system_prompt", default=SYSTEM_PROMPT_BASE)
    parser.add_argument("--out_file", default="showcase/final_samples.md")
    parser.add_argument("--no_4bit", action="store_true")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model_name, args.tokenizer_name or None)
    model = load_causal_model(
        args.model_name,
        args.adapter_path or None,
        load_in_4bit=not args.no_4bit,
        tokenizer_len=len(tokenizer),
    )
    lines = ["# MariChatmen Samples", ""]
    for prompt in PROMPTS:
        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": prompt},
        ]
        answer = generate_response(model, tokenizer, messages)
        lines.extend([f"## {prompt}", "", answer, ""])
    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote samples to {out}")


if __name__ == "__main__":
    main()
