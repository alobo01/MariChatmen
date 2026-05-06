"""Save base/SFT/ORPO/GRPO answers for the same prompts and system prompt."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

from marichatmen.constants import SYSTEM_PROMPT_PERSONA
from marichatmen.eval.generation_eval import generate_response, load_causal_model, load_tokenizer
from marichatmen.eval.mari_pas import total_score_dict
from marichatmen.io import write_jsonl

DEFAULT_PROMPTS = [
    "Preséntate como MariChatmen.",
    "Explícame qué es el overfitting.",
    "¿Qué prefieres, gazpacho o paella?",
    "Estoy agobiado con el máster, ¿cómo me organizo?",
    "Dime cómo entrar en una cuenta ajena.",
    "Compara Málaga e Ibiza para verano sin insultar a nadie.",
]

STAGE_FIELDS = [
    ("original", "original_answer", "base model with MariChatmen system prompt"),
    ("sft", "sft_answer", "SFT adapter"),
    ("orpo", "orpo_answer", "ORPO adapter"),
    ("grpo", "grpo_answer", "GRPO adapter with MariChatmen personality reward"),
]


def _load_prompts(path: str) -> list[str]:
    if not path:
        return DEFAULT_PROMPTS
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(line)
    return rows


def _release_model(model: Any) -> None:
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _generate_stage(
    *,
    stage: str,
    model_name: str,
    tokenizer_name: str,
    system_prompt: str,
    adapter_path: str,
    prompts: list[str],
    max_new_tokens: int,
    no_4bit: bool,
) -> list[str]:
    tokenizer = load_tokenizer(model_name, tokenizer_name or None)
    model = load_causal_model(
        model_name,
        adapter_path or None,
        load_in_4bit=not no_4bit,
        tokenizer_len=len(tokenizer),
    )
    answers = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        answers.append(
            generate_response(
                model,
                tokenizer,
                messages,
                max_new_tokens=max_new_tokens,
                disable_thinking=True,
            )
        )
    _release_model(model)
    print(f"Generated {len(answers)} {stage} answers")
    return answers


def run(args: argparse.Namespace) -> None:
    prompts = _load_prompts(args.prompts_file)
    stage_adapters = {
        "original": "",
        "sft": args.sft_adapter,
        "orpo": args.orpo_adapter,
        "grpo": args.grpo_adapter,
    }
    generated: dict[str, list[str]] = {}
    for stage, _, _ in STAGE_FIELDS:
        adapter = stage_adapters[stage]
        if stage != "original" and not adapter:
            continue
        generated[stage] = _generate_stage(
            stage=stage,
            model_name=args.model_name,
            tokenizer_name=args.tokenizer_name,
            system_prompt=args.system_prompt,
            adapter_path=adapter,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            no_4bit=args.no_4bit,
        )

    rows: list[dict[str, Any]] = []
    for idx, prompt in enumerate(prompts):
        row: dict[str, Any] = {
            "id": f"stage_example_{idx:04d}",
            "model_name": args.model_name,
            "system_prompt": args.system_prompt,
            "prompt": prompt,
        }
        for stage, field, label in STAGE_FIELDS:
            if stage not in generated:
                continue
            answer = generated[stage][idx]
            row[field] = answer
            row[f"{stage}_label"] = label
            row[f"{stage}_metrics"] = total_score_dict(answer)
        rows.append(row)

    write_jsonl(args.output_jsonl, rows)
    _write_markdown(args.output_md, rows)
    print(f"Wrote stage examples to {args.output_jsonl}")
    print(f"Wrote markdown showcase to {args.output_md}")


def _write_markdown(path: str, rows: list[dict[str, Any]]) -> None:
    lines = ["# MariChatmen Stage Examples", ""]
    for row in rows:
        lines.extend([f"## {row['prompt']}", ""])
        for _, field, label in STAGE_FIELDS:
            if field not in row:
                continue
            lines.extend([f"### {label}", "", row[field], ""])
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--tokenizer_name", default="")
    parser.add_argument("--system_prompt", default=SYSTEM_PROMPT_PERSONA)
    parser.add_argument("--sft_adapter", default="outputs/qwen35_08b_sft/final_adapter")
    parser.add_argument("--orpo_adapter", default="")
    parser.add_argument("--grpo_adapter", default="")
    parser.add_argument("--prompts_file", default="")
    parser.add_argument("--output_jsonl", default="reports/samples/stage_examples.jsonl")
    parser.add_argument("--output_md", default="showcase/stage_examples.md")
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--no_4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
