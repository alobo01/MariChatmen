"""Save Qwen-Andaluh stage examples with the base assistant system prompt."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

from marichatmen.constants import SYSTEM_PROMPT_BASE
from marichatmen.eval.generation_eval import generate_response, load_causal_model, load_tokenizer
from marichatmen.eval.mari_aas import score_dict
from marichatmen.io import write_jsonl

DEFAULT_PROMPTS = [
    "Explícame qué es el overfitting.",
    "¿Cómo puedo organizarme para estudiar un máster sin agobiarme?",
    "Resume qué es una API REST.",
    "¿Qué diferencias hay entre Málaga e Ibiza para ir de vacaciones?",
    "Dime cómo instalar transformers con uv.",
    "No entiendo la validación cruzada, explícamela fácil.",
]

STAGES = [
    ("original", "Original Qwen"),
    ("cpt", "Qwen-Andaluh CPT"),
    ("sft", "Qwen-Andaluh SFT"),
    ("orpo", "Qwen-Andaluh ORPO"),
]


def _load_prompts(path: str) -> list[str]:
    if not path:
        return DEFAULT_PROMPTS
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _release_model(model: Any) -> None:
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _generate(
    *,
    model_name: str,
    tokenizer_name: str,
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
        answers.append(
            generate_response(
                model,
                tokenizer,
                [
                    {"role": "system", "content": SYSTEM_PROMPT_BASE},
                    {"role": "user", "content": prompt},
                ],
                max_new_tokens=max_new_tokens,
                disable_thinking=True,
            )
        )
    _release_model(model)
    return answers


def run(args: argparse.Namespace) -> None:
    prompts = _load_prompts(args.prompts_file)
    adapters = {
        "original": "",
        "cpt": args.cpt_adapter,
        "sft": args.sft_adapter,
        "orpo": args.orpo_adapter,
    }
    stage_tokenizers = {
        "original": "",
        "cpt": args.tokenizer_name,
        "sft": args.tokenizer_name,
        "orpo": args.tokenizer_name,
    }

    generated: dict[str, list[str]] = {}
    for stage, label in STAGES:
        adapter = adapters[stage]
        if stage != "original" and not adapter:
            continue
        generated[stage] = _generate(
            model_name=args.model_name,
            tokenizer_name=stage_tokenizers[stage],
            adapter_path=adapter,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            no_4bit=args.no_4bit,
        )
        print(f"Generated {len(prompts)} answers for {label}")

    rows: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts):
        row: dict[str, Any] = {
            "id": f"qwen_andaluh_example_{index:04d}",
            "model_name": args.model_name,
            "system_prompt": SYSTEM_PROMPT_BASE,
            "prompt": prompt,
        }
        for stage, label in STAGES:
            if stage not in generated:
                continue
            answer = generated[stage][index]
            row[f"{stage}_label"] = label
            row[f"{stage}_answer"] = answer
            row[f"{stage}_mari_aas"] = score_dict(answer)
            row[f"{stage}_has_think_block"] = "<think>" in answer.lower()
        rows.append(row)

    write_jsonl(args.output_jsonl, rows)
    _write_markdown(Path(args.output_md), rows)
    print(f"Wrote Qwen-Andaluh examples to {args.output_jsonl}")


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# Qwen-Andaluh Stage Examples", "", "System prompt for every stage: `Eres un asistente`", ""]
    for row in rows:
        lines.extend([f"## {row['prompt']}", ""])
        for stage, label in STAGES:
            field = f"{stage}_answer"
            if field not in row:
                continue
            metrics = row.get(f"{stage}_mari_aas", {})
            score = metrics.get("score") if isinstance(metrics, dict) else None
            suffix = f" (MARI-AAS {score:.2f})" if isinstance(score, float) else ""
            lines.extend([f"### {label}{suffix}", "", row[field], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--tokenizer_name", default="outputs/tokenizers/qwen35_08b_andaluh")
    parser.add_argument("--cpt_adapter", default="outputs/qwen_andaluh_08b_cpt/final_adapter")
    parser.add_argument("--sft_adapter", default="outputs/qwen_andaluh_08b_sft/final_adapter")
    parser.add_argument("--orpo_adapter", default="")
    parser.add_argument("--prompts_file", default="")
    parser.add_argument("--output_jsonl", default="reports/samples/qwen_andaluh_stage_examples.jsonl")
    parser.add_argument("--output_md", default="showcase/qwen_andaluh_stage_examples.md")
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--no_4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
