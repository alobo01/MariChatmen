"""Build neutral Qwen-Andaluh SFT data from an instructed Qwen teacher.

The default teacher is the instructed/chat Qwen checkpoint, not the base model.
It answers in normal Spanish with the neutral project system prompt
(`Eres un asistente`). The assistant target is then converted to Andaluh EPA.
This keeps behaviour close to a useful instructed model while changing the
student's output register.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path
from typing import Any

from marichatmen.constants import ARTIFACT_ROOT, SYSTEM_PROMPT_BASE
from marichatmen.data.transliterate_andaluh import strip_thinking, to_andaluh
from marichatmen.eval.generation_eval import generate_response, load_causal_model, load_tokenizer
from marichatmen.io import append_jsonl, iter_jsonl, write_jsonl

PROMPT_TOPICS = [
    "overfitting",
    "validación cruzada",
    "una API REST",
    "un entorno virtual de Python con uv",
    "la diferencia entre CPU y GPU",
    "la fotosíntesis",
    "el cambio climático",
    "la inflación",
    "la planificación de un máster",
    "un correo profesional breve",
    "una función de Python que limpia texto",
    "la seguridad al publicar datos",
    "el método científico",
    "la memoria de una GPU",
    "un presupuesto personal sencillo",
    "la diferencia entre precisión y recall",
    "un plan de estudio de una semana",
    "la gestión de errores en una API",
    "un resumen de un texto largo",
    "una explicación para una persona principiante",
]

PROMPT_TEMPLATES = [
    "Explícame {topic} con claridad y un ejemplo.",
    "Dame una respuesta breve sobre {topic}.",
    "Haz una lista de pasos para entender {topic}.",
    "Compara {topic} con una idea parecida y explica la diferencia.",
    "Resume {topic} en tres puntos.",
    "Dame un ejemplo práctico de {topic}.",
    "¿Qué errores comunes hay al aprender {topic}?",
    "Explícame {topic} como si estuviera empezando.",
    "Propón una forma de practicar {topic}.",
    "¿Qué debería revisar antes de aplicar {topic}?",
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


def _default_prompt_rows(count: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        topic = PROMPT_TOPICS[index % len(PROMPT_TOPICS)]
        template = PROMPT_TEMPLATES[(index // len(PROMPT_TOPICS)) % len(PROMPT_TEMPLATES)]
        if index >= len(PROMPT_TOPICS) * len(PROMPT_TEMPLATES):
            topic = f"{topic} en el caso {index}"
        prompt = template.format(topic=topic)
        rows.append(
            {
                "id": f"synthetic_prompt_{index:06d}",
                "prompt": prompt,
                "category": rng.choice(
                    [
                        "technical_explanation",
                        "general_qa",
                        "planning",
                        "writing",
                        "study_support",
                    ]
                ),
            }
        )
    return rows


def _load_prompt_rows(path: str, count: int, seed: int) -> list[dict[str, str]]:
    if not path:
        return _default_prompt_rows(count, seed)

    prompt_path = Path(path)
    rows: list[dict[str, str]] = []
    for line_index, line in enumerate(prompt_path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            row = {"prompt": line}
        prompt = row.get("prompt") if isinstance(row, dict) else None
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        rows.append(
            {
                "id": str(row.get("id") or f"prompt_{line_index:06d}"),
                "prompt": prompt.strip(),
                "category": str(row.get("category") or "general_qa"),
            }
        )
        if len(rows) >= count:
            break
    if len(rows) < count:
        defaults = _default_prompt_rows(count - len(rows), seed + 17)
        rows.extend(defaults)
    return rows[:count]


def _load_existing_generations(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = {}
    for row in iter_jsonl(path):
        prompt_id = row.get("id")
        if isinstance(prompt_id, str):
            rows[prompt_id] = row
    return rows


def _generate_missing(args: argparse.Namespace, prompt_rows: list[dict[str, str]], raw_file: Path) -> list[dict[str, Any]]:
    existing = _load_existing_generations(raw_file) if args.reuse_generations else {}
    missing = [row for row in prompt_rows if row["id"] not in existing]
    if missing:
        tokenizer = load_tokenizer(args.reference_model, args.reference_tokenizer or None)
        model = load_causal_model(
            args.reference_model,
            None,
            load_in_4bit=not args.no_4bit,
            tokenizer_len=len(tokenizer),
        )
        for row in missing:
            answer = generate_response(
                model,
                tokenizer,
                [
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": row["prompt"]},
                ],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                disable_thinking=True,
            )
            generated = {
                **row,
                "system_prompt": args.system_prompt,
                "reference_model": args.reference_model,
                "original_assistant": strip_thinking(answer),
            }
            append_jsonl(raw_file, generated)
            existing[row["id"]] = generated
        _release_model(model)
    return [existing[row["id"]] for row in prompt_rows if row["id"] in existing]


def _build_sft_row(raw: dict[str, Any], args: argparse.Namespace, row_index: int) -> dict[str, Any] | None:
    prompt = raw.get("prompt")
    original = raw.get("original_assistant")
    if not isinstance(prompt, str) or not isinstance(original, str):
        return None
    original = strip_thinking(original)
    if len(original.split()) < args.min_answer_words:
        return None
    assistant = to_andaluh(
        original,
        variant=args.variant,
        informal_strength=args.informal_strength,
        seed=args.seed + row_index,
    )
    rng = random.Random(args.seed + row_index * 7919)
    user_content = (
        to_andaluh(
            prompt,
            variant=args.variant,
            informal_strength=args.informal_strength * 0.35,
            seed=args.seed + row_index * 31,
        )
        if rng.random() < args.user_andaluh_ratio
        else prompt
    )
    return {
        "messages": [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "source_dataset": "synthetic_qwen_distillation",
                "source_model": args.reference_model,
                "source_model_role": "instructed_teacher",
                "source_license": "Qwen model output; review before public dataset release",
            "category": raw.get("category", "general_qa"),
            "language": "spa",
            "system_prompt": args.system_prompt,
            "transformation": f"assistant_to_andaluh_epa_{args.variant}",
            "transliterator": "andalugeeks/andaluh-py",
            "original_user": prompt,
            "original_assistant": original,
        },
    }


def _make_bench(valid_rows: list[dict[str, Any]], max_prompts: int) -> list[dict[str, Any]]:
    bench = []
    for index, row in enumerate(valid_rows[:max_prompts]):
        messages = row["messages"]
        prompt = [item for item in messages if item["role"] in {"system", "user"}]
        reference = next(item["content"] for item in reversed(messages) if item["role"] == "assistant")
        bench.append(
            {
                "id": f"qwen_andaluh_distilled_{index:05d}",
                "prompt": prompt,
                "reference": reference,
                "original_reference": row.get("metadata", {}).get("original_assistant", ""),
                "category": row.get("metadata", {}).get("category", "general_qa"),
            }
        )
    return bench


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    needed = args.n_train + args.n_valid
    prompt_count = args.prompt_count or needed
    prompt_rows = _load_prompt_rows(args.prompts_file, prompt_count, args.seed)
    generated = _generate_missing(args, prompt_rows, raw_dir / "original_generations.jsonl")

    processed = []
    for row_index, raw in enumerate(generated):
        item = _build_sft_row(raw, args, row_index)
        if item:
            processed.append(item)

    rng = random.Random(args.seed)
    rng.shuffle(processed)
    train_rows = processed[: args.n_train]
    valid_rows = processed[args.n_train : args.n_train + args.n_valid]
    write_jsonl(out_dir / "sft_train.jsonl", train_rows)
    write_jsonl(out_dir / "sft_valid.jsonl", valid_rows)
    write_jsonl(out_dir / "mari_bench_v1.jsonl", _make_bench(valid_rows, args.eval_prompts))
    manifest = {
        "builder": "build_distilled_sft",
        "reference_model": args.reference_model,
        "reference_model_role": "instructed_teacher",
        "system_prompt": args.system_prompt,
        "n_train": len(train_rows),
        "n_valid": len(valid_rows),
        "prompt_count": len(prompt_rows),
        "user_andaluh_ratio": args.user_andaluh_ratio,
        "assistant_andaluh_ratio": 1.0,
        "variant": args.variant,
        "informal_strength": args.informal_strength,
        "transformation": "original_model_spanish_answer_to_andaluh_epa",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote distilled SFT data to {out_dir}: "
        f"train={len(train_rows)}, valid={len(valid_rows)}, raw_generations={len(generated)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--reference_tokenizer", default="")
    parser.add_argument("--prompts_file", default="")
    parser.add_argument("--out_dir", default=str(ARTIFACT_ROOT / "data/processed/distilled_sft"))
    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--n_valid", type=int, default=100)
    parser.add_argument("--prompt_count", type=int, default=0)
    parser.add_argument("--eval_prompts", type=int, default=100)
    parser.add_argument("--system_prompt", default=SYSTEM_PROMPT_BASE)
    parser.add_argument("--user_andaluh_ratio", type=float, default=0.3)
    parser.add_argument("--variant", default="sevillian_ce")
    parser.add_argument("--informal_strength", type=float, default=0.2)
    parser.add_argument("--min_answer_words", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--reuse_generations", type=lambda value: value.lower() == "true", default=True)
    parser.add_argument("--seed", type=int, default=51)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
