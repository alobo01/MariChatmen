"""Evaluate Qwen-Andaluh without persona metrics."""

from __future__ import annotations

import argparse
import re
import statistics
import time
from typing import Any

from marichatmen.constants import ARTIFACT_ROOT
from marichatmen.eval.generation_eval import generate_response, load_causal_model, load_tokenizer
from marichatmen.eval.mari_aas import score_dict
from marichatmen.eval.quality_metrics import (
    direct_answer_score,
    has_generation_artifact,
    has_reasoning_preamble,
    repetition_rate,
    technical_correctness_score,
)
from marichatmen.io import append_csv_row, read_jsonl, write_jsonl

PROMPT_LEAK_RE = re.compile(
    r"\b(andal[uúû]h?|epa|seseo|sevillan[oa]s?|çebiyan[oa]s?|dialect|translit|"
    r"responde\s+en|respóndeme\s+en|contesta\s+en|responde\s+siempre|"
    r"contesta\s+siempre)\b",
    re.IGNORECASE,
)


def _prompt_from_row(row: dict[str, Any]) -> list[dict[str, str]]:
    prompt = row.get("prompt") or row.get("messages")
    if isinstance(prompt, list):
        return [m for m in prompt if isinstance(m, dict) and m.get("role") in {"system", "user"}]
    raise ValueError("Qwen-Andaluh eval rows need `prompt` or `messages`.")


def _assert_no_prompt_leak(prompt: list[dict[str, str]]) -> None:
    """Keep neutral Qwen-Andaluh eval free of orthography/persona hints."""
    for message in prompt:
        if message.get("role") not in {"system", "user"}:
            continue
        content = str(message.get("content", ""))
        match = PROMPT_LEAK_RE.search(content)
        if match:
            raise ValueError(
                "Qwen-Andaluh eval prompt leaks the target style: "
                f"{match.group(0)!r} in {content!r}"
            )


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def run(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.eval_file)[: args.limit or None]
    tokenizer = load_tokenizer(
        args.model_name,
        args.tokenizer_name or None,
        args.adapter_path or None,
    )
    model = load_causal_model(
        args.model_name,
        args.adapter_path or None,
        load_in_4bit=not args.no_4bit,
        tokenizer_len=len(tokenizer),
        tokenizer=tokenizer,
        tokenizer_name=args.tokenizer_name or args.model_name,
    )

    samples = []
    started = time.perf_counter()
    for index, row in enumerate(rows):
        prompt = _prompt_from_row(row)
        if not args.allow_prompt_leak:
            _assert_no_prompt_leak(prompt)
        reference = row.get("reference")
        output = generate_response(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            disable_thinking=True,
        )
        metrics = score_dict(output, reference if isinstance(reference, str) else None)
        prompt_text = "\n".join(
            str(message.get("content", ""))
            for message in prompt
            if message.get("role") == "user"
        )
        quality = {
            "generation_artifact": float(has_generation_artifact(output)),
            "reasoning_preamble": float(has_reasoning_preamble(output)),
            "direct_answer": direct_answer_score(output),
            "technical_correctness": technical_correctness_score(prompt_text, output),
            "repetition_rate": repetition_rate(output),
            "avg_output_len": len(output),
        }
        samples.append(
            {
                "id": row.get("id", f"qwen_andaluh_eval_{index:05d}"),
                "run_name": args.run_name,
                "stage": args.stage,
                "prompt": prompt,
                "reference": reference,
                "output": output,
                "metrics": metrics,
                "quality": quality,
            }
        )
    duration = time.perf_counter() - started

    write_jsonl(args.output_jsonl, samples)
    scores = [sample["metrics"] for sample in samples]
    aas = [row["score"] for row in scores]
    chrf = [row["chrfpp_reference"] for row in scores]
    leak = [row["spanish_leak"] for row in scores]
    degeneration = [row["degeneration_penalty"] for row in scores]
    caricature = [row["caricature_penalty"] for row in scores]
    quality_rows = [sample["quality"] for sample in samples]
    reasoning_preamble = [row["reasoning_preamble"] for row in quality_rows]
    generation_artifact = [row["generation_artifact"] for row in quality_rows]
    direct_answer = [row["direct_answer"] for row in quality_rows]
    technical_correctness = [row["technical_correctness"] for row in quality_rows]
    repetition = [row["repetition_rate"] for row in quality_rows]
    output_len = [row["avg_output_len"] for row in quality_rows]
    clear = [
        1.0
        if row["score"] >= args.pass_aas and row["spanish_leak"] <= args.pass_leak
        else 0.0
        for row in scores
    ]
    summary = {
        "run_name": args.run_name,
        "stage": args.stage,
        "model_name": args.model_name,
        "adapter_path": args.adapter_path,
        "tokenizer_name": args.tokenizer_name,
        "eval_file": args.eval_file,
        "n_samples": len(samples),
        "mari_aas_mean": _mean(aas),
        "mari_aas_std": statistics.pstdev(aas) if len(aas) > 1 else 0.0,
        "chrfpp_reference_mean": _mean(chrf),
        "spanish_leak_mean": _mean(leak),
        "clear_andaluh_rate": _mean(clear),
        "reasoning_preamble_rate": _mean(reasoning_preamble),
        "generation_artifact_rate": _mean(generation_artifact),
        "direct_answer_rate": _mean(direct_answer),
        "technical_correctness_rate": _mean(technical_correctness),
        "repetition_rate": _mean(repetition),
        "avg_output_len": _mean(output_len),
        "degeneration_penalty_mean": _mean(degeneration),
        "caricature_penalty_mean": _mean(caricature),
        "raw_train_tokens": args.raw_train_tokens,
        "planned_seen_tokens": args.planned_seen_tokens,
        "source_rows": args.source_rows,
        "eval_generation_seconds": duration,
    }
    append_csv_row(args.metrics_csv, summary)
    print(
        f"{args.run_name}: AAS={summary['mari_aas_mean']:.2f}, "
        f"clear={summary['clear_andaluh_rate']:.2%}, leak={summary['spanish_leak_mean']:.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--tokenizer_name", default="")
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--stage", default="unknown")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument(
        "--metrics_csv",
        default=str(ARTIFACT_ROOT / "reports/qwen_andaluh_eval_history.csv"),
    )
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--repetition_penalty", type=float, default=1.08)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--pass_aas", type=float, default=80.0)
    parser.add_argument("--pass_leak", type=float, default=0.15)
    parser.add_argument("--raw_train_tokens", type=int, default=0)
    parser.add_argument("--planned_seen_tokens", type=int, default=0)
    parser.add_argument("--source_rows", type=int, default=0)
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--allow_prompt_leak", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
