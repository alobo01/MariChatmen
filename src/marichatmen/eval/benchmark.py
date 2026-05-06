"""Generate and score MariChatmen benchmark samples."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any

from marichatmen.eval.generation_eval import generate_response, load_causal_model, load_tokenizer
from marichatmen.eval.mari_pas import province_diversity_entropy, total_score_dict
from marichatmen.io import append_csv_row, read_jsonl, write_jsonl


def _prompt_from_row(row: dict[str, Any]) -> list[dict[str, str]]:
    prompt = row.get("prompt") or row.get("messages")
    if isinstance(prompt, list):
        return [m for m in prompt if isinstance(m, dict) and m.get("role") in {"system", "user"}]
    raise ValueError("Benchmark row must contain prompt or messages")


def run(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.eval_file)
    tokenizer = load_tokenizer(args.model_name, args.tokenizer_name or None)
    model = load_causal_model(
        args.model_name,
        args.adapter_path or None,
        load_in_4bit=not args.no_4bit,
        tokenizer_len=len(tokenizer),
    )

    samples: list[dict[str, Any]] = []
    started = time.perf_counter()
    for idx, row in enumerate(rows):
        prompt = _prompt_from_row(row)
        output = generate_response(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            disable_thinking=not args.enable_thinking,
        )
        metrics = total_score_dict(output, row.get("reference"))
        samples.append(
            {
                "id": row.get("id", f"sample_{idx:04d}"),
                "run_name": args.run_name,
                "prompt": prompt,
                "reference": row.get("reference"),
                "output": output,
                "metrics": metrics,
            }
        )

    duration = time.perf_counter() - started
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_jsonl, samples)

    outputs = [sample["output"] for sample in samples]
    aas_scores = [sample["metrics"]["mari_aas"] for sample in samples]
    pas_scores = [sample["metrics"]["mari_pas"] for sample in samples]
    total_scores = [sample["metrics"]["mari_total"] for sample in samples]
    leaks = [sample["metrics"]["aas_spanish_leak"] for sample in samples]
    caricatures = [sample["metrics"]["aas_caricature_penalty"] for sample in samples]
    flourish = [sample["metrics"]["pas_province_flourish"] for sample in samples]
    hostility = [sample["metrics"]["pas_regional_hostility_penalty"] for sample in samples]
    alcohol = [sample["metrics"]["pas_alcohol_safety_penalty"] for sample in samples]
    row = {
        "run_name": args.run_name,
        "model_name": args.model_name,
        "adapter_path": args.adapter_path,
        "n_samples": len(samples),
        "mari_aas_mean": statistics.fmean(aas_scores) if aas_scores else 0.0,
        "mari_aas_std": statistics.pstdev(aas_scores) if len(aas_scores) > 1 else 0.0,
        "mari_pas_mean": statistics.fmean(pas_scores) if pas_scores else 0.0,
        "mari_total_mean": statistics.fmean(total_scores) if total_scores else 0.0,
        "province_flourish_rate": sum(1 for item in flourish if item >= 1.0) / max(1, len(flourish)),
        "province_diversity_entropy": province_diversity_entropy(outputs),
        "sevilla_identity_score": statistics.fmean(
            [sample["metrics"]["pas_sevillian_identity"] for sample in samples]
        )
        if samples
        else 0.0,
        "andalusian_pride_score": statistics.fmean(
            [sample["metrics"]["pas_andalusian_pride"] for sample in samples]
        )
        if samples
        else 0.0,
        "music_reference_rate": statistics.fmean(
            [sample["metrics"]["pas_music_reference"] for sample in samples]
        )
        if samples
        else 0.0,
        "feria_reference_rate": statistics.fmean(
            [sample["metrics"]["pas_feria_reference"] for sample in samples]
        )
        if samples
        else 0.0,
        "gazpacho_paella_preference_pass": statistics.fmean(
            [sample["metrics"]["pas_gazpacho_paella_preference_pass"] for sample in samples]
        )
        if samples
        else 0.0,
        "malaga_ibiza_preference_pass": statistics.fmean(
            [sample["metrics"]["pas_malaga_ibiza_preference_pass"] for sample in samples]
        )
        if samples
        else 0.0,
        "regional_hostility_penalty": statistics.fmean(hostility) if hostility else 0.0,
        "alcohol_safety_penalty": statistics.fmean(alcohol) if alcohol else 0.0,
        "persona_consistency_score": statistics.fmean(
            [sample["metrics"]["pas_consistency"] for sample in samples]
        )
        if samples
        else 0.0,
        "spanish_leak_rate": statistics.fmean(leaks) if leaks else 0.0,
        "caricature_penalty": statistics.fmean(caricatures) if caricatures else 0.0,
        "eval_generation_seconds": duration,
    }
    append_csv_row(args.metrics_csv, row)
    print(f"Wrote {len(samples)} samples to {args.output_jsonl}")
    print(f"MARI-AAS mean: {row['mari_aas_mean']:.2f}")
    print(f"MARI-PAS mean: {row['mari_pas_mean']:.2f}")
    print(f"MARI total mean: {row['mari_total_mean']:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--tokenizer_name", default="")
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--eval_file", default="data/processed/mari_bench_v1.jsonl")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--metrics_csv", default="reports/eval_history.csv")
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--no_4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
