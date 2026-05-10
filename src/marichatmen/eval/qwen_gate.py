"""Gate Qwen-Andaluh stages using generated samples, not loss alone."""

from __future__ import annotations

import argparse
import json
import statistics
import re
from pathlib import Path
from typing import Any

from marichatmen.io import read_jsonl


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


PERSONA_LEAK_RE = re.compile(
    r"\b(marichatmen|maricarmen|expo\s*['’]?92|cruzcampo|litrito|caseta|"
    r"sfdk|toteking|feria|triana|macarena)\b",
    re.IGNORECASE,
)


def summarize(samples: list[dict[str, Any]], *, min_aas: float, max_leak: float) -> dict[str, float]:
    metrics = [row.get("metrics", {}) for row in samples]
    quality = [row.get("quality", {}) for row in samples]
    outputs = [str(row.get("output", "")) for row in samples]
    return {
        "n_samples": float(len(samples)),
        "mari_aas_mean": _mean([float(row.get("score", 0.0)) for row in metrics]),
        "spanish_leak_mean": _mean([float(row.get("spanish_leak", 1.0)) for row in metrics]),
        "clear_andaluh_rate": _mean(
            [
                1.0
                if float(row.get("score", 0.0)) >= min_aas and float(row.get("spanish_leak", 1.0)) <= max_leak
                else 0.0
                for row in metrics
            ]
        ),
        "reasoning_preamble_rate": _mean([float(row.get("reasoning_preamble", 1.0)) for row in quality]),
        "generation_artifact_rate": _mean([float(row.get("generation_artifact", 0.0)) for row in quality]),
        "direct_answer_rate": _mean([float(row.get("direct_answer", 0.0)) for row in quality]),
        "technical_correctness_rate": _mean([float(row.get("technical_correctness", 0.0)) for row in quality]),
        "repetition_rate": _mean([float(row.get("repetition_rate", 1.0)) for row in quality]),
        "persona_leak_rate": _mean([1.0 if PERSONA_LEAK_RE.search(output) else 0.0 for output in outputs]),
    }


def pass_thresholds(summary: dict[str, float], args: argparse.Namespace) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if summary["n_samples"] < args.min_samples:
        failures.append(f"n_samples {summary['n_samples']:.0f} < {args.min_samples}")
    if summary["mari_aas_mean"] < args.min_aas:
        failures.append(f"mari_aas_mean {summary['mari_aas_mean']:.2f} < {args.min_aas:.2f}")
    if summary["clear_andaluh_rate"] < args.min_clear_andaluh:
        failures.append(
            f"clear_andaluh_rate {summary['clear_andaluh_rate']:.3f} < {args.min_clear_andaluh:.3f}"
        )
    if summary["spanish_leak_mean"] > args.max_spanish_leak:
        failures.append(
            f"spanish_leak_mean {summary['spanish_leak_mean']:.3f} > {args.max_spanish_leak:.3f}"
        )
    if args.gate_persona_leak and summary["persona_leak_rate"] > args.max_persona_leak:
        failures.append(f"persona_leak_rate {summary['persona_leak_rate']:.3f} > {args.max_persona_leak:.3f}")
    if summary["reasoning_preamble_rate"] > args.max_reasoning_preamble:
        failures.append(
            "reasoning_preamble_rate "
            f"{summary['reasoning_preamble_rate']:.3f} > {args.max_reasoning_preamble:.3f}"
        )
    if summary["generation_artifact_rate"] > args.max_generation_artifact:
        failures.append(
            "generation_artifact_rate "
            f"{summary['generation_artifact_rate']:.3f} > {args.max_generation_artifact:.3f}"
        )
    if summary["technical_correctness_rate"] < args.min_technical_correctness:
        failures.append(
            "technical_correctness_rate "
            f"{summary['technical_correctness_rate']:.3f} < {args.min_technical_correctness:.3f}"
        )
    if summary["direct_answer_rate"] < args.min_direct_answer:
        failures.append(f"direct_answer_rate {summary['direct_answer_rate']:.3f} < {args.min_direct_answer:.3f}")
    return not failures, failures


def needs_repair(summary: dict[str, float], args: argparse.Namespace) -> bool:
    coherent = (
        summary["technical_correctness_rate"] >= args.min_technical_correctness
        and summary["mari_aas_mean"] >= args.min_aas
        and summary["clear_andaluh_rate"] >= args.min_clear_andaluh
    )
    remaining_bad_habit = (
        summary["reasoning_preamble_rate"] > args.repair_reasoning_preamble
        or summary["generation_artifact_rate"] > args.repair_generation_artifact
        or summary["spanish_leak_mean"] > args.repair_spanish_leak
        or summary["repetition_rate"] > args.repair_repetition
        or summary["direct_answer_rate"] < args.repair_direct_answer
    )
    return coherent and remaining_bad_habit


def run(args: argparse.Namespace) -> int:
    samples = read_jsonl(args.samples_file)
    summary = summarize(samples, min_aas=args.min_aas, max_leak=args.max_spanish_leak)
    passed, failures = pass_thresholds(summary, args)
    decision = {
        "samples_file": args.samples_file,
        "mode": args.mode,
        "passed": passed,
        "needs_repair": needs_repair(summary, args),
        "failures": failures,
        "summary": summary,
    }
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(
            json.dumps(decision, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    if args.mode == "repair":
        return 0 if decision["needs_repair"] else 1
    return 0 if passed else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_file", required=True)
    parser.add_argument("--mode", choices=["orpo", "release", "repair"], default="orpo")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--min_aas", type=float, default=65.0)
    parser.add_argument("--max_spanish_leak", type=float, default=0.15)
    parser.add_argument("--min_clear_andaluh", type=float, default=0.70)
    parser.add_argument("--max_persona_leak", type=float, default=0.0)
    parser.add_argument("--gate_persona_leak", action="store_true")
    parser.add_argument("--min_samples", type=int, default=5)
    parser.add_argument("--max_reasoning_preamble", type=float, default=0.05)
    parser.add_argument("--max_generation_artifact", type=float, default=0.0)
    parser.add_argument("--min_technical_correctness", type=float, default=0.70)
    parser.add_argument("--min_direct_answer", type=float, default=0.90)
    parser.add_argument("--repair_reasoning_preamble", type=float, default=0.03)
    parser.add_argument("--repair_generation_artifact", type=float, default=0.0)
    parser.add_argument("--repair_spanish_leak", type=float, default=0.07)
    parser.add_argument("--repair_repetition", type=float, default=0.03)
    parser.add_argument("--repair_direct_answer", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
