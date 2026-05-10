"""Trainer callbacks for fixed neutral generation probes."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from marichatmen.constants import TRAINING_METRICS_FILE
from marichatmen.eval.generation_eval import generate_response
from marichatmen.eval.mari_aas import score_dict
from marichatmen.eval.quality_metrics import (
    direct_answer_score,
    has_generation_artifact,
    has_reasoning_preamble,
    repetition_rate,
    technical_correctness_score,
)
from marichatmen.io import append_jsonl

try:
    from transformers import TrainerCallback
except Exception:  # pragma: no cover
    TrainerCallback = object  # type: ignore


PROMPT_LEAK_RE = re.compile(
    r"\b(andal[uúû]h?|epa|seseo|sevillan[oa]s?|çebiyan[oa]s?|dialect|translit|"
    r"responde\s+en|respóndeme\s+en|contesta\s+en|responde\s+siempre|"
    r"contesta\s+siempre)\b",
    re.IGNORECASE,
)


def load_generation_prompts(path: str, limit: int) -> list[str]:
    if not path or not Path(path).exists():
        return []
    prompts = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        prompt = line.strip()
        if not prompt:
            continue
        match = PROMPT_LEAK_RE.search(prompt)
        if match:
            raise ValueError(
                "Neutral generation probe prompt leaks the target style: "
                f"{match.group(0)!r} in {path}:{line_number}"
            )
        prompts.append(prompt)
    return prompts[:limit]


class NeutralGenerationProbeCallback(TrainerCallback):
    """Save five fixed neutral generations after eval checkpoints.

    The default mode intentionally avoids asking for Andaluh/EPA in the prompt.
    CPT can opt into an additional explicit-capacity mode, but SFT/ORPO should
    use neutral-only probes for default-behaviour gating.
    """

    def __init__(
        self,
        *,
        run_name: str,
        stage: str,
        tokenizer: Any,
        prompts_file: str,
        output_jsonl: str,
        limit: int,
        max_new_tokens: int,
        system_prompt: str = "Eres un asistente",
        repetition_penalty: float = 1.08,
        no_repeat_ngram_size: int = 4,
        include_explicit_mode: bool = False,
    ) -> None:
        self.run_name = run_name
        self.stage = stage
        self.tokenizer = tokenizer
        self.prompts = load_generation_prompts(prompts_file, limit)
        self.output_jsonl = output_jsonl
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.include_explicit_mode = include_explicit_mode

    def on_evaluate(self, args: Any, state: Any, control: Any, model: Any | None = None, **_: Any):
        if model is None or not self.prompts:
            return control
        was_training = model.training
        model.eval()
        modes = ["default", "explicit"] if self.include_explicit_mode else ["default"]
        summary = {
            "run_id": self.run_name,
            "stage": f"{self.stage}_generation_probe",
            "global_step": getattr(state, "global_step", None),
        }
        try:
            for mode in modes:
                rows: list[dict[str, float]] = []
                for index, prompt in enumerate(self.prompts):
                    user = prompt if mode == "default" else f"Respóndeme en Andalûh: {prompt}"
                    output = generate_response(
                        model,
                        self.tokenizer,
                        [
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": user},
                        ],
                        max_new_tokens=self.max_new_tokens,
                        temperature=0.3,
                        top_p=0.9,
                        top_k=20,
                        repetition_penalty=self.repetition_penalty,
                        no_repeat_ngram_size=self.no_repeat_ngram_size,
                        disable_thinking=True,
                    )
                    metrics = score_dict(output)
                    quality = {
                        "clear_andaluh": float(metrics["score"] >= 65.0 and metrics["spanish_leak"] <= 0.15),
                        "generation_artifact": float(has_generation_artifact(output)),
                        "reasoning_preamble": float(has_reasoning_preamble(output)),
                        "direct_answer": direct_answer_score(output),
                        "technical_correctness": technical_correctness_score(prompt, output),
                        "repetition_rate": repetition_rate(output),
                    }
                    rows.append(quality)
                    append_jsonl(
                        self.output_jsonl,
                        {
                            "run_id": self.run_name,
                            "stage": self.stage,
                            "global_step": getattr(state, "global_step", None),
                            "mode": mode,
                            "prompt_index": index,
                            "system_prompt": self.system_prompt,
                            "prompt": prompt,
                            "user_message": user,
                            "output": output,
                            "metrics": metrics,
                            "quality": quality,
                        },
                    )
                denom = max(1, len(rows))
                summary[f"{mode}_andaluh_rate"] = sum(row["clear_andaluh"] for row in rows) / denom
                summary[f"{mode}_reasoning_preamble_rate"] = sum(row["reasoning_preamble"] for row in rows) / denom
                summary[f"{mode}_generation_artifact_rate"] = sum(row["generation_artifact"] for row in rows) / denom
                summary[f"{mode}_direct_answer_rate"] = sum(row["direct_answer"] for row in rows) / denom
                summary[f"{mode}_technical_correctness_rate"] = (
                    sum(row["technical_correctness"] for row in rows) / denom
                )
                summary[f"{mode}_repetition_rate"] = sum(row["repetition_rate"] for row in rows) / denom
            append_jsonl(TRAINING_METRICS_FILE, summary)
        finally:
            try:
                import gc
                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            if was_training:
                model.train()
        return control
