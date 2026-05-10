"""QLoRA continual-pretraining entrypoint for Qwen-Andaluh."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

from marichatmen.constants import TIMING_METRICS_FILE, TRAINING_METRICS_FILE
from marichatmen.eval.generation_eval import generate_response
from marichatmen.eval.mari_aas import score_dict
from marichatmen.eval.quality_metrics import (
    direct_answer_score,
    has_generation_artifact,
    has_reasoning_preamble,
    repetition_rate,
    technical_correctness_score,
)
from marichatmen.train.generation_probes import PROMPT_LEAK_RE
from marichatmen.io import append_jsonl, iter_jsonl
from marichatmen.train.callbacks import JsonlLogCallback
from marichatmen.train.common import (
    add_early_stopping,
    assert_adapter_tokenizer_compatible,
    acquire_output_dir_lock_or_skip,
    bool_arg,
    config_from_supported,
    load_json_dataset,
    load_model_and_tokenizer,
    make_lora_config,
    new_token_indices,
    save_adapter,
    save_training_record,
    write_accelerate_note,
)
from marichatmen.train.timing import timed_stage

try:
    from transformers import TrainerCallback
except Exception:  # pragma: no cover
    TrainerCallback = object  # type: ignore


def _safe_exp(loss: float) -> float:
    return math.exp(min(20.0, max(0.0, loss)))


def _load_probe_batches(
    path: str,
    tokenizer: Any,
    *,
    max_seq_length: int,
    limit: int,
) -> list[dict[str, Any]]:
    if not path or not Path(path).exists():
        return []
    batches = []
    for row in iter_jsonl(path):
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        if encoded["input_ids"].shape[-1] < 2:
            continue
        batches.append(encoded)
        if len(batches) >= limit:
            break
    return batches


def _load_generation_prompts(path: str, limit: int) -> list[str]:
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


class ProbePerplexityCallback(TrainerCallback):
    def __init__(
        self,
        *,
        run_name: str,
        tokenizer: Any,
        spanish_eval_file: str,
        andaluh_eval_file: str,
        max_seq_length: int,
        limit: int,
    ) -> None:
        self.run_name = run_name
        self.probes = {
            "spanish": _load_probe_batches(
                spanish_eval_file,
                tokenizer,
                max_seq_length=max_seq_length,
                limit=limit,
            ),
            "andaluh": _load_probe_batches(
                andaluh_eval_file,
                tokenizer,
                max_seq_length=max_seq_length,
                limit=limit,
            ),
        }

    def _loss(self, model: Any, batches: list[dict[str, Any]]) -> float | None:
        if not batches:
            return None
        import torch

        was_training = model.training
        model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for batch in batches:
                moved = {key: value.to(model.device) for key, value in batch.items()}
                output = model(**moved, labels=moved["input_ids"])
                losses.append(float(output.loss.detach().cpu()))
        if was_training:
            model.train()
        return sum(losses) / len(losses)

    def on_evaluate(self, args: Any, state: Any, control: Any, model: Any | None = None, **_: Any):
        if model is None:
            return control
        row: dict[str, Any] = {
            "run_id": self.run_name,
            "stage": "cpt_probe",
            "global_step": getattr(state, "global_step", None),
        }
        for name, batches in self.probes.items():
            loss = self._loss(model, batches)
            if loss is None:
                continue
            row[f"{name}_eval_loss"] = loss
            row[f"{name}_perplexity"] = _safe_exp(loss)
        append_jsonl(TRAINING_METRICS_FILE, row)
        return control


class GenerationProbeCallback(TrainerCallback):
    def __init__(
        self,
        *,
        run_name: str,
        tokenizer: Any,
        prompts_file: str,
        output_jsonl: str,
        limit: int,
        max_new_tokens: int,
        include_explicit_mode: bool = False,
    ) -> None:
        self.run_name = run_name
        self.tokenizer = tokenizer
        self.prompts = _load_generation_prompts(prompts_file, limit)
        self.output_jsonl = output_jsonl
        self.max_new_tokens = max_new_tokens
        self.include_explicit_mode = include_explicit_mode

    def on_evaluate(self, args: Any, state: Any, control: Any, model: Any | None = None, **_: Any):
        if model is None or not self.prompts:
            return control
        was_training = model.training
        model.eval()
        summary = {
            "run_id": self.run_name,
            "stage": "cpt_generation_probe",
            "global_step": getattr(state, "global_step", None),
        }
        modes = ["default", "explicit"] if self.include_explicit_mode else ["default"]
        mode_rows: dict[str, list[dict[str, float]]] = {mode: [] for mode in modes}
        for mode in modes:
            for index, prompt in enumerate(self.prompts):
                user = prompt if mode == "default" else f"Respóndeme en Andalûh: {prompt}"
                output = generate_response(
                    model,
                    self.tokenizer,
                    [
                        {"role": "system", "content": "Eres un asistente"},
                        {"role": "user", "content": user},
                    ],
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.3,
                    top_p=0.9,
                    top_k=20,
                    repetition_penalty=1.08,
                    no_repeat_ngram_size=4,
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
                mode_rows[mode].append(quality)
                append_jsonl(
                    self.output_jsonl,
                    {
                        "run_id": self.run_name,
                        "stage": "cpt",
                        "global_step": getattr(state, "global_step", None),
                        "mode": mode,
                        "prompt_index": index,
                        "system_prompt": "Eres un asistente",
                        "prompt": prompt,
                        "user_message": user,
                        "output": output,
                        "metrics": metrics,
                        "quality": quality,
                    },
                )
        for mode, rows in mode_rows.items():
            denom = max(1, len(rows))
            summary[f"{mode}_andaluh_rate"] = sum(row["clear_andaluh"] for row in rows) / denom
            summary[f"{mode}_reasoning_preamble_rate"] = sum(row["reasoning_preamble"] for row in rows) / denom
            summary[f"{mode}_generation_artifact_rate"] = (
                sum(row["generation_artifact"] for row in rows) / denom
            )
            summary[f"{mode}_direct_answer_rate"] = sum(row["direct_answer"] for row in rows) / denom
            summary[f"{mode}_technical_correctness_rate"] = sum(row["technical_correctness"] for row in rows) / denom
            summary[f"{mode}_repetition_rate"] = sum(row["repetition_rate"] for row in rows) / denom
        append_jsonl(TRAINING_METRICS_FILE, summary)
        if was_training:
            model.train()
        return control


def _tokenize_dataset(dataset: Any, tokenizer: Any, max_seq_length: int, num_proc: int):
    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    columns = dataset["train"].column_names
    map_kwargs: dict[str, Any] = {
        "batched": True,
        "remove_columns": columns,
        "desc": "Tokenizing CPT rows",
    }
    if num_proc > 1:
        map_kwargs["num_proc"] = num_proc
    return dataset.map(tokenize, **map_kwargs)


def run(args: argparse.Namespace) -> None:
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    acquire_output_dir_lock_or_skip(args.output_dir)
    raw_dataset = load_json_dataset(args.train_file, args.valid_file)
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        tokenizer_name=args.tokenizer_name or None,
        bf16=args.bf16,
        resize_token_embeddings=args.resize_token_embeddings,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=args.gradient_checkpointing,
    )
    if args.base_adapter:
        assert_adapter_tokenizer_compatible(
            args.base_adapter,
            tokenizer,
            tokenizer_name=args.tokenizer_name or args.model_name,
            model_name=args.model_name,
        )
        model = PeftModel.from_pretrained(model, args.base_adapter, is_trainable=True)
    else:
        model = get_peft_model(
            model,
            make_lora_config(
                args.lora_r,
                args.lora_alpha,
                args.lora_dropout,
                train_embeddings=args.train_embeddings,
                trainable_token_indices=new_token_indices(tokenizer) if args.train_embeddings else None,
            ),
        )
    tokenized = _tokenize_dataset(
        raw_dataset,
        tokenizer,
        args.max_seq_length,
        args.preprocessing_num_workers,
    )

    training_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "weight_decay": args.weight_decay,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "bf16": args.bf16,
        "logging_steps": args.logging_steps,
        "eval_strategy": "steps",
        "evaluation_strategy": "steps",
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "load_best_model_at_end": args.load_best_model_at_end,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": args.report_to,
        "run_name": args.run_name,
        "remove_unused_columns": False,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": args.dataloader_pin_memory,
    }
    if args.dataloader_num_workers > 0:
        training_kwargs["dataloader_prefetch_factor"] = args.dataloader_prefetch_factor
    training_args = config_from_supported(TrainingArguments, **training_kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.add_callback(JsonlLogCallback(TRAINING_METRICS_FILE, args.run_name, "cpt"))
    add_early_stopping(
        trainer,
        patience=args.early_stopping_patience,
        threshold=args.early_stopping_threshold,
    )
    trainer.add_callback(
        ProbePerplexityCallback(
            run_name=args.run_name,
            tokenizer=tokenizer,
            spanish_eval_file=args.spanish_eval_file,
            andaluh_eval_file=args.andaluh_eval_file,
            max_seq_length=args.max_seq_length,
            limit=args.probe_limit,
        )
    )
    if args.generation_probe_file:
        trainer.add_callback(
            GenerationProbeCallback(
                run_name=args.run_name,
                tokenizer=tokenizer,
                prompts_file=args.generation_probe_file,
                output_jsonl=args.generation_probe_output
                or str(Path(args.output_dir).parent.parent / "reports" / "samples" / f"{args.run_name}_cpt_generation_probes.jsonl"),
                limit=args.generation_probe_limit,
                max_new_tokens=args.generation_probe_max_new_tokens,
                include_explicit_mode=args.generation_probe_include_explicit,
            )
        )
    write_accelerate_note(args.output_dir)
    with timed_stage("cpt_train", TIMING_METRICS_FILE, {"run_id": args.run_name, "model": args.model_name}):
        trainer.train()
    final_dir = save_adapter(trainer, args.output_dir, tokenizer)
    save_training_record(args.run_name, "cpt", args.output_dir, trainer)
    print(f"Saved final CPT adapter to {final_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--tokenizer_name", default="")
    parser.add_argument("--base_adapter", default="")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--valid_file", required=True)
    parser.add_argument("--spanish_eval_file", default="")
    parser.add_argument("--andaluh_eval_file", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=float, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=int(os.environ.get("MCM_EVAL_BATCH_SIZE", "4")),
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=int(os.environ.get("MCM_DATALOADER_NUM_WORKERS", "4")),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=int(os.environ.get("MCM_PREPROCESSING_NUM_WORKERS", "8")),
    )
    parser.add_argument(
        "--dataloader_prefetch_factor",
        type=int,
        default=int(os.environ.get("MCM_DATALOADER_PREFETCH_FACTOR", "2")),
    )
    parser.add_argument("--dataloader_pin_memory", type=bool_arg, default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--resize_token_embeddings", type=bool_arg, default=False)
    parser.add_argument("--train_embeddings", type=bool_arg, default=False)
    parser.add_argument("--gradient_checkpointing", type=bool_arg, default=True)
    parser.add_argument("--bf16", type=bool_arg, default=True)
    parser.add_argument("--report_to", default="wandb")
    parser.add_argument("--run_name", default="qwen_andaluh_cpt")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--probe_limit", type=int, default=32)
    parser.add_argument("--generation_probe_file", default="")
    parser.add_argument("--generation_probe_output", default="")
    parser.add_argument("--generation_probe_limit", type=int, default=5)
    parser.add_argument("--generation_probe_max_new_tokens", type=int, default=128)
    parser.add_argument("--generation_probe_include_explicit", type=bool_arg, default=False)
    parser.add_argument("--load_best_model_at_end", type=bool_arg, default=False)
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
