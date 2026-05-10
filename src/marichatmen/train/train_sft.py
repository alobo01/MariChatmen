"""QLoRA SFT training entrypoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from marichatmen.constants import TIMING_METRICS_FILE, TRAINING_METRICS_FILE
from marichatmen.train.callbacks import JsonlLogCallback
from marichatmen.train.common import (
    add_length_kwargs,
    bool_arg,
    config_from_supported,
    acquire_output_dir_lock_or_skip,
    add_early_stopping,
    assert_adapter_tokenizer_compatible,
    load_json_dataset,
    load_model_and_tokenizer,
    make_lora_config,
    new_token_indices,
    save_adapter,
    save_training_record,
    write_accelerate_note,
)
from marichatmen.train.generation_probes import NeutralGenerationProbeCallback
from marichatmen.train.timing import timed_stage


def run(args: argparse.Namespace) -> None:
    from peft import PeftModel
    from trl import SFTConfig, SFTTrainer

    acquire_output_dir_lock_or_skip(args.output_dir)
    dataset = load_json_dataset(args.train_file, args.valid_file)
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        tokenizer_name=args.tokenizer_name or None,
        bf16=args.bf16,
        resize_token_embeddings=args.resize_token_embeddings,
    )
    if args.filter_token_budget:
        dataset = filter_sft_token_budget(
            dataset,
            tokenizer,
            max_seq_length=args.max_seq_length,
            max_prompt_tokens=args.max_prompt_tokens,
            min_assistant_tokens=args.min_assistant_tokens,
        )
    peft_config = None
    if args.base_adapter:
        assert_adapter_tokenizer_compatible(
            args.base_adapter,
            tokenizer,
            tokenizer_name=args.tokenizer_name or args.model_name,
            model_name=args.model_name,
        )
        model = PeftModel.from_pretrained(model, args.base_adapter, is_trainable=True)
        if args.freeze_base_adapter_token_deltas:
            frozen = 0
            for name, parameter in model.named_parameters():
                if "token_adapter" in name or "trainable_tokens_delta" in name:
                    parameter.requires_grad = False
                    frozen += parameter.numel()
            print(f"Froze {frozen:,} base-adapter token-delta parameters for SFT")
    else:
        peft_config = make_lora_config(
            args.lora_r,
            args.lora_alpha,
            args.lora_dropout,
            train_embeddings=args.train_embeddings,
            trainable_token_indices=new_token_indices(tokenizer) if args.train_embeddings else None,
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
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": args.load_best_model_at_end,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": args.report_to,
        "run_name": args.run_name,
        "packing": False,
        "assistant_only_loss": args.assistant_only_loss,
        "max_prompt_length": args.max_prompt_tokens,
        "remove_unused_columns": False,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": args.dataloader_pin_memory,
        "dataset_num_proc": args.preprocessing_num_workers,
    }
    if args.dataloader_num_workers > 0:
        training_kwargs["dataloader_prefetch_factor"] = args.dataloader_prefetch_factor
    add_length_kwargs(training_kwargs, SFTConfig, args.max_seq_length)
    training_config = config_from_supported(SFTConfig, **training_kwargs)
    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.add_callback(JsonlLogCallback(TRAINING_METRICS_FILE, args.run_name, "sft"))
    if args.generation_probe_file:
        trainer.add_callback(
            NeutralGenerationProbeCallback(
                run_name=args.run_name,
                stage="sft",
                tokenizer=tokenizer,
                prompts_file=args.generation_probe_file,
                output_jsonl=args.generation_probe_output
                or str(Path(args.output_dir) / "generation_probes.jsonl"),
                limit=args.generation_probe_limit,
                max_new_tokens=args.generation_probe_max_new_tokens,
                system_prompt=args.generation_probe_system_prompt,
            )
        )
    add_early_stopping(
        trainer,
        patience=args.early_stopping_patience,
        threshold=args.early_stopping_threshold,
    )
    write_accelerate_note(args.output_dir)
    with timed_stage("sft_train", TIMING_METRICS_FILE, {"run_id": args.run_name, "model": args.model_name}):
        trainer.train()
    final_dir = save_adapter(trainer, args.output_dir, tokenizer)
    save_training_record(args.run_name, "sft", args.output_dir, trainer)
    print(f"Saved final adapter to {final_dir}")


def _render_messages(tokenizer, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception:
        rendered = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            rendered.append(f"{role}: {content}")
        if add_generation_prompt:
            rendered.append("assistant:")
        return "\n".join(rendered)


def filter_sft_token_budget(
    dataset,
    tokenizer,
    *,
    max_seq_length: int,
    max_prompt_tokens: int,
    min_assistant_tokens: int,
):
    def keep(row: dict) -> bool:
        messages = row.get("messages") or []
        if not messages:
            return False
        prompt_messages = [m for m in messages if m.get("role") != "assistant"]
        assistant_text = "\n".join(
            m.get("content", "") for m in messages if m.get("role") == "assistant"
        ).strip()
        if not assistant_text:
            return False
        prompt_text = _render_messages(tokenizer, prompt_messages, add_generation_prompt=True)
        full_text = _render_messages(tokenizer, messages, add_generation_prompt=False)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        assistant_ids = tokenizer(assistant_text, add_special_tokens=False).input_ids
        full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
        if len(prompt_ids) > max_prompt_tokens:
            return False
        if len(assistant_ids) < min_assistant_tokens:
            return False
        return len(full_ids) <= max_seq_length

    filtered = {}
    for split_name, split in dataset.items():
        before = len(split)
        kept = split.filter(keep, desc=f"Filtering SFT {split_name} rows by token budget")
        if len(kept) == 0:
            raise ValueError(
                f"SFT {split_name} split is empty after token-budget filtering. "
                "Increase --max_seq_length/--max_prompt_tokens or lower --min_assistant_tokens."
            )
        print(f"Kept {len(kept)}/{before} {split_name} SFT rows after token-budget filtering")
        filtered[split_name] = kept
    return dataset.__class__(filtered)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--tokenizer_name", default="")
    parser.add_argument("--base_adapter", default="")
    parser.add_argument("--freeze_base_adapter_token_deltas", type=bool_arg, default=False)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--valid_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=float, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
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
    parser.add_argument("--run_name", default="marichatmen_sft")
    parser.add_argument("--assistant_only_loss", type=bool_arg, default=True)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--load_best_model_at_end", type=bool_arg, default=False)
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0)
    parser.add_argument("--filter_token_budget", type=bool_arg, default=True)
    parser.add_argument("--max_prompt_tokens", type=int, default=384)
    parser.add_argument("--min_assistant_tokens", type=int, default=8)
    parser.add_argument("--generation_probe_file", default="")
    parser.add_argument("--generation_probe_output", default="")
    parser.add_argument("--generation_probe_limit", type=int, default=5)
    parser.add_argument("--generation_probe_max_new_tokens", type=int, default=128)
    parser.add_argument("--generation_probe_system_prompt", default="Eres un asistente")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
