"""QLoRA SFT training entrypoint."""

from __future__ import annotations

import argparse

from marichatmen.constants import TIMING_METRICS_FILE, TRAINING_METRICS_FILE
from marichatmen.train.callbacks import JsonlLogCallback
from marichatmen.train.common import (
    add_length_kwargs,
    bool_arg,
    config_from_supported,
    acquire_output_dir_lock_or_skip,
    add_early_stopping,
    load_json_dataset,
    load_model_and_tokenizer,
    make_lora_config,
    new_token_indices,
    save_adapter,
    save_training_record,
    write_accelerate_note,
)
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
    peft_config = None
    if args.base_adapter:
        model = PeftModel.from_pretrained(model, args.base_adapter, is_trainable=True)
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
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "bf16": args.bf16,
        "logging_steps": args.logging_steps,
        "eval_strategy": "steps",
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "load_best_model_at_end": args.load_best_model_at_end,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": args.report_to,
        "run_name": args.run_name,
        "packing": False,
        "assistant_only_loss": args.assistant_only_loss,
        "remove_unused_columns": False,
    }
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
    add_early_stopping(
        trainer,
        patience=args.early_stopping_patience,
        threshold=args.early_stopping_threshold,
    )
    write_accelerate_note(args.output_dir)
    with timed_stage("sft_train", TIMING_METRICS_FILE, {"run_id": args.run_name, "model": args.model_name}):
        trainer.train()
    final_dir = save_adapter(trainer, args.output_dir)
    save_training_record(args.run_name, "sft", args.output_dir, trainer)
    print(f"Saved final adapter to {final_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--tokenizer_name", default="")
    parser.add_argument("--base_adapter", default="")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--valid_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=float, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
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
    parser.add_argument("--load_best_model_at_end", type=bool_arg, default=False)
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
