"""QLoRA ORPO training entrypoint."""

from __future__ import annotations

import argparse

from marichatmen.constants import TIMING_METRICS_FILE, TRAINING_METRICS_FILE
from marichatmen.train.callbacks import JsonlLogCallback
from marichatmen.train.common import (
    add_early_stopping,
    add_length_kwargs,
    bool_arg,
    config_from_supported,
    acquire_output_dir_lock_or_skip,
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
    try:
        from trl import ORPOConfig, ORPOTrainer
    except ImportError:
        from trl.experimental.orpo import ORPOConfig, ORPOTrainer

    acquire_output_dir_lock_or_skip(args.output_dir)
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        tokenizer_name=args.tokenizer_name or None,
        bf16=args.bf16,
        resize_token_embeddings=args.resize_token_embeddings,
    )
    dataset = load_json_dataset(args.train_file, args.valid_file)
    if args.filter_token_budget:
        dataset = filter_orpo_token_budget(
            dataset,
            tokenizer,
            max_seq_length=args.max_seq_length,
            max_prompt_tokens=args.max_prompt_tokens,
            min_completion_tokens=args.min_completion_tokens,
        )
    peft_config = None
    if args.sft_adapter:
        model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)
    else:
        peft_config = make_lora_config(
            args.lora_r,
            args.lora_alpha,
            0.05,
            train_embeddings=args.train_embeddings,
            trainable_token_indices=new_token_indices(tokenizer) if args.train_embeddings else None,
        )
    training_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "max_grad_norm": args.max_grad_norm,
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
        "beta": args.beta,
        "max_completion_length": args.max_completion_length or None,
        "remove_unused_columns": False,
    }
    add_length_kwargs(training_kwargs, ORPOConfig, args.max_seq_length)
    training_config = config_from_supported(ORPOConfig, **training_kwargs)
    trainer = ORPOTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.add_callback(JsonlLogCallback(TRAINING_METRICS_FILE, args.run_name, "orpo"))
    add_early_stopping(
        trainer,
        patience=args.early_stopping_patience,
        threshold=args.early_stopping_threshold,
    )
    write_accelerate_note(args.output_dir)
    with timed_stage("orpo_train", TIMING_METRICS_FILE, {"run_id": args.run_name, "model": args.model_name}):
        trainer.train()
    final_dir = save_adapter(trainer, args.output_dir)
    save_training_record(args.run_name, "orpo", args.output_dir, trainer)
    print(f"Saved final adapter to {final_dir}")


def _chat_tokens(tokenizer, messages: list[dict[str, str]], *, add_generation_prompt: bool = False) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if "input_ids" in encoded:
        return encoded["input_ids"]
    return encoded


def filter_orpo_token_budget(
    dataset,
    tokenizer,
    *,
    max_seq_length: int,
    max_prompt_tokens: int,
    min_completion_tokens: int,
):
    """Drop ORPO rows that would lose assistant labels after max-length truncation."""

    def keep(row: dict) -> bool:
        prompt = row.get("prompt") or []
        chosen = row.get("chosen") or []
        rejected = row.get("rejected") or []
        if not prompt or not chosen or not rejected:
            return False

        try:
            prompt_ids = _chat_tokens(tokenizer, prompt, add_generation_prompt=True)
            chosen_ids = _chat_tokens(tokenizer, prompt + chosen)
            rejected_ids = _chat_tokens(tokenizer, prompt + rejected)
        except Exception:
            return False

        chosen_completion = max(0, len(chosen_ids) - len(prompt_ids))
        rejected_completion = max(0, len(rejected_ids) - len(prompt_ids))
        if len(prompt_ids) > max_prompt_tokens:
            return False
        if chosen_completion < min_completion_tokens or rejected_completion < min_completion_tokens:
            return False
        return len(chosen_ids) <= max_seq_length and len(rejected_ids) <= max_seq_length

    filtered = dataset.filter(
        keep,
        desc="Filtering ORPO rows by token budget",
        load_from_cache_file=False,
    )
    for split, split_dataset in filtered.items():
        if len(split_dataset) == 0:
            raise ValueError(
                f"ORPO {split} split is empty after token-budget filtering. "
                "Increase --max_seq_length/--max_prompt_tokens or lower --min_completion_tokens."
            )
        before = len(dataset[split])
        after = len(split_dataset)
        print(f"Kept {after}/{before} {split} ORPO rows after token-budget filtering")
    return filtered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--tokenizer_name", default="")
    parser.add_argument("--sft_adapter", default="")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--valid_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=float, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=24)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--resize_token_embeddings", type=bool_arg, default=False)
    parser.add_argument("--train_embeddings", type=bool_arg, default=False)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_completion_length", type=int, default=0)
    parser.add_argument("--filter_token_budget", type=bool_arg, default=True)
    parser.add_argument("--max_prompt_tokens", type=int, default=384)
    parser.add_argument("--min_completion_tokens", type=int, default=8)
    parser.add_argument("--gradient_checkpointing", type=bool_arg, default=True)
    parser.add_argument("--bf16", type=bool_arg, default=True)
    parser.add_argument("--report_to", default="wandb")
    parser.add_argument("--run_name", default="marichatmen_orpo")
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
