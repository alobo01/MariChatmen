"""RLOO repair stage for neutral Qwen-Andaluh checkpoints."""

from __future__ import annotations

import argparse
import os

from marichatmen.constants import TIMING_METRICS_FILE, TRAINING_METRICS_FILE
from marichatmen.eval.repair_reward import repair_reward
from marichatmen.train.callbacks import JsonlLogCallback
from marichatmen.train.common import (
    acquire_output_dir_lock_or_skip,
    assert_adapter_tokenizer_compatible,
    bool_arg,
    config_from_supported,
    load_json_dataset,
    load_model_and_tokenizer,
    save_adapter,
    save_training_record,
    write_accelerate_note,
)
from marichatmen.train.timing import timed_stage


def run(args: argparse.Namespace) -> None:
    from peft import PeftModel
    from trl import RLOOConfig, RLOOTrainer

    acquire_output_dir_lock_or_skip(args.output_dir)
    dataset = load_json_dataset(args.prompts_file, args.valid_file or None)
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        tokenizer_name=args.tokenizer_name or None,
        bf16=args.bf16,
        resize_token_embeddings=args.resize_token_embeddings,
    )
    tokenizer.padding_side = "left"
    if not args.adapter and not args.allow_no_adapter:
        raise ValueError(
            "RLOO repair must start from a gated SFT/ORPO adapter. "
            "Pass --allow_no_adapter true only for an explicit trainer smoke test."
        )
    if args.adapter:
        assert_adapter_tokenizer_compatible(
            args.adapter,
            tokenizer,
            tokenizer_name=args.tokenizer_name or args.model_name,
            model_name=args.model_name,
        )
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)

    training_kwargs = {
        "output_dir": args.output_dir,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "bf16": args.bf16,
        "logging_steps": args.logging_steps,
        "eval_strategy": "steps" if args.valid_file else "no",
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "report_to": args.report_to,
        "run_name": args.run_name,
        "remove_unused_columns": False,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": args.dataloader_pin_memory,
        "dataloader_prefetch_factor": args.dataloader_prefetch_factor
        if args.dataloader_num_workers > 0
        else None,
        "num_generations": args.num_generations,
        "generation_batch_size": args.generation_batch_size or args.num_generations,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "beta": args.beta,
        "reward_clip_range": (-1.0, 1.0),
        "mask_truncated_completions": args.mask_truncated_completions,
        "log_completions": args.log_completions,
        "num_completions_to_print": args.num_completions_to_print,
        "use_vllm": False,
    }
    if training_kwargs["dataloader_prefetch_factor"] is None:
        training_kwargs.pop("dataloader_prefetch_factor")
    training_config = config_from_supported(RLOOConfig, **training_kwargs)
    trainer = RLOOTrainer(
        model=model,
        reward_funcs=repair_reward,
        args=training_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        processing_class=tokenizer,
    )
    trainer.add_callback(JsonlLogCallback(TRAINING_METRICS_FILE, args.run_name, "repair_rloo"))
    write_accelerate_note(args.output_dir)
    with timed_stage("repair_rloo_train", TIMING_METRICS_FILE, {"run_id": args.run_name, "model": args.model_name}):
        trainer.train()
    final_dir = save_adapter(trainer, args.output_dir, tokenizer)
    save_training_record(args.run_name, "repair_rloo", args.output_dir, trainer)
    print(f"Saved final adapter to {final_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--tokenizer_name", default="")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--allow_no_adapter", type=bool_arg, default=False)
    parser.add_argument("--prompts_file", required=True)
    parser.add_argument("--valid_file", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=int(os.environ.get("MCM_EVAL_BATCH_SIZE", "1")),
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=int(os.environ.get("MCM_DATALOADER_NUM_WORKERS", "2")),
    )
    parser.add_argument(
        "--dataloader_prefetch_factor",
        type=int,
        default=int(os.environ.get("MCM_DATALOADER_PREFETCH_FACTOR", "2")),
    )
    parser.add_argument("--dataloader_pin_memory", type=bool_arg, default=True)
    parser.add_argument("--gradient_checkpointing", type=bool_arg, default=True)
    parser.add_argument("--resize_token_embeddings", type=bool_arg, default=False)
    parser.add_argument("--bf16", type=bool_arg, default=True)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--generation_batch_size", type=int, default=0)
    parser.add_argument("--max_completion_length", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--mask_truncated_completions", type=bool_arg, default=True)
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--run_name", default="qwen_andaluh_repair_rloo")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--log_completions", type=bool_arg, default=False)
    parser.add_argument("--num_completions_to_print", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
