"""Optional GRPO accent optimization entrypoint."""

from __future__ import annotations

import argparse

from marichatmen.constants import TIMING_METRICS_FILE, TRAINING_METRICS_FILE
from marichatmen.eval.mari_aas import score_text as aas_score_text
from marichatmen.eval.mari_pas import score_text as pas_score_text
from marichatmen.eval.mari_reward import score_mari_answer
from marichatmen.train.callbacks import JsonlLogCallback
from marichatmen.train.common import (
    bool_arg,
    config_from_supported,
    acquire_output_dir_lock_or_skip,
    assert_adapter_tokenizer_compatible,
    load_json_dataset,
    load_model_and_tokenizer,
    save_adapter,
    save_training_record,
    write_accelerate_note,
)
from marichatmen.train.timing import timed_stage


def _completion_text(completion) -> str:
    if isinstance(completion, list):
        return completion[-1].get("content", "") if completion else ""
    return str(completion)


def mari_aas_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        rewards.append(aas_score_text(_completion_text(completion)).score / 100)
    return rewards


def mari_persona_reward(completions, **kwargs):
    return [pas_score_text(_completion_text(completion)).score / 100 for completion in completions]


def province_flourish_reward(completions, **kwargs):
    return [
        pas_score_text(_completion_text(completion)).province_flourish for completion in completions
    ]


def helpfulness_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        scored = aas_score_text(_completion_text(completion))
        rewards.append(max(0.0, min(1.0, scored.chrfpp_reference + 0.25 * (1 - scored.degeneration_penalty))))
    return rewards


def self_verified_mari_reward(completions, prompts=None, **kwargs):
    rewards = []
    prompts = prompts or kwargs.get("prompt") or [""] * len(completions)
    for prompt, completion in zip(prompts, completions):
        if isinstance(prompt, list):
            prompt_text = next(
                (
                    item.get("content", "")
                    for item in reversed(prompt)
                    if isinstance(item, dict) and item.get("role") == "user"
                ),
                "",
            )
        else:
            prompt_text = str(prompt)
        rewards.append(score_mari_answer(prompt_text, _completion_text(completion)).reward)
    return rewards


def caricature_penalty_reward(completions, **kwargs):
    return [aas_score_text(_completion_text(completion)).caricature_penalty for completion in completions]


def regional_hostility_penalty_reward(completions, **kwargs):
    return [
        max(
            pas_score_text(_completion_text(completion)).regional_hostility_penalty,
            pas_score_text(_completion_text(completion)).alcohol_safety_penalty,
        )
        for completion in completions
    ]


REWARD_FUNCS = [
    mari_aas_reward,
    mari_persona_reward,
    province_flourish_reward,
    helpfulness_reward,
    self_verified_mari_reward,
    caricature_penalty_reward,
    regional_hostility_penalty_reward,
]

REWARD_WEIGHTS = [0.20, 0.20, 0.10, 0.10, 0.40, -0.25, -0.25]


def run(args: argparse.Namespace) -> None:
    from peft import PeftModel
    from trl import GRPOConfig, GRPOTrainer

    acquire_output_dir_lock_or_skip(args.output_dir)
    dataset = load_json_dataset(args.prompts_file)
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        tokenizer_name=args.tokenizer_name or None,
        bf16=args.bf16,
        resize_token_embeddings=args.resize_token_embeddings,
    )
    if args.sft_or_orpo_adapter:
        assert_adapter_tokenizer_compatible(
            args.sft_or_orpo_adapter,
            tokenizer,
            tokenizer_name=args.tokenizer_name or args.model_name,
            model_name=args.model_name,
        )
        model = PeftModel.from_pretrained(model, args.sft_or_orpo_adapter, is_trainable=True)

    training_kwargs = {
        "output_dir": args.output_dir,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "bf16": args.bf16,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "report_to": args.report_to,
        "run_name": args.run_name,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "reward_weights": REWARD_WEIGHTS,
        "remove_unused_columns": False,
    }
    training_config = config_from_supported(GRPOConfig, **training_kwargs)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=REWARD_FUNCS,
        args=training_config,
        train_dataset=dataset["train"],
        processing_class=tokenizer,
    )
    trainer.add_callback(JsonlLogCallback(TRAINING_METRICS_FILE, args.run_name, "grpo"))
    write_accelerate_note(args.output_dir)
    with timed_stage("grpo_train", TIMING_METRICS_FILE, {"run_id": args.run_name, "model": args.model_name}):
        trainer.train()
    final_dir = save_adapter(trainer, args.output_dir, tokenizer)
    save_training_record(args.run_name, "grpo", args.output_dir, trainer)
    print(f"Saved final adapter to {final_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--tokenizer_name", default="")
    parser.add_argument("--sft_or_orpo_adapter", default="")
    parser.add_argument("--prompts_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_prompt_length", type=int, default=384)
    parser.add_argument("--max_completion_length", type=int, default=192)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--resize_token_embeddings", type=bool_arg, default=False)
    parser.add_argument("--bf16", type=bool_arg, default=True)
    parser.add_argument("--report_to", default="wandb")
    parser.add_argument("--run_name", default="marichatmen_grpo")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
