"""Shared training helpers."""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import Any

from marichatmen.constants import TRAINING_METRICS_FILE
from marichatmen.io import append_jsonl
from marichatmen.tokenizer_templates import ensure_text_training_chat_template

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

_OUTPUT_LOCKS: list[Any] = []


def bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


def load_json_dataset(train_file: str, valid_file: str | None = None):
    from datasets import load_dataset

    data_files: dict[str, str] = {"train": train_file}
    if valid_file:
        data_files["validation"] = valid_file
    dataset = load_dataset("json", data_files=data_files)
    return dataset


def make_lora_config(
    r: int,
    alpha: int,
    dropout: float = 0.05,
    *,
    train_embeddings: bool = False,
    trainable_token_indices: list[int] | None = None,
):
    from peft import LoraConfig

    modules_to_save = None
    if train_embeddings and not trainable_token_indices:
        modules_to_save = ["embed_tokens", "lm_head"]
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
        modules_to_save=modules_to_save,
        trainable_token_indices=trainable_token_indices,
        ensure_weight_tying=bool(trainable_token_indices),
    )


def new_token_indices(tokenizer: Any) -> list[int]:
    base_vocab_size = getattr(tokenizer, "vocab_size", len(tokenizer))
    if len(tokenizer) <= base_vocab_size:
        return []
    return list(range(base_vocab_size, len(tokenizer)))


def make_bnb_config(bf16: bool = True):
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_model_and_tokenizer(
    model_name: str,
    *,
    tokenizer_name: str | None = None,
    bf16: bool = True,
    resize_token_embeddings: bool = False,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name, trust_remote_code=True)
    ensure_text_training_chat_template(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=make_bnb_config(bf16),
        device_map="auto",
        trust_remote_code=True,
    )
    embedding_count = model.get_input_embeddings().num_embeddings
    if len(tokenizer) > embedding_count:
        if not resize_token_embeddings:
            raise ValueError(
                f"Tokenizer has {len(tokenizer)} tokens but model embeddings have "
                f"{embedding_count}. Pass --resize_token_embeddings true for tokenizer "
                "extension experiments."
            )
        model.resize_token_embeddings(len(tokenizer))
        if hasattr(model, "config"):
            model.config.vocab_size = len(tokenizer)
    if hasattr(model, "config"):
        model.config.use_cache = False
    return model, tokenizer


def assert_adapter_tokenizer_compatible(
    adapter_path: str,
    tokenizer: Any,
    *,
    tokenizer_name: str | None = None,
    model_name: str | None = None,
) -> None:
    """Fail early when an adapter is paired with the wrong tokenizer or base model."""

    if not adapter_path:
        return

    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        raise ValueError(f"Adapter path does not exist: {adapter_dir}")

    adapter_config_path = adapter_dir / "adapter_config.json"
    if adapter_config_path.exists():
        try:
            adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid adapter config JSON: {adapter_config_path}") from exc

        adapter_base = str(adapter_config.get("base_model_name_or_path") or "")
        if model_name and adapter_base and adapter_base != model_name:
            raise ValueError(
                "Adapter/base-model mismatch before PEFT load: "
                f"adapter {adapter_dir} was trained for {adapter_base!r}, "
                f"but this run requested {model_name!r}."
            )

        raw_trainable_indices = adapter_config.get("trainable_token_indices") or []
        if isinstance(raw_trainable_indices, dict):
            trainable_indices = [
                int(index)
                for indices in raw_trainable_indices.values()
                for index in (indices or [])
            ]
        else:
            trainable_indices = [int(index) for index in raw_trainable_indices]
        if trainable_indices:
            max_index = max(trainable_indices)
            if max_index >= len(tokenizer):
                raise ValueError(
                    "Adapter/tokenizer mismatch before PEFT load: "
                    f"adapter {adapter_dir} trains token id {max_index}, "
                    f"but tokenizer {tokenizer_name or '<loaded tokenizer>'} has "
                    f"only {len(tokenizer)} tokens."
                )

    tokenizer_files = [
        adapter_dir / "tokenizer.json",
        adapter_dir / "tokenizer.model",
        adapter_dir / "tokenizer_config.json",
    ]
    if not any(path.exists() for path in tokenizer_files):
        raise ValueError(
            "Adapter directory does not contain its tokenizer files. "
            f"Refusing to continue because tokenizer compatibility cannot be proven: {adapter_dir}"
        )

    from transformers import AutoTokenizer

    adapter_tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    if len(adapter_tokenizer) != len(tokenizer):
        raise ValueError(
            "Adapter/tokenizer length mismatch before PEFT load: "
            f"adapter tokenizer at {adapter_dir} has {len(adapter_tokenizer)} tokens, "
            f"but tokenizer {tokenizer_name or '<loaded tokenizer>'} has {len(tokenizer)} tokens. "
            "Use the tokenizer saved inside the adapter directory for the next stage."
        )

    adapter_added = adapter_tokenizer.get_added_vocab()
    loaded_added = tokenizer.get_added_vocab()
    if adapter_added != loaded_added:
        adapter_items = sorted(adapter_added.items())
        loaded_items = sorted(loaded_added.items())
        first_diff = next(
            (
                (expected, actual)
                for expected, actual in zip(adapter_items, loaded_items)
                if expected != actual
            ),
            None,
        )
        if first_diff is None and len(adapter_items) != len(loaded_items):
            first_diff = (
                adapter_items[len(loaded_items) : len(loaded_items) + 1],
                loaded_items[len(adapter_items) : len(adapter_items) + 1],
            )
        raise ValueError(
            "Adapter/tokenizer added-token mismatch before PEFT load: "
            f"adapter {adapter_dir} and tokenizer {tokenizer_name or '<loaded tokenizer>'} "
            f"do not share the same added-token mapping. First difference: {first_diff}. "
            "Use the tokenizer saved inside the adapter directory for the next stage."
        )


def config_from_supported(config_cls: type, **kwargs: Any):
    signature = inspect.signature(config_cls)
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return config_cls(**supported)


def add_length_kwargs(kwargs: dict[str, Any], config_cls: type, max_seq_length: int) -> dict[str, Any]:
    signature = inspect.signature(config_cls)
    if "max_length" in signature.parameters:
        kwargs["max_length"] = max_seq_length
    if "max_seq_length" in signature.parameters:
        kwargs["max_seq_length"] = max_seq_length
    if "max_prompt_length" in signature.parameters:
        kwargs.setdefault("max_prompt_length", max(128, max_seq_length // 2))
    return kwargs


def acquire_output_dir_lock_or_skip(output_dir: str) -> None:
    """Prevent two launcher queues from training into the same adapter directory."""
    import fcntl

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    final_dir = output_path / "final_adapter"
    lock_handle = (output_path / ".train.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"Another process is using {output_path}; waiting for its lock.")
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        if final_dir.exists():
            print(f"Final adapter already exists after waiting: {final_dir}")
            raise SystemExit(0)
    if final_dir.exists():
        print(f"Final adapter already exists; skipping training: {final_dir}")
        raise SystemExit(0)
    _OUTPUT_LOCKS.append(lock_handle)


def save_training_record(run_name: str, stage: str, output_dir: str, trainer: Any | None = None) -> None:
    record: dict[str, Any] = {
        "run_id": run_name,
        "stage": stage,
        "output_dir": output_dir,
    }
    if trainer is not None and getattr(trainer, "state", None) is not None:
        state = trainer.state
        record.update(
            {
                "global_step": getattr(state, "global_step", None),
                "epoch": getattr(state, "epoch", None),
                "log_history": getattr(state, "log_history", None),
            }
        )
    append_jsonl(TRAINING_METRICS_FILE, record)


def add_early_stopping(trainer: Any, patience: int, threshold: float = 0.0) -> None:
    """Attach Transformers early stopping when a validation set is being evaluated."""
    if patience <= 0:
        return
    from transformers import EarlyStoppingCallback

    trainer.add_callback(
        EarlyStoppingCallback(
            early_stopping_patience=patience,
            early_stopping_threshold=threshold,
        )
    )


def save_adapter(trainer: Any, output_dir: str, tokenizer: Any | None = None) -> Path:
    final_dir = Path(output_dir) / "final_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    if tokenizer is None:
        tokenizer = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.save_pretrained(str(final_dir))
    return final_dir


def write_accelerate_note(output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    note = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
    }
    (Path(output_dir) / "run_environment.json").write_text(
        json.dumps(note, indent=2) + "\n",
        encoding="utf-8",
    )
