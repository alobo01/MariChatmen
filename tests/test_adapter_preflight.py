from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from marichatmen.eval.generation_eval import load_tokenizer
from marichatmen.train.common import assert_adapter_tokenizer_compatible, save_adapter


class FakeTokenizer:
    def __init__(self, size: int, added_vocab: dict[str, int] | None = None):
        self.size = size
        self.added_vocab = added_vocab or {}
        self.chat_template = ""
        self.pad_token = "<|endoftext|>"
        self.eos_token = "<|endoftext|>"

    def __len__(self) -> int:
        return self.size

    def get_added_vocab(self) -> dict[str, int]:
        return self.added_vocab


def write_adapter(tmp_path: Path) -> Path:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "Qwen/Qwen3.5-4B-Base",
                "trainable_token_indices": [100, 101],
            }
        ),
        encoding="utf-8",
    )
    return adapter


def write_adapter_with_trainable_token_dict(tmp_path: Path) -> Path:
    adapter = write_adapter(tmp_path)
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "Qwen/Qwen3.5-4B-Base",
                "trainable_token_indices": {"embed_tokens": [100, 101]},
            }
        ),
        encoding="utf-8",
    )
    return adapter


def patch_auto_tokenizer(monkeypatch, fake_from_pretrained):
    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=fake_from_pretrained)
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def test_adapter_preflight_accepts_matching_tokenizer(tmp_path, monkeypatch):
    adapter = write_adapter(tmp_path)
    expected = FakeTokenizer(128, {"ç": 100, "Andalûh": 101})
    patch_auto_tokenizer(monkeypatch, lambda *_args, **_kwargs: expected)

    assert_adapter_tokenizer_compatible(
        str(adapter),
        FakeTokenizer(128, {"ç": 100, "Andalûh": 101}),
        tokenizer_name=str(adapter),
        model_name="Qwen/Qwen3.5-4B-Base",
    )


def test_adapter_preflight_accepts_trainable_token_dict_shape(tmp_path, monkeypatch):
    adapter = write_adapter_with_trainable_token_dict(tmp_path)
    expected = FakeTokenizer(128, {"ç": 100, "Andalûh": 101})
    patch_auto_tokenizer(monkeypatch, lambda *_args, **_kwargs: expected)

    assert_adapter_tokenizer_compatible(
        str(adapter),
        FakeTokenizer(128, {"ç": 100, "Andalûh": 101}),
        tokenizer_name=str(adapter),
        model_name="Qwen/Qwen3.5-4B-Base",
    )


def test_adapter_preflight_rejects_wrong_base_model(tmp_path, monkeypatch):
    adapter = write_adapter(tmp_path)
    expected = FakeTokenizer(128, {"ç": 100})
    patch_auto_tokenizer(monkeypatch, lambda *_args, **_kwargs: expected)

    with pytest.raises(ValueError, match="Adapter/base-model mismatch"):
        assert_adapter_tokenizer_compatible(
            str(adapter),
            FakeTokenizer(128, {"ç": 100}),
            tokenizer_name=str(adapter),
            model_name="Qwen/Qwen3.5-2B-Base",
        )


def test_adapter_preflight_rejects_short_tokenizer(tmp_path, monkeypatch):
    adapter = write_adapter(tmp_path)
    expected = FakeTokenizer(128, {"ç": 100})
    patch_auto_tokenizer(monkeypatch, lambda *_args, **_kwargs: expected)

    with pytest.raises(ValueError, match="trains token id"):
        assert_adapter_tokenizer_compatible(
            str(adapter),
            FakeTokenizer(101, {"ç": 100}),
            tokenizer_name="plain-qwen-tokenizer",
            model_name="Qwen/Qwen3.5-4B-Base",
        )


def test_adapter_preflight_rejects_added_vocab_mismatch(tmp_path, monkeypatch):
    adapter = write_adapter(tmp_path)
    expected = FakeTokenizer(128, {"ç": 100, "Andalûh": 101})
    patch_auto_tokenizer(monkeypatch, lambda *_args, **_kwargs: expected)

    with pytest.raises(ValueError, match="added-token mismatch"):
        assert_adapter_tokenizer_compatible(
            str(adapter),
            FakeTokenizer(128, {"s": 100, "Andaluh": 101}),
            tokenizer_name="wrong-expanded-tokenizer",
            model_name="Qwen/Qwen3.5-4B-Base",
        )


def test_save_adapter_persists_tokenizer_files(tmp_path):
    class FakeTrainer:
        def save_model(self, output_dir: str) -> None:
            Path(output_dir, "adapter_config.json").write_text("{}", encoding="utf-8")

    class SaveableTokenizer:
        def save_pretrained(self, output_dir: str) -> None:
            Path(output_dir, "tokenizer_config.json").write_text("{}", encoding="utf-8")

    final_dir = save_adapter(FakeTrainer(), str(tmp_path / "output"), SaveableTokenizer())

    assert (final_dir / "adapter_config.json").exists()
    assert (final_dir / "tokenizer_config.json").exists()


def test_eval_load_tokenizer_prefers_adapter_tokenizer(tmp_path, monkeypatch):
    adapter = write_adapter(tmp_path)
    calls = []

    def fake_from_pretrained(name, **_kwargs):
        calls.append(name)
        return FakeTokenizer(128, {"ç": 100})

    patch_auto_tokenizer(monkeypatch, fake_from_pretrained)

    load_tokenizer("Qwen/Qwen3.5-4B-Base", "wrong-tokenizer", str(adapter))

    assert calls == [str(adapter)]
