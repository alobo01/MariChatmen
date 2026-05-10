from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from marichatmen import cli
from marichatmen.config import load_config, merged_options, options_to_env
from marichatmen.resources.anchors import get_anchor_preset, list_anchor_presets, source_urls
from marichatmen.schemas import (
    validate_benchmark_row,
    validate_cpt_row,
    validate_orpo_row,
    validate_sft_row,
)


def test_config_precedence_cli_env_yaml(tmp_path, monkeypatch):
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        "model_name: config-model\nmax_seq_length: 512\nrun_slug: yaml-slug\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MCM_MODEL_NAME", "env-model")

    config = load_config(config_path)
    merged = merged_options(config=config, cli={"max_seq_length": 1024})

    assert merged["model_name"] == "env-model"
    assert merged["max_seq_length"] == 1024
    assert merged["run_slug"] == "yaml-slug"
    assert options_to_env({"model_name": "cli-model"}) == {"MCM_MODEL_NAME": "cli-model"}


def test_cli_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])

    assert exc.value.code == 0
    assert "build-data" in capsys.readouterr().out


def test_cli_dry_run_merges_config_and_cli(tmp_path, capsys, monkeypatch):
    config_path = tmp_path / "run.yaml"
    config_path.write_text("model_name: config-model\nrun_slug: config-slug\n", encoding="utf-8")
    monkeypatch.setenv("MCM_MODEL_NAME", "env-model")

    result = cli.main(
        [
            "train",
            "qwen-andaluh",
            "--config",
            str(config_path),
            "--model-name",
            "cli-model",
            "--dry-run",
        ]
    )

    out = capsys.readouterr().out
    assert result == 0
    assert "scripts/train_qwen_andaluh_local.sh" in out
    assert "MCM_MODEL_NAME=cli-model" in out
    assert "MCM_RUN_SLUG=config-slug" in out


def test_cli_qwen_gate_dispatch(monkeypatch):
    calls = []

    def fake_run(module: str, argv: list[str]) -> int:
        calls.append((module, argv))
        return 0

    monkeypatch.setattr(cli, "_run_python_module", fake_run)

    result = cli.main(["eval", "qwen-gate", "--samples", "samples.jsonl", "--mode", "release"])

    assert result == 0
    assert calls == [
        (
            "marichatmen.eval.qwen_gate",
            ["--samples_file", "samples.jsonl", "--mode", "release"],
        )
    ]


def test_cli_chat_preserves_passthrough_order(monkeypatch):
    calls = []

    def fake_run(module: str, argv: list[str]) -> int:
        calls.append((module, argv))
        return 0

    monkeypatch.setattr(cli, "_run_python_module", fake_run)

    result = cli.main(["chat", "--model_name", "Qwen/test"])

    assert result == 0
    assert calls == [("marichatmen.serve.chat", ["--model_name", "Qwen/test"])]


def test_schema_validators_accept_public_rows():
    message = {"role": "user", "content": "Hola"}
    assistant = {"role": "assistant", "content": "Buenas"}

    assert validate_cpt_row({"text": "texto"})["text"] == "texto"
    assert validate_sft_row({"messages": [message, assistant]})["messages"][0]["role"] == "user"
    assert validate_orpo_row(
        {"prompt": [message], "chosen": [assistant], "rejected": [assistant]}
    )["chosen"][0]["role"] == "assistant"
    assert validate_benchmark_row(
        {"id": "case-1", "prompt": [message], "reference": "Buenas"}
    )["id"] == "case-1"


def test_schema_validators_reject_bad_role():
    with pytest.raises(ValueError, match="Invalid chat role"):
        validate_sft_row({"messages": [{"role": "tool", "content": "x"}]})


def test_anchor_resources_load_from_package_data():
    presets = list_anchor_presets()
    anchors, weights = get_anchor_preset("mari_final")

    assert "mari_final" in presets
    assert anchors
    assert weights
    assert source_urls("sfdk")


def test_default_artifact_root_is_untracked_directory(monkeypatch):
    monkeypatch.delenv("MCM_ARTIFACT_ROOT", raising=False)
    constants = importlib.reload(importlib.import_module("marichatmen.constants"))

    assert constants.ARTIFACT_ROOT == Path(".artifacts")
