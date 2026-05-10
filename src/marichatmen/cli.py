"""Unified command line interface for MariChatmen."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from marichatmen.config import load_config, merged_options, options_to_env, parse_key_value


ROOT = Path(__file__).resolve().parents[2]


def _strip_separator(argv: Sequence[str]) -> list[str]:
    items = list(argv)
    return items[1:] if items[:1] == ["--"] else items


def _run_python_module(module: str, argv: Sequence[str]) -> int:
    old_argv = sys.argv[:]
    sys.argv = [f"marichatmen {module}", *argv]
    try:
        imported = __import__(module, fromlist=["main"])
        result = imported.main()
        return int(result or 0)
    finally:
        sys.argv = old_argv


def _env_from_args(args: argparse.Namespace) -> dict[str, str]:
    config = load_config(args.config)
    cli_values = parse_key_value(getattr(args, "set_values", None))
    for attr in ("artifact_root", "model_name", "run_slug"):
        value = getattr(args, attr, None)
        if value is not None:
            cli_values[attr] = value
    merged = merged_options(config=config, cli=cli_values)
    return {**os.environ, **options_to_env(merged)}


def _run_shell(script: str, args: argparse.Namespace) -> int:
    env = _env_from_args(args)
    command = ["bash", str(ROOT / script), *getattr(args, "passthrough", [])]
    if args.dry_run:
        env_delta = {
            key: value
            for key, value in sorted(env.items())
            if key.startswith("MCM_") and os.environ.get(key) != value
        }
        print(" ".join(command))
        for key, value in env_delta.items():
            print(f"{key}={value}")
        return 0
    return subprocess.call(command, env=env, cwd=ROOT)


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="", help="YAML/JSON config file.")
    parser.add_argument("--artifact-root", default=None, help="Sets MCM_ARTIFACT_ROOT.")
    parser.add_argument("--model-name", default=None, help="Sets MCM_MODEL_NAME.")
    parser.add_argument("--run-slug", default=None, help="Sets MCM_RUN_SLUG.")
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Set an MCM_* option. CLI values override env and config.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print command and env changes.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="marichatmen")
    sub = parser.add_subparsers(dest="command", required=True)

    chat = sub.add_parser("chat", help="Start the local chat CLI.")
    chat.add_argument("passthrough", nargs=argparse.REMAINDER)

    benchmark = sub.add_parser("benchmark", help="Run the MariChatmen benchmark.")
    benchmark.add_argument("passthrough", nargs=argparse.REMAINDER)

    build_data = sub.add_parser("build-data", help="Build public training artifacts.")
    _add_config_args(build_data)
    build_data.add_argument("passthrough", nargs=argparse.REMAINDER)

    train = sub.add_parser("train", help="Run training workflows.")
    train_sub = train.add_subparsers(dest="train_command", required=True)
    qwen = train_sub.add_parser("qwen-andaluh", help="Run the Qwen-Andaluh CPT/SFT/ORPO workflow.")
    _add_config_args(qwen)
    qwen.add_argument("passthrough", nargs=argparse.REMAINDER)
    persona = train_sub.add_parser("persona", help="Run the MariChatmen persona workflow.")
    _add_config_args(persona)
    persona.add_argument(
        "--allow-persona",
        action="store_true",
        help="Sets MCM_ALLOW_PERSONA_TRAINING=1.",
    )
    persona.add_argument("passthrough", nargs=argparse.REMAINDER)

    eval_parser = sub.add_parser("eval", help="Run evaluation workflows.")
    eval_sub = eval_parser.add_subparsers(dest="eval_command", required=True)
    qwen_gate = eval_sub.add_parser("qwen-gate", help="Apply Qwen-Andaluh quality gates.")
    qwen_gate.add_argument("--samples", dest="samples_file", default="")
    qwen_gate.add_argument("--mode", choices=["orpo", "release", "repair"], default=None)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["chat"]:
        return _run_python_module("marichatmen.serve.chat", _strip_separator(argv[1:]))
    if argv[:1] == ["benchmark"]:
        return _run_python_module("marichatmen.eval.benchmark", _strip_separator(argv[1:]))

    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        if args.command == "eval" and args.eval_command == "qwen-gate":
            args.passthrough = unknown
        elif hasattr(args, "passthrough"):
            args.passthrough = [*args.passthrough, *unknown]
        else:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    if args.command == "build-data":
        return _run_shell("scripts/build_data.sh", args)
    if args.command == "train" and args.train_command == "qwen-andaluh":
        return _run_shell("scripts/train_qwen_andaluh_local.sh", args)
    if args.command == "train" and args.train_command == "persona":
        if args.allow_persona:
            args.set_values = [*args.set_values, "allow_persona_training=1"]
        return _run_shell("scripts/train_marichatmen_local.sh", args)
    if args.command == "eval" and args.eval_command == "qwen-gate":
        passthrough = list(getattr(args, "passthrough", []))
        if args.samples_file:
            passthrough = ["--samples_file", args.samples_file, *passthrough]
        if args.mode:
            passthrough = [*passthrough, "--mode", args.mode]
        return _run_python_module("marichatmen.eval.qwen_gate", passthrough)
    parser.error("Unhandled command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
