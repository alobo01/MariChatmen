#!/usr/bin/env python
"""Validate the Python environment for MariChatmen."""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys

REQUIRED = [
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "peft",
    "trl",
    "bitsandbytes",
    "sacrebleu",
    "rapidfuzz",
    "andaluh",
]


def version_for(module_name: str) -> str:
    package = {"PIL": "pillow"}.get(module_name, module_name)
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        module = importlib.import_module(module_name)
        return getattr(module, "__version__", "unknown")


def main() -> None:
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    if not ((3, 11) <= sys.version_info < (3, 13)):
        raise SystemExit("Python must be >=3.11,<3.13. Run: uv sync --python 3.12")

    failures = []
    for module_name in REQUIRED:
        try:
            importlib.import_module(module_name)
            print(f"{module_name}: {version_for(module_name)}")
        except Exception as exc:
            failures.append((module_name, exc))
            print(f"{module_name}: FAILED ({exc})")

    if failures:
        names = ", ".join(name for name, _ in failures)
        raise SystemExit(f"Missing or broken dependencies: {names}")
    print("Environment OK")


if __name__ == "__main__":
    main()
