#!/usr/bin/env bash
set -euo pipefail

# The workstation may expose global pytest plugins outside this virtualenv.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

uv run python -m compileall src scripts tests sitecustomize.py
uv run python -m pytest tests
