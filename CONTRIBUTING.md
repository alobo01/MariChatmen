# Contributing

Use Python 3.12 with `uv`:

```bash
uv sync --python 3.12 --extra train --extra eval --extra dev
bash scripts/run_tests.sh
```

Keep implementation logic in `src/marichatmen/`; scripts should stay thin.
Do not add generated datasets, checkpoints, logs, or reports to git.

When changing data contracts, update `marichatmen.schemas` and add focused
tests. When changing adapter loading, preserve tokenizer compatibility checks.
