"""Minimal report generator from evaluation CSV and sample files."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_csv", default="reports/eval_history.csv")
    parser.add_argument("--out_file", default="reports/final_report.md")
    args = parser.parse_args()
    csv_path = Path(args.eval_csv)
    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        out.write_text("# MariChatmen Final Report\n\nNo evaluation rows yet.\n", encoding="utf-8")
        return
    out.write_text(
        "# MariChatmen Final Report\n\n"
        "Evaluation history is available in `reports/eval_history.csv`.\n\n"
        "Add qualitative sample analysis after local smoke and 2B runs complete.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
