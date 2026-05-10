"""Build a small markdown/SVG report from Qwen-Andaluh budget sweep metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from marichatmen.constants import ARTIFACT_ROOT


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key) or 0)
    except ValueError:
        return 0.0


def _line_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 960, 440
    left, top, right, bottom = 84, 52, 36, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [_float(row, "raw_train_tokens") for row in rows]
    max_x = max(xs or [1.0])
    max_y = 100.0
    colours = {"cpt": "#2563eb", "sft": "#16a34a", "orpo": "#dc2626"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="{left}" y="32" font-size="23" font-weight="700">Qwen-Andaluh Token Budget Sweep</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222"/>',
    ]
    for tick in range(0, 101, 20):
        y = top + plot_h - tick / max_y * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="32" y="{y + 4:.1f}" font-size="12">{tick}</text>')
    for stage, colour in colours.items():
        stage_rows = sorted((row for row in rows if row.get("stage") == stage), key=lambda row: _float(row, "raw_train_tokens"))
        if not stage_rows:
            continue
        points = []
        for row in stage_rows:
            x = left + _float(row, "raw_train_tokens") / max_x * plot_w
            y = top + plot_h - _float(row, "mari_aas_mean") / max_y * plot_h
            points.append((x, y))
        point_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        parts.append(f'<polyline fill="none" stroke="{colour}" stroke-width="3" points="{point_str}"/>')
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colour}"/>')
    legend_x = left + 12
    for idx, (stage, colour) in enumerate(colours.items()):
        y = height - 34 + idx * 0
        x = legend_x + idx * 120
        parts.append(f'<rect x="{x}" y="{y - 10}" width="14" height="14" fill="{colour}"/>')
        parts.append(f'<text x="{x + 20}" y="{y + 2}" font-size="13">{stage.upper()}</text>')
    parts.append(f'<text x="{left}" y="{height - 10}" font-size="12">x-axis: raw stage tokens in the subset; y-axis: mean MARI-AAS on fixed probes.</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _best_by_stage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best = {}
    for row in rows:
        stage = row.get("stage", "unknown")
        current = best.get(stage)
        if current is None or _float(row, "mari_aas_mean") > _float(current, "mari_aas_mean"):
            best[stage] = row
    return [best[key] for key in sorted(best)]


def run(args: argparse.Namespace) -> None:
    rows = _read_rows(Path(args.metrics_csv))
    out_md = Path(args.output_md)
    out_svg = Path(args.output_svg)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_md.write_text("# Qwen-Andaluh Budget Sweep\n\nNo rows yet.\n", encoding="utf-8")
        return
    _line_svg(out_svg, rows)
    lines = [
        "# Qwen-Andaluh Budget Sweep",
        "",
        "This report tracks how much stage data was needed before fixed probes improved. "
        "It is intentionally accent-only: no MariChatmen persona metrics are used here.",
        "",
        f"![Budget sweep]({out_svg.as_posix()})",
        "",
        "## Best Row Per Stage",
        "",
        "| Stage | Raw tokens | Planned seen tokens | Rows | MARI-AAS | Clear Andaluh rate | Spanish leak |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in _best_by_stage(rows):
        lines.append(
            f"| {row.get('stage', '')} | {_float(row, 'raw_train_tokens'):.0f} | "
            f"{_float(row, 'planned_seen_tokens'):.0f} | {_float(row, 'source_rows'):.0f} | "
            f"{_float(row, 'mari_aas_mean'):.2f} | {_float(row, 'clear_andaluh_rate'):.2%} | "
            f"{_float(row, 'spanish_leak_mean'):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- CPT should reduce Andaluh perplexity and raise the accent score before SFT.",
            "- SFT should improve instruction following while keeping Spanish leak low.",
            "- ORPO should mainly reduce Spanish leakage and weak-Andaluh outputs; if it lowers coherence, the rejected set is too sharp or too narrow.",
            "",
        ]
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_md} and {out_svg}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics_csv",
        default=str(ARTIFACT_ROOT / "reports/qwen_andaluh_budget_sweep.csv"),
    )
    parser.add_argument(
        "--output_md",
        default=str(ARTIFACT_ROOT / "reports/stage_reports/qwen_andaluh_budget_sweep.md"),
    )
    parser.add_argument(
        "--output_svg",
        default=str(ARTIFACT_ROOT / "reports/plots/qwen_andaluh_budget_sweep.svg"),
    )
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
