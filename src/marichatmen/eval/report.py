"""Build a MariChatmen training report from run artefacts.

The output favors clear figures, short explanations, qualitative examples, and
explicit caveats about what the current run does and does not prove.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STAGE_LABELS = {
    "cpt": "CPT / AAP",
    "sft": "SFT",
    "orpo": "ORPO",
    "cpt_probe": "CPT probes",
}

MODEL_LABELS = {
    "08b": "0.8B",
    "2b": "2B",
    "4b": "4B",
}

SOURCES = [
    (
        "Qwen3.5 model family",
        "https://huggingface.co/Qwen/Qwen3.5-4B-Base",
    ),
    (
        "TRL SFTTrainer, ORPOTrainer, GRPO/RLOO family",
        "https://huggingface.co/docs/trl",
    ),
    (
        "LoRA",
        "https://arxiv.org/abs/2106.09685",
    ),
    (
        "QLoRA",
        "https://arxiv.org/abs/2305.14314",
    ),
    (
        "ORPO",
        "https://arxiv.org/abs/2403.07691",
    ),
    (
        "AndaluGeeks EPA",
        "https://andaluh.es/an/epa/",
    ),
    (
        "andaluh-py",
        "https://github.com/andalugeeks/andaluh-py",
    ),
    (
        "Spanish Wikipedia dump used as CPT source",
        "https://dumps.wikimedia.org/eswiki/20260501/",
    ),
    (
        "VillanovaAI/villanova-sft-2603",
        "https://huggingface.co/datasets/VillanovaAI/villanova-sft-2603",
    ),
    (
        "Iker/OpenHermes-2.5-Spanish",
        "https://huggingface.co/datasets/Iker/OpenHermes-2.5-Spanish",
    ),
    (
        "CohereLabs/aya_dataset",
        "https://huggingface.co/datasets/CohereLabs/aya_dataset",
    ),
    (
        "projecte-aina/MentorES",
        "https://huggingface.co/datasets/projecte-aina/MentorES",
    ),
]


@dataclass(frozen=True)
class RunPart:
    model: str
    branch: str
    stage: str


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _parse_run_id(run_tag: str, run_id: str) -> RunPart | None:
    prefix = f"{run_tag}_"
    if not run_id.startswith(prefix):
        return None
    tail = run_id[len(prefix) :]
    match = re.match(r"(?P<model>08b|2b|4b)_(?P<branch>qwen|mari)_(?P<stage>cpt|sft|orpo)$", tail)
    if not match:
        if tail.endswith("_qwen_cpt") and "cpt_probe" in run_id:
            return None
        return None
    return RunPart(**match.groupdict())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _load_metrics(path: Path, run_tag: str) -> list[dict[str, Any]]:
    return [row for row in _load_jsonl(path) if run_tag in str(row.get("run_id", ""))]


def _load_status(path: Path) -> list[tuple[str, str, str]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                rows.append((parts[0], parts[1], parts[2]))
    return rows


def _load_gpu(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            used = _safe_float(str(row.get("memory.used [MiB]", "")).replace(" MiB", ""))
            total = _safe_float(str(row.get("memory.total [MiB]", "")).replace(" MiB", ""))
            util = _safe_float(str(row.get("utilization.gpu [%]", "")).replace(" %", ""))
            power = _safe_float(str(row.get("power.draw [W]", "")).replace(" W", ""))
            if used is None or total is None or util is None:
                continue
            rows.append(
                {
                    "step": float(index),
                    "memory_gb": used / 1024.0,
                    "memory_total_gb": total / 1024.0,
                    "util": util,
                    "power_w": power or 0.0,
                }
            )
    return rows


def _parse_tokenizer_report(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"summary": [], "tokens": []}
    if not path.exists():
        return result
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("- "):
            result["summary"].append(line[2:])
    in_table = False
    for line in text.splitlines():
        if line.startswith("| Token |"):
            in_table = True
            continue
        if in_table and (line.startswith("| ---") or not line.startswith("|")):
            if not line.startswith("| ---"):
                in_table = False
            continue
        if in_table:
            cells = [cell.strip().strip("`") for cell in line.strip("|").split("|")]
            if len(cells) >= 4:
                token, freq, pieces, saving = cells[:4]
                result["tokens"].append(
                    {
                        "token": token,
                        "freq": _safe_float(freq) or 0.0,
                        "pieces": _safe_float(pieces) or 0.0,
                        "saving": _safe_float(saving) or 0.0,
                    }
                )
    examples = []
    blocks = re.findall(r"```text\n(.*?)```", text, flags=re.S)
    for block in blocks[:3]:
        examples.append(block.strip())
    result["examples"] = examples
    return result


def _series_for_loss(metrics: list[dict[str, Any]], run_tag: str) -> dict[str, list[tuple[float, float]]]:
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in metrics:
        run_id = str(row.get("run_id", ""))
        part = _parse_run_id(run_tag, run_id)
        step = _safe_float(row.get("global_step"))
        loss = _safe_float(row.get("loss"))
        if not part or step is None or loss is None:
            continue
        label = f"{MODEL_LABELS.get(part.model, part.model)} {part.branch.upper()} {STAGE_LABELS.get(part.stage, part.stage)}"
        series[label].append((step, loss))
    return dict(series)


def _series_for_eval(metrics: list[dict[str, Any]], run_tag: str) -> dict[str, list[tuple[float, float]]]:
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in metrics:
        run_id = str(row.get("run_id", ""))
        part = _parse_run_id(run_tag, run_id)
        step = _safe_float(row.get("global_step"))
        loss = _safe_float(row.get("eval_loss"))
        if not part or step is None or loss is None:
            continue
        label = f"{MODEL_LABELS.get(part.model, part.model)} {part.branch.upper()} {STAGE_LABELS.get(part.stage, part.stage)}"
        series[label].append((step, loss))
    return dict(series)


def _series_for_cpt_ppl(metrics: list[dict[str, Any]], run_tag: str) -> dict[str, list[tuple[float, float]]]:
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in metrics:
        if row.get("stage") != "cpt_probe":
            continue
        run_id = str(row.get("run_id", ""))
        part = _parse_run_id(run_tag, run_id.replace("_cpt_probe", "_cpt"))
        step = _safe_float(row.get("global_step"))
        if not part or step is None:
            continue
        for key, label in [
            ("spanish_perplexity", "Spanish PPL"),
            ("andaluh_perplexity", "Andaluh PPL"),
        ]:
            value = _safe_float(row.get(key))
            if value is not None:
                series[f"{MODEL_LABELS.get(part.model, part.model)} {label}"].append((step, value))
    return dict(series)


def _series_for_orpo(metrics: list[dict[str, Any]], run_tag: str) -> dict[str, list[tuple[float, float]]]:
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in metrics:
        run_id = str(row.get("run_id", ""))
        part = _parse_run_id(run_tag, run_id)
        if not part or part.stage != "orpo":
            continue
        step = _safe_float(row.get("global_step"))
        if step is None:
            continue
        margin = _safe_float(row.get("eval_rewards/margins"))
        accuracy = _safe_float(row.get("eval_rewards/accuracies"))
        if margin is not None:
            series[f"{MODEL_LABELS.get(part.model, part.model)} {part.branch.upper()} margin"].append((step, margin))
        if accuracy is not None:
            series[f"{MODEL_LABELS.get(part.model, part.model)} {part.branch.upper()} accuracy"].append((step, accuracy))
    return dict(series)


def _latest_table(metrics: list[dict[str, Any]], run_tag: str) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in metrics:
        run_id = str(row.get("run_id", ""))
        part = _parse_run_id(run_tag, run_id)
        if not part:
            continue
        key = (part.model, part.branch, part.stage)
        old_step = _safe_float(latest.get(key, {}).get("global_step")) or -1.0
        new_step = _safe_float(row.get("global_step")) or -1.0
        if new_step >= old_step:
            latest[key] = row
    table = []
    for (model, branch, stage), row in sorted(latest.items()):
        table.append(
            {
                "model": MODEL_LABELS.get(model, model),
                "branch": branch.upper(),
                "stage": STAGE_LABELS.get(stage, stage),
                "step": row.get("global_step", ""),
                "loss": row.get("loss", ""),
                "eval_loss": row.get("eval_loss", ""),
                "train_loss": row.get("train_loss", ""),
                "orpo_margin": row.get("eval_rewards/margins", ""),
                "orpo_accuracy": row.get("eval_rewards/accuracies", ""),
            }
        )
    return table


def _scale_points(points: list[tuple[float, float]], width: int, height: int, pad: int):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_x == max_x:
        max_x += 1
    if min_y == max_y:
        max_y += 1
    def scale(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        sx = pad + (x - min_x) / (max_x - min_x) * (width - 2 * pad)
        sy = height - pad - (y - min_y) / (max_y - min_y) * (height - 2 * pad)
        return sx, sy
    return scale, (min_x, max_x, min_y, max_y)


def _line_svg(
    path: Path,
    title: str,
    series: dict[str, list[tuple[float, float]]],
    y_label: str,
    *,
    note: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height, pad = 920, 480, 62
    colours = [
        "#006847",
        "#c0182f",
        "#1f77b4",
        "#d68c00",
        "#6f42c1",
        "#111111",
        "#2ca02c",
        "#8c564b",
    ]
    non_empty = {k: sorted(v) for k, v in series.items() if v}
    if not non_empty:
        path.write_text(
            f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="#fffaf0"/>
<text x="40" y="55" font-size="28" font-family="Georgia">{html.escape(title)}</text>
<text x="40" y="105" font-size="18" font-family="Arial" fill="#555">No data yet. This panel will fill when the run reaches that stage.</text>
</svg>""",
            encoding="utf-8",
        )
        return
    all_points = [point for points in non_empty.values() for point in points]
    scale, bounds = _scale_points(all_points, width, height, pad)
    min_x, max_x, min_y, max_y = bounds
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffaf0"/>',
        f'<text x="40" y="38" font-size="25" font-family="Georgia" fill="#111">{html.escape(title)}</text>',
        f'<text x="40" y="65" font-size="13" font-family="Arial" fill="#555">{html.escape(note)}</text>' if note else "",
        f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#333" stroke-width="1.2"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#333" stroke-width="1.2"/>',
        f'<text x="{width/2-40}" y="{height-16}" font-size="14" font-family="Arial">global step</text>',
        f'<text transform="translate(18 {height/2+60}) rotate(-90)" font-size="14" font-family="Arial">{html.escape(y_label)}</text>',
    ]
    for i in range(5):
        x = pad + i * (width - 2 * pad) / 4
        step = min_x + i * (max_x - min_x) / 4
        lines.append(f'<line x1="{x:.1f}" y1="{height-pad}" x2="{x:.1f}" y2="{height-pad+5}" stroke="#333"/>')
        lines.append(f'<text x="{x-16:.1f}" y="{height-pad+22}" font-size="11" font-family="Arial">{step:.0f}</text>')
        y = height - pad - i * (height - 2 * pad) / 4
        value = min_y + i * (max_y - min_y) / 4
        lines.append(f'<line x1="{pad-5}" y1="{y:.1f}" x2="{pad}" y2="{y:.1f}" stroke="#333"/>')
        lines.append(f'<text x="8" y="{y+4:.1f}" font-size="11" font-family="Arial">{value:.2g}</text>')
        if i:
            lines.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{width-pad}" y2="{y:.1f}" stroke="#eadfca" stroke-width="1"/>')
    for index, (label, points) in enumerate(non_empty.items()):
        colour = colours[index % len(colours)]
        scaled = [scale(point) for point in points]
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in scaled)
        lines.append(f'<polyline fill="none" stroke="{colour}" stroke-width="2.4" points="{d}"/>')
        for x, y in scaled[-2:]:
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{colour}"/>')
    legend_x, legend_y = width - 265, 82
    lines.append(f'<rect x="{legend_x-12}" y="{legend_y-24}" width="250" height="{24+22*len(non_empty)}" rx="6" fill="#ffffff" stroke="#eadfca"/>')
    for index, label in enumerate(non_empty):
        colour = colours[index % len(colours)]
        y = legend_y + index * 22
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+24}" y2="{y}" stroke="{colour}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x+32}" y="{y+4}" font-size="12" font-family="Arial">{html.escape(label)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(line for line in lines if line), encoding="utf-8")


def _bar_svg(path: Path, title: str, rows: list[dict[str, Any]], key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height, pad = 900, 430, 66
    values = [(row["token"], float(row.get(key, 0.0))) for row in rows[:14] if row.get(key, 0.0)]
    if not values:
        _line_svg(path, title, {}, key)
        return
    max_value = max(value for _, value in values) or 1.0
    bar_h = (height - 2 * pad) / len(values) * 0.7
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffaf0"/>',
        f'<text x="40" y="42" font-size="25" font-family="Georgia">{html.escape(title)}</text>',
    ]
    for index, (label, value) in enumerate(values):
        y = pad + index * (height - 2 * pad) / len(values)
        w = (width - 300) * value / max_value
        lines.append(f'<text x="40" y="{y+bar_h*0.72:.1f}" font-size="14" font-family="monospace">{html.escape(label)}</text>')
        lines.append(f'<rect x="235" y="{y:.1f}" width="{w:.1f}" height="{bar_h:.1f}" fill="#006847"/>')
        lines.append(f'<text x="{245+w:.1f}" y="{y+bar_h*0.72:.1f}" font-size="13" font-family="Arial">{value:.0f}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _gpu_svg(path: Path, gpu_rows: list[dict[str, float]]) -> None:
    util = [(row["step"], row["util"]) for row in gpu_rows]
    mem = [(row["step"], row["memory_gb"]) for row in gpu_rows]
    _line_svg(
        path,
        "GPU0 during the run",
        {"utilisation %": util, "VRAM GB": mem},
        "utilisation / GB",
        note="A deliberately boring plot is good news: stable VRAM, busy GPU, no fireworks.",
    )


def _write_html(md_path: Path, html_path: Path) -> None:
    body = md_path.read_text(encoding="utf-8")
    body = _markdown_to_html(body)
    css = """
body{font-family:Georgia,'Times New Roman',serif;line-height:1.58;color:#171717;background:#fffaf0;margin:0}
main{max-width:860px;margin:0 auto;padding:42px 26px 80px}
h1{font-size:42px;line-height:1.05;margin-bottom:10px}
h2{font-size:28px;margin-top:46px;border-top:1px solid #eadfca;padding-top:22px}
h3{font-size:20px;margin-top:30px}
p,li{font-size:17px}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
pre{background:#111;color:#f8f8f2;padding:16px;border-radius:8px;overflow:auto}
blockquote{border-left:4px solid #006847;margin-left:0;padding:8px 18px;background:#f5efd7}
img{max-width:100%;display:block;margin:20px auto;border:1px solid #eadfca;border-radius:8px;background:white}
table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}
td,th{border:1px solid #eadfca;padding:7px 8px;text-align:left}
th{background:#f1e7c9}
.widget{border:2px dashed #006847;background:#eef8f1;border-radius:10px;padding:14px 16px;margin:18px 0}
.caption{font-size:14px;color:#555;text-align:center;margin-top:-10px}
"""
    html_doc = f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body><main>{body}</main></body></html>"
    html_path.write_text(html_doc, encoding="utf-8")


def _markdown_to_html(text: str) -> str:
    # A small markdown renderer for this controlled report. It supports the
    # constructs we emit below and keeps the repo dependency-light.
    lines = text.splitlines()
    out: list[str] = []
    in_code = False
    in_ul = False
    in_table = False
    table_rows: list[str] = []

    def flush_ul() -> None:
        nonlocal in_ul
        if in_ul:
            out.append("</ul>")
            in_ul = False

    def flush_table() -> None:
        nonlocal in_table, table_rows
        if not in_table:
            return
        out.append("<table>")
        for index, row in enumerate(table_rows):
            cells = [cell.strip() for cell in row.strip("|").split("|")]
            if all(set(cell) <= {"-", ":"} for cell in cells):
                continue
            tag = "th" if index == 0 else "td"
            out.append("<tr>" + "".join(f"<{tag}>{html.escape(cell)}</{tag}>" for cell in cells) + "</tr>")
        out.append("</table>")
        in_table = False
        table_rows = []

    for line in lines:
        if line.startswith("```"):
            flush_ul()
            flush_table()
            if in_code:
                out.append("</code></pre>")
                in_code = False
            else:
                out.append("<pre><code>")
                in_code = True
            continue
        if in_code:
            out.append(html.escape(line))
            continue
        if line.startswith("|") and line.endswith("|"):
            flush_ul()
            in_table = True
            table_rows.append(line)
            continue
        flush_table()
        if not line.strip():
            flush_ul()
            out.append("")
            continue
        if line.startswith("# "):
            flush_ul()
            out.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            flush_ul()
            out.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            flush_ul()
            out.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("> "):
            flush_ul()
            out.append(f"<blockquote>{_inline_html(line[2:])}</blockquote>")
        elif line.startswith("- "):
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{_inline_html(line[2:])}</li>")
        elif line.startswith("![") and "](" in line and line.endswith(")"):
            alt = line[2:].split("](", 1)[0]
            src = line.split("](", 1)[1][:-1]
            out.append(f'<img src="{html.escape(src)}" alt="{html.escape(alt)}">')
        elif line.startswith("<div class=\"widget\">"):
            out.append(line)
        elif line.startswith("</div>"):
            out.append(line)
        else:
            out.append(f"<p>{_inline_html(line)}</p>")
    flush_ul()
    flush_table()
    return "\n".join(out)


def _inline_html(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    return escaped


def _write_pdf(html_path: Path, pdf_path: Path) -> bool:
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
    if chrome:
        command = [
            chrome,
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            f"--print-to-pdf={pdf_path}",
            str(html_path.resolve()),
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError:
            pass
    pandoc = shutil.which("pandoc")
    if pandoc:
        try:
            subprocess.run([pandoc, str(html_path), "-o", str(pdf_path)], check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    return False


def _copy_samples(samples_dir: Path, run_tag: str) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for path in sorted(samples_dir.glob(f"*{run_tag}*examples.jsonl")):
        for row in _load_jsonl(path)[:2]:
            prompt = str(row.get("prompt", ""))
            answer = (
                row.get("orpo_answer")
                or row.get("sft_answer")
                or row.get("cpt_answer")
                or row.get("original_answer")
                or ""
            )
            if prompt and answer:
                examples.append({"file": path.name, "prompt": prompt, "answer": str(answer)})
    return examples[:6]


def _table_md(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No stage summary rows yet."
    headers = ["model", "branch", "stage", "step", "loss", "eval_loss", "train_loss", "orpo_margin", "orpo_accuracy"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        cells = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                value = f"{value:.4g}"
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _status_summary(status_rows: list[tuple[str, str, str]]) -> str:
    if not status_rows:
        return "No status rows found."
    last = status_rows[-1]
    events = ", ".join(f"`{event}`" for _, event, _ in status_rows[-6:])
    return f"Latest event: `{last[1]}` ({last[2]}). Recent event trail: {events}."


def _write_markdown(
    *,
    out_path: Path,
    run_tag: str,
    status_rows: list[tuple[str, str, str]],
    metrics: list[dict[str, Any]],
    tokenizer: dict[str, Any],
    figures: dict[str, Path],
    sample_examples: list[dict[str, str]],
    partial: bool,
) -> None:
    latest = _latest_table(metrics, run_tag)
    token_rows = tokenizer.get("tokens", [])[:10]
    token_text = ", ".join(f"`{row['token']}`" for row in token_rows[:8]) or "not available yet"
    status_note = "LIVE PARTIAL DRAFT" if partial else "FULL RUN DRAFT"
    lines = [
        "# MariChatmen: training Qwen to speak Andalûh, then giving it a Sevillian soul",
        "",
        f"**Status:** {status_note}. Run tag: `{run_tag}`.",
        "",
        "> The first rule of dialectal post-training is cruel but useful: make the model answer the question before you let it wear the hat.",
        "",
        "I am building two models, not one confused costume:",
        "",
        "- **Qwen-Andaluh**: a neutral assistant that understands Spanish or Andalûh and answers in Andalûh.",
        "- **MariChatmen**: Qwen-Andaluh plus a fictional Sevillian persona, Expo '92 energy, Feria references, and just enough gazpacho-based confidence to be charming rather than legally actionable.",
        "",
        "This draft is generated directly from the training artefacts: JSONL metrics, GPU logs, tokenizer audits, checkpoints, and saved probe generations. If a plot is empty, that stage has not run yet; no decorative pretending.",
        "",
        "## The current run at a glance",
        "",
        _status_summary(status_rows),
        "",
        _table_md(latest),
        "",
        "![Training loss curves](figures/loss_curves.svg)",
        "",
        "The first thing I want from a long run is not magic. I want the loss curve to look boring in the right direction. The model should become less surprised by Andalûh text before I ask it to be funny, warm, or culturally specific.",
        "",
        "![Evaluation loss curves](figures/eval_loss_curves.svg)",
        "",
        "Evaluation loss is the seatbelt. It is less entertaining than samples, but it catches the classic small-model tragedy: the model learns the training set and forgets how to be useful anywhere else.",
        "",
        "![CPT Spanish vs Andaluh perplexity](figures/cpt_probe_perplexity.svg)",
        "",
        "The CPT probe is the plot I care about most for Qwen-Andaluh. Spanish perplexity should not explode, while Andalûh perplexity should fall. If Andalûh gets cheaper for the model without torching Spanish competence, the foundation is doing its job.",
        "",
        "## Tokeniser expansion: teaching Qwen that `ç` is not a keyboard accident",
        "",
        "The expanded tokenizer is deliberately Qwen-compatible. I am not retraining the tokenizer from scratch because that would scramble the meaning of the pretrained embedding table. Instead, I add common Andalûh forms and initialise each new embedding from its original Qwen subpieces.",
        "",
        f"The most useful added tokens in the audit include {token_text}. These are not cute decorations; they are places where the old tokenizer repeatedly broke one Andalûh form into too many pieces.",
        "",
        "![Most useful added Andaluh tokens](figures/tokenizer_tokens.svg)",
        "",
    ]
    for summary in tokenizer.get("summary", [])[:5]:
        lines.append(f"- {summary}")
    lines.extend(
        [
            "",
            "Example transformations from the audit:",
            "",
        ]
    )
    for example in tokenizer.get("examples", [])[:2]:
        lines.extend(["```text", example, "```", ""])
    lines.extend(
        [
            "<div class=\"widget\">",
            "**Interactive widget placeholder: tokenizer playground.** In the web post, put a small box here where the reader types a Spanish sentence and sees Spanish tokens, Andalûh tokens, ATIR, and the exact added tokens that reduce fragmentation.",
            "</div>",
            "",
            "## Why the pipeline is staged",
            "",
            "The earlier smoke run taught the wrong lesson on purpose: the machinery worked, but the model learned surface personality before robust behaviour. That is how you get a tiny model muttering about Feria, SFDK and gazpacho while not actually explaining overfitting. Funny once. Not a model card.",
            "",
            "So the current pipeline is staged:",
            "",
            "```text",
            "Qwen base",
            "-> expanded Andalûh tokenizer",
            "-> CPT / AAP on Andalûh text",
            "-> neutral Andalûh SFT",
            "-> neutral Andalûh ORPO",
            "= Qwen-Andaluh",
            "",
            "Qwen-Andaluh",
            "-> MariChatmen persona SFT",
            "-> MariChatmen persona ORPO",
            "= MariChatmen v1",
            "```",
            "",
            "CPT teaches the distribution. SFT teaches instruction following. ORPO teaches preference: good Andalûh over standard Spanish leakage, weak Andalûh, or over-transcribed soup. Persona comes only after that, because style should sit on top of competence, not replace it.",
            "",
            "![ORPO preference metrics](figures/orpo_metrics.svg)",
            "",
            "## GPU behaviour",
            "",
            "For quality runs, keep model caches and generated artefacts outside the source checkout, record GPU utilisation, and run larger models alone when memory pressure would make comparisons noisy.",
            "",
            "![GPU history](figures/gpu_history.svg)",
            "",
            "<div class=\"widget\">",
            "**Interactive widget placeholder: run explorer.** Let the reader choose model size and stage, then update loss curves, eval curves, GPU utilisation, and sample answers from the same dropdown.",
            "</div>",
            "",
            "## Qualitative samples",
            "",
        ]
    )
    if sample_examples:
        for item in sample_examples:
            lines.extend(
                [
                    f"### {item['prompt']}",
                    "",
                    f"Source file: `{item['file']}`",
                    "",
                    "```text",
                    item["answer"][:1800],
                    "```",
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "No saved stage samples are available for this run yet. The launcher writes five Qwen-Andaluh probes and five MariChatmen probes after each model pipeline finishes.",
                "",
            ]
        )
    lines.extend(
        [
            "<div class=\"widget\">",
            "**Interactive widget placeholder: before/after carousel.** Show the same prompt across base Qwen, CPT, SFT, ORPO, and MariChatmen. Add toggles for `Spanish leak`, `coherent answer`, `persona`, and `keyword soup`.",
            "</div>",
            "",
            "## What would make this a win?",
            "",
            "For Qwen-Andaluh, I want the model to answer in Andalûh without losing ordinary assistant behaviour. The acceptance gates are practical rather than ceremonial: high MARI-AAS, Spanish leak below 7%, semantic drift under control, and manual samples that answer the question instead of merely looking Andalusian.",
            "",
            "For MariChatmen, the bar is higher. She should sound Sevillian and fictional, but she should not become a slot machine for `miarma`, `Feria`, `Cruzcampo`, and `gazpacho`. The good version has rhythm and restraint. The bad version is a fridge magnet with a GPU.",
            "",
            "## Sources and data provenance",
            "",
            "The CPT and SFT data is transformed Spanish data. The assistant side is converted into Andalûh using `andaluh-py` with EPA-style output, favouring `ç` for the Sevillian target, plus light informal reductions. Fragile spans such as code, URLs, model IDs and package names are protected before transliteration.",
            "",
            "Datasets and references used in the project:",
            "",
        ]
    )
    for label, url in SOURCES:
        lines.append(f"- {label}: {url}")
    lines.extend(
        [
            "",
            "## Closing intuition",
            "",
            "The useful mental model is not translation. It is distribution shaping. First make Andalûh cheap for the model. Then make instruction following reliable in that distribution. Then teach preference. Only after that should MariChatmen walk into the caseta and start telling you that Málaga beats Ibiza with a straight face.",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def build_report(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"
    metrics = _load_metrics(Path(args.metrics_jsonl), args.run_tag)
    status_rows = _load_status(Path(args.status_tsv))
    gpu_rows = _load_gpu(Path(args.gpu_csv))
    tokenizer = _parse_tokenizer_report(Path(args.tokenizer_md))
    samples = _copy_samples(Path(args.samples_dir), args.run_tag)

    figures = {
        "loss": figures_dir / "loss_curves.svg",
        "eval_loss": figures_dir / "eval_loss_curves.svg",
        "ppl": figures_dir / "cpt_probe_perplexity.svg",
        "orpo": figures_dir / "orpo_metrics.svg",
        "gpu": figures_dir / "gpu_history.svg",
        "tokens": figures_dir / "tokenizer_tokens.svg",
    }
    _line_svg(figures["loss"], "Training loss", _series_for_loss(metrics, args.run_tag), "loss")
    _line_svg(figures["eval_loss"], "Evaluation loss", _series_for_eval(metrics, args.run_tag), "eval loss")
    _line_svg(figures["ppl"], "CPT probe perplexity", _series_for_cpt_ppl(metrics, args.run_tag), "perplexity")
    _line_svg(figures["orpo"], "ORPO margin and accuracy", _series_for_orpo(metrics, args.run_tag), "margin / accuracy")
    _gpu_svg(figures["gpu"], gpu_rows)
    _bar_svg(figures["tokens"], "Most useful added tokens", tokenizer.get("tokens", []), "saving")

    partial = not any(event == "run_done" for _, event, _ in status_rows)
    md_path = out_dir / "marichatmen_lesswrong_draft.md"
    html_path = out_dir / "marichatmen_lesswrong_draft.html"
    pdf_path = out_dir / "marichatmen_lesswrong_draft.pdf"
    _write_markdown(
        out_path=md_path,
        run_tag=args.run_tag,
        status_rows=status_rows,
        metrics=metrics,
        tokenizer=tokenizer,
        figures=figures,
        sample_examples=samples,
        partial=partial,
    )
    _write_html(md_path, html_path)
    pdf_ok = _write_pdf(html_path, pdf_path) if args.pdf else False
    print(f"Wrote {md_path}")
    print(f"Wrote {html_path}")
    if args.pdf:
        if pdf_ok:
            print(f"Wrote {pdf_path}")
        else:
            print("PDF generation failed; HTML and Markdown were written.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_tag", required=True)
    parser.add_argument("--metrics_jsonl", default=str(Path(".artifacts/reports/training_runs.jsonl")))
    parser.add_argument("--status_tsv", default="")
    parser.add_argument("--gpu_csv", default="")
    parser.add_argument(
        "--tokenizer_md",
        default=str(Path(".artifacts/reports/tokenizer/tokenizer_impact.md")),
    )
    parser.add_argument("--samples_dir", default=str(Path(".artifacts/reports/samples")))
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()
    if not args.out_dir:
        args.out_dir = str(Path(".artifacts/reports/run_reports") / args.run_tag)
    if not args.status_tsv:
        args.status_tsv = str(Path(".artifacts/reports/longrun_logs") / args.run_tag / "status.tsv")
    if not args.gpu_csv:
        args.gpu_csv = str(Path(".artifacts/reports/gpu_history") / f"{args.run_tag}_gpu0.csv")
    return args


def main() -> None:
    build_report(parse_args())


if __name__ == "__main__":
    main()
