"""Build CPT rows from a Spanish Wikipedia XML dump or extracted text JSONL."""

from __future__ import annotations

import argparse
import bz2
import json
import random
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from marichatmen.constants import (
    ARTIFACT_ROOT,
    WIKIPEDIA_ESWIKI_20260501_ARTICLES,
    WIKIPEDIA_ESWIKI_20260501_URL,
    WIKIPEDIA_TEXT_LICENSE,
)
from marichatmen.data.transliterate_andaluh import to_andaluh
from marichatmen.io import iter_jsonl, write_jsonl

TEMPLATE_RE = re.compile(r"\{\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\}", re.DOTALL)
REF_RE = re.compile(r"<ref\b[^>/]*(?:/>|>.*?</ref>)", re.DOTALL | re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
FILE_LINK_RE = re.compile(
    r"\[\[\s*(?:Archivo|File|Imagen|Image|Media)\s*:[^\]]+\]\]",
    re.IGNORECASE,
)
LINK_WITH_LABEL_RE = re.compile(r"\[\[[^|\]]+\|([^\]]+)\]\]")
LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
URL_RE = re.compile(r"https?://\S+")
MULTISPACE_RE = re.compile(r"[ \t]+")
PIPE_MARKUP_RE = re.compile(
    r"\b(?:thumb|right|left|center|centre|miniatura|miniaturadeimagen)\s*\||"
    r"\b[\dx]{2,9}\s*px\b|\b[\dx]{2,9}\s*px\s*\||\bpx\s*\|",
    re.IGNORECASE,
)


def _strip_namespace(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def clean_wikitext(text: str) -> str:
    text = FILE_LINK_RE.sub(" ", text)
    text = REF_RE.sub(" ", text)
    previous = None
    while previous != text:
        previous = text
        text = TEMPLATE_RE.sub(" ", text)
    text = LINK_WITH_LABEL_RE.sub(r"\1", text)
    text = LINK_RE.sub(r"\1", text)
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", text)
    text = URL_RE.sub(" ", text)
    text = TAG_RE.sub(" ", text)
    text = text.replace("&nbsp;", " ")
    text = re.sub(r"'{2,5}", "", text)
    text = re.sub(r"^={2,}\s*(.*?)\s*={2,}$", r"\1", text, flags=re.MULTILINE)
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("*", "#", "|", "{|", "}", "!")):
            continue
        if line.lower().startswith(("archivo:", "file:", "imagen:", "categoría:", "category:")):
            continue
        if PIPE_MARKUP_RE.search(line) or line.count("|") >= 2:
            continue
        lines.append(MULTISPACE_RE.sub(" ", line))
    return "\n\n".join(lines).strip()


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def iter_wikipedia_xml(path: str | Path, *, max_pages: int = 0) -> Iterable[dict[str, Any]]:
    opener = bz2.open if str(path).endswith(".bz2") else open
    yielded = 0
    with opener(path, "rb") as handle:
        context = ET.iterparse(handle, events=("end",))
        for _, element in context:
            if _strip_namespace(element.tag) != "page":
                continue
            title = ""
            ns = ""
            redirect = False
            text = ""
            for child in element:
                tag = _strip_namespace(child.tag)
                if tag == "title":
                    title = child.text or ""
                elif tag == "ns":
                    ns = child.text or ""
                elif tag == "redirect":
                    redirect = True
                elif tag == "revision":
                    for revision_child in child:
                        if _strip_namespace(revision_child.tag) == "text":
                            text = revision_child.text or ""
                            break
            element.clear()
            if ns != "0" or redirect or not text:
                continue
            cleaned = clean_wikitext(text)
            if not cleaned:
                continue
            yield {"title": title, "text": cleaned}
            yielded += 1
            if max_pages and yielded >= max_pages:
                return


def iter_text_rows(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    if args.input_text_jsonl:
        for row in iter_jsonl(args.input_text_jsonl):
            text = row.get("text")
            if isinstance(text, str):
                yield {
                    "title": row.get("title", ""),
                    "text": text,
                    "metadata": row.get("metadata", {}),
                }
        return
    if not args.dump_file:
        raise ValueError("Pass --dump_file or --input_text_jsonl.")
    yield from iter_wikipedia_xml(args.dump_file, max_pages=args.max_pages)


def _make_row(
    *,
    text: str,
    title: str,
    split_view: str,
    transformed: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "text": text,
        "metadata": {
            "source_dataset": "Spanish Wikipedia eswiki dump",
            "source_url": WIKIPEDIA_ESWIKI_20260501_URL,
            "source_file": Path(args.dump_file).name if args.dump_file else args.input_text_jsonl,
            "source_article_title": title,
            "source_license": WIKIPEDIA_TEXT_LICENSE,
            "source_dump_file_url": WIKIPEDIA_ESWIKI_20260501_ARTICLES,
            "language": "spa",
            "split_view": split_view,
            "transformation": f"andaluh_epa_{args.variant}" if transformed else "original_spanish",
            "transliterator": "andalugeeks/andaluh-py" if transformed else None,
        },
    }


def build(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    spanish_probe: list[dict[str, Any]] = []
    andaluh_probe: list[dict[str, Any]] = []
    needed = args.n_train + args.n_valid

    for index, row in enumerate(iter_text_rows(args)):
        raw_text = row["text"].strip()
        words = _word_count(raw_text)
        if words < args.min_words or words > args.max_words:
            continue
        title = str(row.get("title") or "")
        andaluh_text = to_andaluh(
            raw_text,
            variant=args.variant,
            informal_strength=args.informal_strength,
            seed=args.seed + index,
        )
        if len(spanish_probe) < args.n_probe:
            spanish_probe.append(
                _make_row(
                    text=raw_text,
                    title=title,
                    split_view="spanish",
                    transformed=False,
                    args=args,
                )
            )
            andaluh_probe.append(
                _make_row(
                    text=andaluh_text,
                    title=title,
                    split_view="andaluh",
                    transformed=True,
                    args=args,
                )
            )
        use_andaluh = rng.random() < args.andaluh_ratio
        rows.append(
            _make_row(
                text=andaluh_text if use_andaluh else raw_text,
                title=title,
                split_view="andaluh" if use_andaluh else "spanish",
                transformed=use_andaluh,
                args=args,
            )
        )
        if len(rows) >= needed:
            break

    rng.shuffle(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = rows[: args.n_train]
    valid_rows = rows[args.n_train : args.n_train + args.n_valid]
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "valid.jsonl", valid_rows)
    write_jsonl(out_dir / "spanish_valid.jsonl", spanish_probe)
    write_jsonl(out_dir / "andaluh_valid.jsonl", andaluh_probe)
    manifest = {
        "source": "Spanish Wikipedia eswiki dump",
        "source_url": WIKIPEDIA_ESWIKI_20260501_URL,
        "source_dump_file_url": WIKIPEDIA_ESWIKI_20260501_ARTICLES,
        "source_license": WIKIPEDIA_TEXT_LICENSE,
        "dump_file": args.dump_file,
        "input_text_jsonl": args.input_text_jsonl,
        "n_train": len(train_rows),
        "n_valid": len(valid_rows),
        "n_probe": len(spanish_probe),
        "andaluh_ratio": args.andaluh_ratio,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "variant": args.variant,
        "informal_strength": args.informal_strength,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote Wikipedia CPT rows to {out_dir}: "
        f"train={len(train_rows)}, valid={len(valid_rows)}, probe={len(spanish_probe)}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_file", default="")
    parser.add_argument("--input_text_jsonl", default="")
    parser.add_argument(
        "--out_dir",
        default=str(ARTIFACT_ROOT / "data/processed/cpt_wikipedia_eswiki_20260501"),
    )
    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--n_valid", type=int, default=100)
    parser.add_argument("--n_probe", type=int, default=100)
    parser.add_argument("--max_pages", type=int, default=0)
    parser.add_argument("--min_words", type=int, default=200)
    parser.add_argument("--max_words", type=int, default=2500)
    parser.add_argument("--andaluh_ratio", type=float, default=0.9)
    parser.add_argument("--variant", default="sevillian_ce")
    parser.add_argument("--informal_strength", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=46)
    return parser.parse_args(argv)


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
