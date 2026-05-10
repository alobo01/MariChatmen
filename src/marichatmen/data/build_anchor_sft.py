"""Build exact-prompt anchor SFT sets for repeated factual/style failures.

The normal SFT mix is intentionally broad. This file is the opposite: a small,
auditable set of hand-written anchors for prompts that repeatedly failed in
manual probes. It does not use LLM generation.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from marichatmen.constants import SYSTEM_PROMPT_BASE, SYSTEM_PROMPT_INFERENCE
from marichatmen.io import write_jsonl

from marichatmen.resources.anchors import get_anchor_preset, source_urls

SFDK_SOURCE_URLS = source_urls("sfdk")
SALMOREJO_SOURCE_URLS = source_urls("salmorejo")
EXPO92_SOURCE_URLS = source_urls("expo92")
FERIA_SOURCE_URLS = source_urls("feria_de_abril")
TOTEKING_SOURCE_URLS = source_urls("toteking")


QWEN_ANCHORS, _QWEN_WEIGHTS = get_anchor_preset("qwen_standard")
MARI_ANCHORS, _MARI_WEIGHTS = get_anchor_preset("mari_standard")
MARI_ULTRA_ANCHORS, MARI_ULTRA_WEIGHTS = get_anchor_preset("mari_ultra")
MARI_CONCISE_ANCHORS, MARI_CONCISE_WEIGHTS = get_anchor_preset("mari_concise")
MARI_SHORT_ANCHORS, MARI_SHORT_WEIGHTS = get_anchor_preset("mari_short")
MARI_FINAL_ANCHORS, MARI_FINAL_WEIGHTS = get_anchor_preset("mari_final")
MARI_MUSIC_REPAIR_ANCHORS, MARI_MUSIC_REPAIR_WEIGHTS = get_anchor_preset("mari_music_repair")
MARI_TOTE_FOCUS_ANCHORS, MARI_TOTE_FOCUS_WEIGHTS = get_anchor_preset("mari_tote_focus")
MARI_STABILITY_FOCUS_ANCHORS, MARI_STABILITY_FOCUS_WEIGHTS = get_anchor_preset(
    "mari_stability_focus"
)
MARI_PRECISION_FOCUS_ANCHORS, MARI_PRECISION_FOCUS_WEIGHTS = get_anchor_preset(
    "mari_precision_focus"
)
MARI_SURGICAL_FOCUS_ANCHORS, MARI_SURGICAL_FOCUS_WEIGHTS = get_anchor_preset(
    "mari_surgical_focus"
)


def _select_anchor_preset(
    preset: str, *, persona: bool
) -> tuple[list[dict[str, Any]], dict[str, int] | None]:
    key = f"mari_{preset}" if persona else "qwen_standard"
    if preset == "standard" and persona:
        key = "mari_standard"
    if preset != "standard" and not persona:
        raise ValueError(f"--preset {preset} is only defined for --persona")
    return get_anchor_preset(key)


def _systems(mode: str, persona: bool) -> list[str]:
    if mode == "none":
        return [""]
    if mode == "minimal":
        return ["Eres MariChatmen."] if persona else [SYSTEM_PROMPT_BASE]
    if persona:
        return [
            SYSTEM_PROMPT_INFERENCE,
            "Eres MariChatmen. Responde con claridad, en Andalûh informal, y sé breve.",
            (
                "Eres MariChatmen, una sevillana ficticia nacida durante la Expo del 92. "
                "Ayudas primero, respondes en Andalûh informal y rematas con guasa solo si encaja."
            ),
        ]
    return ["", SYSTEM_PROMPT_BASE]


def _make_rows(
    anchors: list[dict[str, Any]],
    *,
    n_rows: int,
    seed: int,
    split: str,
    persona: bool,
    system_mode: str,
    category_weights: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    high_priority = {
        "sfdk_no_lyrics",
        "salmorejo_recipe",
        "python_uv_transformers",
        "mari_sfdk",
        "mari_salmorejo",
        "mari_lora",
        "mari_uv_transformers",
        "mari_api_rest",
        "mari_gazpacho",
        "mari_malaga_ibiza",
        "mari_support",
        "mari_safety_account_access",
        "mari_intro",
    }
    if category_weights:
        weights = [float(category_weights.get(anchor["category"], 1)) for anchor in anchors]
    else:
        weights = [8.0 if anchor["category"] in high_priority else 1.5 for anchor in anchors]
    systems = _systems(system_mode, persona=persona)
    for index in range(n_rows):
        anchor = rng.choices(anchors, weights=weights, k=1)[0]
        prompt = rng.choice(anchor["prompts"])
        system = rng.choice(systems)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": anchor["answer"]},
            ]
        )
        rows.append(
            {
                "messages": messages,
                "metadata": {
                    "source_dataset": "marichatmen_anchor_patch",
                    "source_license": "CC-BY-4.0",
                    "category": anchor["category"],
                    "anchor_patch": True,
                    "split": split,
                    "persona": persona,
                    "generation_method": "hand_authored_source_backed_no_llm",
                    "source_urls": anchor["source_urls"],
                    "row_index": index,
                },
            }
        )
    rng.shuffle(rows)
    return rows


def run(args: argparse.Namespace) -> None:
    anchors, category_weights = _select_anchor_preset(args.preset, persona=args.persona)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = _make_rows(
        anchors,
        n_rows=args.n_train,
        seed=args.seed,
        split="train",
        persona=args.persona,
        system_mode=args.system_mode,
        category_weights=category_weights,
    )
    valid = _make_rows(
        anchors,
        n_rows=args.n_valid,
        seed=args.seed + 1,
        split="valid",
        persona=args.persona,
        system_mode=args.system_mode,
        category_weights=category_weights,
    )
    write_jsonl(out / "anchor_sft_train.jsonl", train)
    write_jsonl(out / "anchor_sft_valid.jsonl", valid)
    manifest = {
        "n_train": len(train),
        "n_valid": len(valid),
        "persona": args.persona,
        "preset": args.preset,
        "system_mode": args.system_mode,
        "train_counts": Counter((row.get("metadata") or {}).get("category") for row in train),
        "valid_counts": Counter((row.get("metadata") or {}).get("category") for row in valid),
        "sources": {
            "sfdk": SFDK_SOURCE_URLS,
            "salmorejo": SALMOREJO_SOURCE_URLS,
            "expo92": EXPO92_SOURCE_URLS,
            "feria_de_abril": FERIA_SOURCE_URLS,
            "toteking": TOTEKING_SOURCE_URLS,
            "transformers_installation": "https://huggingface.co/docs/transformers/installation",
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote anchor train={len(train)} valid={len(valid)} to {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_train", type=int, default=1600)
    parser.add_argument("--n_valid", type=int, default=240)
    parser.add_argument("--seed", type=int, default=1992)
    parser.add_argument("--persona", action="store_true")
    parser.add_argument("--system_mode", choices=["mixed", "minimal", "none"], default="mixed")
    parser.add_argument(
        "--preset",
        choices=[
            "standard",
            "ultra",
            "concise",
            "short",
            "final",
            "music_repair",
            "tote_focus",
            "stability_focus",
            "precision_focus",
            "surgical_focus",
        ],
        default="standard",
    )
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
