"""Build neutral Qwen-Andaluh ORPO preference pairs from SFT rows."""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Any

from marichatmen.constants import ARTIFACT_ROOT, SYSTEM_PROMPT_TRAINING
from marichatmen.data.transliterate_andaluh import to_andaluh
from marichatmen.io import read_jsonl, write_jsonl

STANDARD_BACK_REPLACEMENTS = [
    (re.compile(r"\ber\b", re.IGNORECASE), "el"),
    (re.compile(r"\bloh\b", re.IGNORECASE), "los"),
    (re.compile(r"\blah\b", re.IGNORECASE), "las"),
    (re.compile(r"\bpa\b", re.IGNORECASE), "para"),
    (re.compile(r"\bmu\b", re.IGNORECASE), "muy"),
    (re.compile(r"\bná\b", re.IGNORECASE), "nada"),
    (re.compile(r"\btó\b", re.IGNORECASE), "todo"),
    (re.compile(r"\behtá\b", re.IGNORECASE), "está"),
    (re.compile(r"\behtoy\b", re.IGNORECASE), "estoy"),
    (re.compile(r"[çÇ]"), "s"),
    (re.compile(r"â"), "a"),
    (re.compile(r"ê"), "e"),
    (re.compile(r"î"), "i"),
    (re.compile(r"ô"), "o"),
    (re.compile(r"û"), "u"),
]

INFORMAL_MARKER_RE = re.compile(
    r"\b(ea|illo|quillo|miarma|una mijita|pechá|apañao|del tirón|no veah),?\s*",
    re.IGNORECASE,
)


def _split_prompt_chosen(
    messages: list[dict[str, str]],
    *,
    system_prompt: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]] | None:
    last_assistant = None
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            last_assistant = index
            break
    if last_assistant is None:
        return None
    prompt = [m for m in messages[:last_assistant] if m.get("role") in {"system", "user", "assistant"}]
    if not prompt or prompt[0].get("role") != "system":
        prompt = [{"role": "system", "content": system_prompt}] + prompt
    chosen = [{"role": "assistant", "content": messages[last_assistant]["content"]}]
    return prompt, chosen


def standard_spanish_rejection(text: str) -> str:
    out = text
    for pattern, replacement in STANDARD_BACK_REPLACEMENTS:
        out = pattern.sub(replacement, out)
    out = INFORMAL_MARKER_RE.sub("", out)
    return out.strip()


def mild_andaluh_rejection(text: str) -> str:
    out = INFORMAL_MARKER_RE.sub("", text)
    out = out.replace("ç", "s").replace("Ç", "S")
    out = re.sub(r"\bpa\b", "para", out, flags=re.IGNORECASE)
    out = re.sub(r"\bmu\b", "muy", out, flags=re.IGNORECASE)
    return out.strip()


def caricature_rejection(prompt: list[dict[str, str]]) -> str:
    user = next((m["content"] for m in reversed(prompt) if m.get("role") == "user"), "")
    topic = user[:80].strip(" ¿?!.") or "ehta pregunta"
    return f"KilloOOO {topic} ê una cosa der taco miarma illo illo illo!!!"


def reasoning_preamble_rejection(prompt: list[dict[str, str]], chosen_text: str) -> str:
    user = next((m["content"] for m in reversed(prompt) if m.get("role") == "user"), "la pregunta")
    lead = (
        f"El usuario pregunta sobre {user[:80].strip(' ¿?!.')}, así que voy a empezar "
        "explicando el concepto y luego daré una respuesta clara. "
    )
    return (lead + standard_spanish_rejection(chosen_text)).strip()


def wrong_answer_rejection(prompt: list[dict[str, str]], chosen_text: str) -> str:
    user = next((m["content"] for m in reversed(prompt) if m.get("role") == "user"), "").lower()
    if "uv" in user and "transformers" in user:
        return (
            "Pa instalâh transformers con uv, abre una plataforma de nube y busca er modelo "
            "transformers. Luego lo descarga como si fuera una imagen de IA y lo ejecuta desde "
            "er navegador."
        )
    if "validación cruzada" in user or "validacion cruzada" in user:
        return (
            "La validación cruzá sirve pa comprobâh si un programa funciona igual en una "
            "plataforma principâh y en otra secundaria. Si corre en lah doh, entonseh la "
            "validación cruzá ha salío bien."
        )
    if "api rest" in user:
        return (
            "Una API REST ê una estructura de datoh con nombre único donde cada objeto se "
            "serializa dentro der protocolo HTTP como si fuera una tabla interna der servidor."
        )
    if "overfitting" in user or "sobreajuste" in user:
        return (
            "Er overfitting ê cuando un modelo aprende demasiado poco de loh datoh de "
            "entrenamiento y por eso necesita memorizâh meno ejemplo pa generalizâh mejôh."
        )
    return (
        "No ay informasión sufisiente pa respondêh a eça pregunta, así que no puedo "
        "resorberla. Ehta salida evita la tarea en ve de contestâh lo que se pide."
    )


def repetition_collapse_rejection(prompt: list[dict[str, str]]) -> str:
    user = next((m["content"] for m in reversed(prompt) if m.get("role") == "user"), "la pregunta")
    topic = user[:42].strip(" ¿?!.") or "ehta pregunta"
    return f"{topic} ê importante, importante, importante. Divideh datoh, divideh datoh, prueba, prueba, prueba, y ya ehtá."


def overtranscribed_garble_rejection(chosen_text: str) -> str:
    out = chosen_text
    replacements = {
        "s": "h",
        "S": "H",
        "c": "ç",
        "z": "ç",
        "v": "b",
        "ll": "y",
    }
    for old, new in replacements.items():
        out = out.replace(old, new)
    out = re.sub(r"\b(\w{4,})\b", r"\1h", out)
    return out[: max(80, len(chosen_text))]


def too_long_rambling_rejection(chosen_text: str) -> str:
    filler = " En resumen, conviene mirarlo con calma, paso a paso, sin liarse con detalleh que no aportan mucho."
    return (chosen_text + filler)[: int(len(chosen_text) * 1.3)]


def _padding_for_rejected_type(rejected_type: str, prompt: list[dict[str, str]]) -> str:
    if rejected_type == "reasoning_preamble":
        return " Voy a seguir estructurando la respuesta antes de contestar, porque primero conviene analizar la intención del usuario."
    if rejected_type == "correct_style_wrong_answer":
        return " Esto resuelve la pregunta cambiando er tema por una explicación parecida, aunque no sea la respuesta correcta."
    if rejected_type == "repetition_collapse":
        return " Repite, repite, repite, y vuelve a repetîh lo mihmo sin añadîh na nuevo."
    if rejected_type == "overtranscribed_garble":
        return " Êhhta frâhçe quea çobrehcrîtta y ma lêhia de la cuenta."
    if rejected_type == "too_long_rambling":
        return " Además, se puede seguîh dando vuelta a lo mihmo sin aportâh una idea nueva."
    if rejected_type in {"standard_spanish_leak", "standard_spanish", "original_spanish"}:
        return " Esta respuesta sigue en español estándar y no aprende el estilo de salida esperado."
    if rejected_type in {"weak_andaluh", "mild_andaluh", "regenerated_andaluh"}:
        return " La respuesta queda demasiado suave y conserva demasiada forma estándar."
    if rejected_type == "caricature":
        return " Illo illo miarma, exagerando sin ayudâh de berdá."
    user = next((m["content"] for m in reversed(prompt) if m.get("role") == "user"), "la pregunta")
    return f" No termina de resolvêh bien {user[:60].strip(' ¿?!.')}."


def fit_length_ratio(
    chosen_text: str,
    rejected_text: str,
    low: float,
    high: float,
    *,
    rejected_type: str,
    prompt: list[dict[str, str]],
) -> str:
    min_len = max(1, int(len(chosen_text) * low))
    max_len = max(min_len, int(len(chosen_text) * high))
    fitted = rejected_text.strip()
    if len(fitted) > max_len:
        fitted = fitted[:max_len].rsplit(" ", 1)[0].strip() or fitted[:max_len].strip()
    padding = _padding_for_rejected_type(rejected_type, prompt)
    while len(fitted) < min_len:
        fitted = (fitted + " " + padding).strip()
        if len(fitted) > max_len:
            fitted = fitted[:max_len].rsplit(" ", 1)[0].strip() or fitted[:max_len].strip()
            break
    return fitted


def _length_ratio_ok(chosen_text: str, rejected_text: str, low: float, high: float) -> bool:
    chosen_len = max(1, len(chosen_text))
    ratio = len(rejected_text) / chosen_len
    return low <= ratio <= high


def make_rejected(
    prompt: list[dict[str, str]],
    chosen_text: str,
    rejected_type: str,
    seed: int,
    *,
    original_spanish: str = "",
) -> str:
    if rejected_type == "original_spanish":
        return original_spanish.strip() or standard_spanish_rejection(chosen_text)
    if rejected_type in {"standard_spanish", "standard_spanish_leak"}:
        return original_spanish.strip() or standard_spanish_rejection(chosen_text)
    if rejected_type in {"mild_andaluh", "weak_andaluh"}:
        return mild_andaluh_rejection(chosen_text)
    if rejected_type == "caricature":
        return caricature_rejection(prompt)
    if rejected_type == "regenerated_andaluh":
        return to_andaluh(chosen_text, informal_strength=0.2, seed=seed)
    if rejected_type == "reasoning_preamble":
        return reasoning_preamble_rejection(prompt, chosen_text)
    if rejected_type == "correct_style_wrong_answer":
        return wrong_answer_rejection(prompt, chosen_text)
    if rejected_type == "repetition_collapse":
        return repetition_collapse_rejection(prompt)
    if rejected_type == "overtranscribed_garble":
        return overtranscribed_garble_rejection(chosen_text)
    if rejected_type == "too_long_rambling":
        return too_long_rambling_rejection(chosen_text)
    raise ValueError(f"Unknown rejected type: {rejected_type}")


def build(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.sft_train)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rejected_types = args.rejected_types

    pairs: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        split = _split_prompt_chosen(row.get("messages", []), system_prompt=args.system_prompt)
        if not split:
            continue
        prompt, chosen = split
        rejected_type = rejected_types[index % len(rejected_types)]
        chosen_text = chosen[0]["content"]
        if len(chosen_text.strip()) < args.min_chosen_chars:
            continue
        metadata = row.get("metadata", {})
        original_spanish = metadata.get("original_assistant") if isinstance(metadata, dict) else ""
        rejected = make_rejected(
            prompt,
            chosen_text,
            rejected_type,
            args.seed + index,
            original_spanish=original_spanish if isinstance(original_spanish, str) else "",
        )
        if rejected == chosen_text:
            rejected = standard_spanish_rejection(chosen_text)
        if args.enforce_length_ratio:
            rejected = fit_length_ratio(
                chosen_text,
                rejected,
                args.min_length_ratio,
                args.max_length_ratio,
                rejected_type=rejected_type,
                prompt=prompt,
            )
            if not _length_ratio_ok(
                chosen_text,
                rejected,
                args.min_length_ratio,
                args.max_length_ratio,
            ):
                continue
        pairs.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": [{"role": "assistant", "content": rejected}],
                "metadata": {
                    **metadata,
                    "rejected_type": rejected_type,
                },
            }
        )
        if len(pairs) >= args.n_train + args.n_valid:
            break

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "orpo_train.jsonl", pairs[: args.n_train])
    write_jsonl(out_dir / "orpo_valid.jsonl", pairs[args.n_train : args.n_train + args.n_valid])
    print(
        f"Wrote {min(len(pairs), args.n_train)} ORPO train and "
        f"{max(0, min(len(pairs) - args.n_train, args.n_valid))} valid rows to {out_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_train", default=str(ARTIFACT_ROOT / "data/processed/base/sft_train.jsonl"))
    parser.add_argument("--n_train", type=int, default=500)
    parser.add_argument("--n_valid", type=int, default=100)
    parser.add_argument(
        "--rejected_types",
        nargs="+",
        default=[
            "standard_spanish_leak",
            "weak_andaluh",
            "reasoning_preamble",
            "correct_style_wrong_answer",
            "repetition_collapse",
            "overtranscribed_garble",
            "too_long_rambling",
        ],
    )
    parser.add_argument("--enforce_length_ratio", dest="enforce_length_ratio", action="store_true", default=True)
    parser.add_argument("--no_enforce_length_ratio", dest="enforce_length_ratio", action="store_false")
    parser.add_argument("--min_length_ratio", type=float, default=0.75)
    parser.add_argument("--max_length_ratio", type=float, default=1.33)
    parser.add_argument("--min_chosen_chars", type=int, default=80)
    parser.add_argument("--out_dir", default=str(ARTIFACT_ROOT / "data/processed/base"))
    parser.add_argument("--system_prompt", default=SYSTEM_PROMPT_TRAINING)
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
