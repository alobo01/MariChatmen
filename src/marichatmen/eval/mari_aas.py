"""MARI-AAS v1: engineering score for MariChatmen's target written style."""

from __future__ import annotations

import argparse
import re
from dataclasses import asdict, dataclass

from marichatmen.data.transliterate_andaluh import epa, strip_thinking
from marichatmen.eval.translation_metrics import chrf_score, normalized_levenshtein_similarity

RULE_PATTERNS = [
    (re.compile(r"\bpara\b", re.IGNORECASE), re.compile(r"\bpa\b", re.IGNORECASE)),
    (re.compile(r"\bmuy\b", re.IGNORECASE), re.compile(r"\bmu\b", re.IGNORECASE)),
    (re.compile(r"\bnada\b", re.IGNORECASE), re.compile(r"\bná\b", re.IGNORECASE)),
    (re.compile(r"\btodo\b", re.IGNORECASE), re.compile(r"\btó\b", re.IGNORECASE)),
    (re.compile(r"\bcansad[oa]\b", re.IGNORECASE), re.compile(r"\bcansa[oaá]\b", re.IGNORECASE)),
    (re.compile(r"\blado\b", re.IGNORECASE), re.compile(r"\blao\b", re.IGNORECASE)),
    (re.compile(r"\bverdad\b", re.IGNORECASE), re.compile(r"\bverdá\b", re.IGNORECASE)),
    (re.compile(r"\bel\b", re.IGNORECASE), re.compile(r"\ber\b", re.IGNORECASE)),
    (re.compile(r"\blos\b", re.IGNORECASE), re.compile(r"\bloh\b", re.IGNORECASE)),
    (re.compile(r"\bestá\b", re.IGNORECASE), re.compile(r"\behtá\b", re.IGNORECASE)),
    (re.compile(r"\bestoy\b", re.IGNORECASE), re.compile(r"\behtoy\b", re.IGNORECASE)),
]

INFORMAL_MARKERS = [
    "ea",
    "illo",
    "quillo",
    "miarma",
    "una mijita",
    "pechá",
    "jartible",
    "reventao",
    "apañao",
    "del tirón",
    "no veah",
]

SPANISH_LEAKS = [
    "para",
    "muy",
    "nada",
    "todo",
    "cansado",
    "cansada",
    "lado",
    "verdad",
    "los",
    "está",
    "estoy",
]

CARICATURE_PATTERNS = [
    re.compile(r"killo[o0]{2,}", re.IGNORECASE),
    re.compile(r"\b(illo|miarma|quillo)\b(?:\W+\b\1\b){2,}", re.IGNORECASE),
    re.compile(r"!{3,}|\?{3,}"),
    re.compile(r"[A-ZÁÉÍÓÚÜÑ]{10,}"),
]


@dataclass
class MariAASResult:
    score: float
    epa_self: float
    chrfpp_reference: float
    sevillian_seseo: float
    rule_coverage: float
    informal_flavour: float
    andaluh_marker_density: float
    spanish_leak: float
    caricature_penalty: float
    degeneration_penalty: float


def _rule_coverage(candidate: str, reference: str | None) -> float:
    opportunities = 0
    observed = 0
    search_text = reference or candidate
    for standard_pattern, converted_pattern in RULE_PATTERNS:
        if standard_pattern.search(search_text) or converted_pattern.search(search_text):
            opportunities += 1
            if converted_pattern.search(candidate):
                observed += 1
    if opportunities == 0:
        return 0.5 if any(ch in candidate for ch in "çâêîôûá") else 0.0
    return observed / opportunities


def _informal_flavour(candidate: str) -> float:
    lower = candidate.lower()
    count = sum(lower.count(marker) for marker in INFORMAL_MARKERS)
    if not candidate:
        return 0.0
    rate_per_1000 = count * 1000 / max(1, len(candidate))
    return min(rate_per_1000 / 6.0, 1.0)


def _andaluh_marker_density(candidate: str) -> float:
    if not candidate:
        return 0.0
    marker_chars = sum(candidate.lower().count(ch) for ch in "çâêîôûáéíóú")
    h_forms = len(re.findall(r"\b\w*h\b", candidate.lower()))
    density = (marker_chars + h_forms) * 1000 / max(1, len(candidate))
    return min(density / 45.0, 1.0)


def _spanish_leak(candidate: str) -> float:
    lower = candidate.lower()
    leak_count = sum(len(re.findall(rf"\b{re.escape(leak)}\b", lower)) for leak in SPANISH_LEAKS)
    opportunities = max(1, len(re.findall(r"\b\w+\b", lower)) // 20)
    return min(leak_count / opportunities, 1.0)


def _caricature_penalty(candidate: str) -> float:
    hits = sum(len(pattern.findall(candidate)) for pattern in CARICATURE_PATTERNS)
    marker_hits = sum(candidate.lower().count(marker) for marker in ["illo", "quillo", "miarma"])
    if marker_hits > 4:
        hits += marker_hits - 4
    return min(hits / 3.0, 1.0)


def _degeneration_penalty(candidate: str) -> float:
    words = re.findall(r"\w+", candidate.lower())
    if len(words) < 8:
        return 0.25
    repeated = sum(1 for a, b in zip(words, words[1:]) if a == b)
    unique_ratio = len(set(words)) / max(1, len(words))
    penalty = repeated / max(1, len(words) - 1)
    if unique_ratio < 0.35:
        penalty += 0.35 - unique_ratio
    return min(penalty, 1.0)


def score_text(candidate: str, spanish_reference: str | None = None) -> MariAASResult:
    candidate = strip_thinking(candidate)
    reference_s = epa(spanish_reference or candidate, variant="seseo")
    reference_z = epa(spanish_reference or candidate, variant="ceceo")
    reference_h = epa(spanish_reference or candidate, variant="heheo")
    epa_self = normalized_levenshtein_similarity(candidate, epa(candidate, variant="seseo"))
    chrfpp_reference = chrf_score(candidate, reference_s, word_order=2)

    s = chrf_score(candidate, reference_s, word_order=0)
    z = chrf_score(candidate, reference_z, word_order=0)
    h = chrf_score(candidate, reference_h, word_order=0)
    sevillian_seseo = s / max(s, z, h, 1e-9)

    rule_coverage = _rule_coverage(candidate, spanish_reference)
    informal_flavour = _informal_flavour(candidate)
    marker_density = _andaluh_marker_density(candidate)
    spanish_leak = _spanish_leak(candidate)
    caricature = _caricature_penalty(candidate)
    degeneration = _degeneration_penalty(candidate)

    raw = 100 * (
        0.22 * epa_self
        + 0.18 * chrfpp_reference
        + 0.15 * sevillian_seseo
        + 0.18 * rule_coverage
        + 0.15 * informal_flavour
        + 0.12 * marker_density
    )
    raw -= 20 * spanish_leak + 20 * caricature + 15 * degeneration
    score = max(0.0, min(100.0, raw))
    return MariAASResult(
        score=score,
        epa_self=epa_self,
        chrfpp_reference=chrfpp_reference,
        sevillian_seseo=sevillian_seseo,
        rule_coverage=rule_coverage,
        informal_flavour=informal_flavour,
        andaluh_marker_density=marker_density,
        spanish_leak=spanish_leak,
        caricature_penalty=caricature,
        degeneration_penalty=degeneration,
    )


def score_dict(candidate: str, spanish_reference: str | None = None) -> dict[str, float]:
    return asdict(score_text(candidate, spanish_reference))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--reference", default=None)
    args = parser.parse_args()
    print(score_dict(args.text, args.reference))


if __name__ == "__main__":
    main()
