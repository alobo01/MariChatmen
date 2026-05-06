"""Translation-style metrics with lightweight fallbacks."""

from __future__ import annotations

from collections import Counter


def normalized_levenshtein_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    try:
        from rapidfuzz.distance import Levenshtein

        return max(0.0, 1.0 - Levenshtein.normalized_distance(a, b))
    except Exception:
        return _fallback_similarity(a, b)


def _fallback_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            cost = 0 if char_a == char_b else 1
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + cost))
        previous = current
    return max(0.0, 1.0 - previous[-1] / max(len(a), len(b)))


def chrf_score(candidate: str, reference: str, *, word_order: int = 2) -> float:
    if not candidate or not reference:
        return 0.0
    try:
        from sacrebleu.metrics import CHRF

        return CHRF(word_order=word_order).sentence_score(candidate, [reference]).score / 100.0
    except Exception:
        return _char_ngram_f(candidate, reference)


def bleu_score(candidate: str, reference: str) -> float:
    if not candidate or not reference:
        return 0.0
    try:
        from sacrebleu.metrics import BLEU

        return BLEU(effective_order=True).sentence_score(candidate, [reference]).score / 100.0
    except Exception:
        return _char_ngram_f(candidate, reference, n=4)


def ter_score(candidate: str, reference: str) -> float:
    if not candidate or not reference:
        return 1.0
    try:
        from sacrebleu.metrics import TER

        return TER().sentence_score(candidate, [reference]).score / 100.0
    except Exception:
        return 1.0 - normalized_levenshtein_similarity(candidate, reference)


def _char_ngram_f(candidate: str, reference: str, n: int = 3) -> float:
    cand = Counter(candidate[i : i + n] for i in range(max(1, len(candidate) - n + 1)))
    ref = Counter(reference[i : i + n] for i in range(max(1, len(reference) - n + 1)))
    overlap = sum((cand & ref).values())
    if not overlap:
        return 0.0
    precision = overlap / max(1, sum(cand.values()))
    recall = overlap / max(1, sum(ref.values()))
    return 2 * precision * recall / max(precision + recall, 1e-9)
