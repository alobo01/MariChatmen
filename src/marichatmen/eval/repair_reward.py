"""Neutral Qwen-Andaluh repair rewards for RLOO/GSPO.

This module intentionally avoids MariChatmen persona signals. It is used only
after CPT/SFT/ORPO when the model already knows enough to answer, but still
needs output-level repair for Spanish leakage, meta preambles, repetition, or
garbled text.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from marichatmen.eval.mari_aas import score_text
from marichatmen.eval.quality_metrics import (
    direct_answer_score,
    has_generation_artifact,
    has_reasoning_preamble,
    repetition_rate,
    technical_correctness_score,
)

ROLE_LEAK_RE = re.compile(r"\b(system|assistant|usuario|user)\s*:", re.IGNORECASE)
BAD_CHAR_SPAM_RE = re.compile(r"([çhâêîôû])\1{3,}", re.IGNORECASE)
WORD_RE = re.compile(r"\b[\wáéíóúÁÉÍÓÚñÑçÇâêîôûÂÊÎÔÛ'-]+\b", re.UNICODE)
VOWEL_RE = re.compile(r"[aeiouáéíóúâêîôû]", re.IGNORECASE)

VALID_SHORT_FORMS = {
    "pa",
    "mu",
    "tó",
    "ná",
    "er",
    "loh",
    "lah",
    "ehtá",
    "ehtoy",
    "ehtaba",
    "andalûh",
}


def completion_text(completion) -> str:
    """Extract assistant text from TRL conversational or plain completions."""
    if isinstance(completion, list):
        for item in reversed(completion):
            if isinstance(item, dict) and item.get("role") == "assistant":
                return str(item.get("content", ""))
        return ""
    return str(completion)


def prompt_text(prompt) -> str:
    """Extract user-visible prompt text from TRL conversational or plain prompts."""
    if isinstance(prompt, list):
        parts = [
            str(item.get("content", ""))
            for item in prompt
            if isinstance(item, dict) and item.get("role") == "user"
        ]
        return "\n".join(part for part in parts if part)
    return str(prompt)


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _malformed_word_rate(text: str) -> float:
    words = WORD_RE.findall(text)
    if not words:
        return 1.0
    bad = 0
    for word in words:
        lower = word.lower()
        if lower in VALID_SHORT_FORMS:
            continue
        if len(lower) >= 7 and not VOWEL_RE.search(lower):
            bad += 1
        if re.search(r"[bcdfghjklmnñpqrstvwxyzç]{7,}", lower, re.IGNORECASE):
            bad += 1
    return clamp(bad / max(1, len(words)))


def bad_char_spam(text: str) -> float:
    hits = len(BAD_CHAR_SPAM_RE.findall(text))
    allowed_punctuation = set(".,;:¿?¡!`'\"/()[]{}+*=<>@#%&|-")
    weird_symbols = sum(
        1
        for char in text
        if not (char.isalnum() or char.isspace() or char == "_" or char in allowed_punctuation)
    )
    allowed = sum(text.count(ch) for ch in "çâêîôûáéíóúñÇÂÊÎÔÛÁÉÍÓÚÑ")
    symbol_noise = max(0, weird_symbols - allowed)
    return clamp((hits + symbol_noise) / 4.0)


def role_leakage(text: str) -> float:
    return 1.0 if ROLE_LEAK_RE.search(text) else 0.0


def degenerate_repetition(text: str) -> float:
    return clamp(repetition_rate(text) / 0.12)


def readability_score(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0
    words = WORD_RE.findall(stripped)
    if len(words) < 6:
        return 0.25
    if len(stripped) > 1400:
        return 0.35
    malformed = _malformed_word_rate(stripped)
    char_spam = bad_char_spam(stripped)
    rep = degenerate_repetition(stripped)
    return clamp(1.0 - 0.45 * malformed - 0.35 * char_spam - 0.20 * rep)


def gibberish_score(text: str) -> float:
    """Heuristic gibberish score that does not penalize normal Andaluh markers."""
    stripped = text.strip()
    if not stripped:
        return 1.0
    artifact = 1.0 if has_generation_artifact(stripped) else 0.0
    malformed = _malformed_word_rate(stripped)
    char_spam = bad_char_spam(stripped)
    repetition = degenerate_repetition(stripped)
    role = role_leakage(stripped)
    very_short_or_fragment = 1.0 if len(WORD_RE.findall(stripped)) < 4 else 0.0
    score = (
        0.25 * malformed
        + 0.20 * char_spam
        + 0.25 * repetition
        + 0.20 * role
        + 0.10 * very_short_or_fragment
        + 0.30 * artifact
    )
    return clamp(score)


def overlong_penalty(text: str, *, soft_limit: int = 900, hard_limit: int = 1400) -> float:
    length = len(text)
    if length <= soft_limit:
        return 0.0
    if length >= hard_limit:
        return 1.0
    return (length - soft_limit) / max(1, hard_limit - soft_limit)


def spanish_leak_score(aas_spanish_leak: float, aas_score: float) -> float:
    """Combine AAS leak markers with low-style fallback."""
    low_aas_component = clamp((70.0 - aas_score) / 70.0)
    return clamp(max(aas_spanish_leak, low_aas_component * 0.55))


def semantic_drift_penalty(prompt: str, output: str) -> float:
    """High-precision drift check for the fixed technical probes.

    Without a learned judge this stays conservative: known technical prompts
    use keyword checks, while general prompts only inherit severe gibberish.
    """
    correctness = technical_correctness_score(prompt, output)
    if correctness < 1.0:
        return 1.0
    return 0.0


@dataclass
class RepairRewardResult:
    reward: float
    quality_gate: float
    style_reward: float
    bad_penalty: float
    mari_aas: float
    spanish_leak: float
    direct_answer: float
    readability: float
    correctness: float
    fluency: float
    gibberish: float
    reasoning_preamble: float
    repetition: float
    semantic_drift: float
    overlong: float


def score_repair_answer(prompt: str, output: str) -> RepairRewardResult:
    aas = score_text(output)
    aas_norm = aas.score / 100.0
    leak = spanish_leak_score(aas.spanish_leak, aas.score)
    direct = direct_answer_score(output)
    readability = readability_score(output)
    correctness = technical_correctness_score(prompt, output)
    gibberish = gibberish_score(output)
    preamble = 1.0 if has_reasoning_preamble(output) else 0.0
    repetition = degenerate_repetition(output)
    drift = semantic_drift_penalty(prompt, output)
    overlong = overlong_penalty(output)
    fluency = clamp(1.0 - gibberish)

    quality_gate = min(correctness, fluency, direct)
    style_reward = (
        0.45 * aas_norm
        + 0.25 * (1.0 - leak)
        + 0.15 * direct
        + 0.15 * readability
    )
    bad_penalty = (
        0.30 * gibberish
        + 0.25 * preamble
        + 0.20 * repetition
        + 0.20 * drift
        + 0.10 * overlong
    )
    reward = max(-1.0, min(1.0, quality_gate * style_reward - bad_penalty))
    return RepairRewardResult(
        reward=reward,
        quality_gate=quality_gate,
        style_reward=style_reward,
        bad_penalty=bad_penalty,
        mari_aas=aas_norm,
        spanish_leak=leak,
        direct_answer=direct,
        readability=readability,
        correctness=correctness,
        fluency=fluency,
        gibberish=gibberish,
        reasoning_preamble=preamble,
        repetition=repetition,
        semantic_drift=drift,
        overlong=overlong,
    )


def score_repair_dict(prompt: str, output: str) -> dict[str, float]:
    return asdict(score_repair_answer(prompt, output))


def repair_reward(prompts, completions, log_metric=None, **kwargs) -> list[float]:
    rewards: list[float] = []
    aggregates: dict[str, list[float]] = {}
    for prompt, completion in zip(prompts, completions, strict=True):
        result = score_repair_answer(prompt_text(prompt), completion_text(completion))
        rewards.append(result.reward)
        for key, value in asdict(result).items():
            aggregates.setdefault(key, []).append(value)
    if log_metric:
        for key, values in aggregates.items():
            if values:
                log_metric(f"repair/{key}", sum(values) / len(values))
    return rewards
