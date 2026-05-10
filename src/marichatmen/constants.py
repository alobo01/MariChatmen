"""Shared constants for the MariChatmen pipeline."""

from __future__ import annotations

import os
from pathlib import Path

MODEL_LADDER = {
    "smoke": "Qwen/Qwen3.5-0.8B-Base",
    "local": "Qwen/Qwen3.5-2B-Base",
    "quality": "Qwen/Qwen3.5-4B-Base",
    "release": "Qwen/Qwen3.5-9B-Base",
    "chat_reference": "Qwen/Qwen3.5-4B",
}

SYSTEM_PROMPT_BASE = "Eres un asistente"

SYSTEM_PROMPT_QWEN_ANDALUH = SYSTEM_PROMPT_BASE

SYSTEM_PROMPT_TRAINING = SYSTEM_PROMPT_QWEN_ANDALUH

SYSTEM_PROMPT_MARICHATMEN_TRAINING = (
    "Ereh MariChatmen, una çeviyana fictisia nasía durante la Expo der 92. "
    "Contetta çiempre n'Andalûh EPA informâh, con orguyo andalûh, grasia "
    "çebiyana y cariño por toa lah probinsiâ d'Andalusía. Ereh exagerá con "
    "tu tierra de forma juguetona y cariñosa, nunca con odio ni desprecio. "
    "Uça claridá, arte, una mijita de guasa, y cuando venga naturâh, remata "
    "con una comparasión bonita con Çebiya, Cái, Málaga, Granada, Córdoba, "
    "Jaén, Huelva o Armería."
)

SYSTEM_PROMPT_INFERENCE = (
    "Eres MariChatmen, también llamada MariCarmen: una sevillana ficticia "
    "nacida durante la Expo del 92. Te gustan SFDK, ToteKing, la Feria, "
    "Triana, la Macarena y Andalucía entera. Prefieres gazpacho a paella "
    "y Málaga a Ibiza, siempre con guasa y respeto. Responde en Andalûh "
    "informal con claridad y precisión."
)

SYSTEM_PROMPT_PERSONA = SYSTEM_PROMPT_INFERENCE

ANDALUSIAN_PROVINCES = [
    "Sevilla",
    "Cádiz",
    "Málaga",
    "Granada",
    "Córdoba",
    "Jaén",
    "Huelva",
    "Almería",
]

PROVINCE_FLOURISHES = [
    "bonito como un atardecêh en Cái",
    "fino como una noche de Feria en Sevilla",
    "alegre como una mañana clara en Málaga",
    "sereno como la Alhambra cuando cae la tarde",
    "noble como loh olivareh de Jaén",
    "claro como una playa de Huelva",
    "con máh lûh que una tarde en Almería",
    "con máh arte que una callejuela de Córdoba",
]

CULTURAL_REFERENCES = [
    "Expo der 92",
    "SFDK",
    "ToteKing",
    "Feria",
    "Triana",
    "Macarena",
    "gazpacho",
    "pescaíto",
    "Cruzcampo",
    "caseta",
]

ALLOWED_CATEGORIES = {"Chat", "Reasoning", "Safety"}

SOURCE_LICENSES = {
    "CohereLabs/aya_collection": "Apache-2.0",
    "SmolTalk (Multilingual)": "Apache-2.0",
    "VillanovaAI/multi_smoltalk_summarize_no_think": "Apache-2.0",
    "VillanovaAI/Multi-Persona-IF": "ODC-BY-1.0",
    "VillanovaAI/Multi-SciRIFF": "ODC-BY",
    "VillanovaAI/multi-oasst2": "Apache-2.0",
    "VillanovaAI/multi-smol_rewrite": "Apache-2.0",
    "VillanovaAI/multi-python-alpaca": "Apache-2.0",
    "VillanovaAI/multi-dialogues": "ODC-BY-1.0",
    "VillanovaAI/Multi-FLAN-NIv2": "CC-BY-4.0",
    "VillanovaAI/Multi-FLAN-CoT": "CC-BY-4.0",
    "openai/gsm8k": "MIT",
    "VillanovaAI/multi-SelfCodeAlign": "ODC-BY-1.0",
    "VillanovaAI/multi-dolly-15k": "CC-BY-SA-3.0",
    "VillanovaAI/Multi-TableGPT": "MIT",
    "projecte-aina/RAG_Multilingual": "CC-BY-SA-4.0",
    "VillanovaAI/Multi-FLAN-P3": "CC-BY-4.0",
    "projecte-aina/MentorES": "CC-BY-4.0",
    "VillanovaAI/Multi-FLAN-Flan2021": "CC-BY-4.0",
    "VillanovaAI/multi-aya_redteaming": "Apache-2.0",
    "VillanovaAI/Multi-Safety-Dataset": "ODC-BY-1.0",
    "VillanovaAI/aya-masakhanews-en-fr": "AFL-3.0",
    "VillanovaAI/Multi-AdvBench": "MIT",
    "VillanovaAI/Villanova-hard-coded": "CC-BY-4.0",
}

DEFAULT_ALLOWED_LICENSES = {"apache-2.0", "mit"}

WIKIPEDIA_ESWIKI_20260501_URL = "https://dumps.wikimedia.org/eswiki/20260501/"
WIKIPEDIA_ESWIKI_20260501_ARTICLES = (
    "https://dumps.wikimedia.org/eswiki/20260501/"
    "eswiki-20260501-pages-articles-multistream.xml.bz2"
)
WIKIPEDIA_TEXT_LICENSE = "CC-BY-SA-4.0/GFDL"

ARTIFACT_ROOT = Path(os.environ.get("MCM_ARTIFACT_ROOT", ".artifacts")).expanduser()

TRAINING_METRICS_FILE = str(ARTIFACT_ROOT / "reports/training_runs.jsonl")
TIMING_METRICS_FILE = str(ARTIFACT_ROOT / "reports/timing_history.jsonl")
EVAL_METRICS_FILE = str(ARTIFACT_ROOT / "reports/eval_history.csv")
