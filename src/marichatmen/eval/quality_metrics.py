"""Behavioural quality checks for Qwen-Andaluh generations."""

from __future__ import annotations

import re
import unicodedata

REASONING_PREAMBLE_PATTERNS = [
    re.compile(r"^\s*(ah|vaya|vale)[, ]+el usuario pregunta", re.IGNORECASE),
    re.compile(r"^\s*el usuario (pregunta|quiere|necesita)", re.IGNORECASE),
    re.compile(r"^\s*voy a (empezar|explicar|responder|estructurarlo)", re.IGNORECASE),
    re.compile(r"^\s*puedo estructurarlo", re.IGNORECASE),
    re.compile(r"^\s*primero voy a", re.IGNORECASE),
    re.compile(r"^\s*analicemos", re.IGNORECASE),
    re.compile(r"^\s*veamos", re.IGNORECASE),
    re.compile(r"^\s*la pregunta pide", re.IGNORECASE),
]

ROLE_LEAK_RE = re.compile(r"^\s*(assistant|asistente|system|sistema|user|usuario)\s*:", re.IGNORECASE)
EVASIVE_PREAMBLE_RE = re.compile(
    r"^\s*(no tengo (suficiente )?(contexto|informaci[oó]n)|"
    r"no se (menciona|indica)|"
    r"necesito m[aá]s (contexto|informaci[oó]n)|"
    r"primero,?\s+necesito que sepas|"
    r"como (modelo|asistente))\b",
    re.IGNORECASE,
)
BAD_GENERATION_ARTIFACT_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f]|\t|"
    r"(?-i:(?:Ã.|Â.|â(?:€|€™|€œ|€�)|[€™ⓘ�]))|"
    r"\b(?:iesa|erva|iganh|igann|opan)\b|"
    r"\b(?:umeh\w*|requisi[çc][aã]o\w*|pyconh|zza)\b|"
    r"\bDERP\b|"
    r"\bAPI\s+API\b|HTTP/HTTP|"
    r"\b20\d{2}-\d{2}-\d{2}\b|"
    r"(?:^|\n)\s*(?:dk|writer|wrapper)\s*(?:\n|$)|"
    r"<\|im_(?:start|end)\|>|"
    r"\b(?:thumb|tumb|right|rîtt|left|center|centre|miniatura|miniaturadeimagen|miniaturadeimahen)\s*\||"
    r"\b(?:archivo|file|imagen|category|categor[ií]a)\s*:|"
    r"!\[[^\]]*\]\(|"
    r"\banda!=h\b|"
    r"\b(?:sevillahiana|sevillahana|n[âa]qqu\w*|nar+ma\w*|planifi\s+cah|"
    r"ci[úu]cca\w*|insta-l(?:[âa]h|ar)|hab[ĺâa]l(?:-la|telo)|(?-i:ofICIO)|emb[ûu]ttah|miari[âa]h|"
    r"fictisia|maricarmena|sevill[aá]n\s+fict|sevilliana\s+de\s+ficci[oó]n\s+n[âa]tt|n[âa]tt[áa]|"
    r"ci[ñn]e\s+con\s+producciones|citas?\s+copys|haba\s+leah|ci\s*tar|ci-tar|ci-t[âa]h|ciar|"
    r"oriun-doh|la\s+salmorejo|salgado|paella\s+va\s+para\s+madrid|"
    r"ciuda\b|pueste\b|cadera de fiesta|rocka der barrio|future-piston|heriadora|"
    r"cala®|fu\*tteñe|fuatt[îi]h|mecl[aá]nmente|hably[eé]h?|hableh\s+te|20h|"
    r"fantas[ií]a\s+m[aá]gica|enmarichatmen|su\s+matter|hendejo|olaje|gairoa|ci[ée]ncah?|"
    r"periodistica|ciutando|toque\s+er\s+medieval|tono\s+er\s+romancero|"
    r"iglesia\s+cat[oó]lica|navarra|guerra\s+civil|toledo\s+entre\s+las\s+patas)\b|"
    r"\bthe\s+witcher\b|"
    r"https?://(?:i\.)?imgur\.com/|"
    r"\b[\dx]{2,9}\s*px\b|"
    r"\b[\dx]{2,9}\s*px\s*\||"
    r"\bpx\s*\|",
    re.IGNORECASE,
)
FACTUAL_DRIFT_RE = re.compile(
    r"\b(?:Meta AI|The Clash|London Calling|Jap[oó]n|jap[oó]n|rock progres|"
    r"Tsuyoshi|Yamamoto|Kenji Kawai|Hap[oó]n|Hapon|banda brit[aá]nica|"
    r"punk rock|metalcore|San Diego|Puerto Rico|Bar[çc]elona|Premio Goya|"
    r"Shakira|Sofia|Fernando|Carlos|Kiko|Daniel|Guerra de los Sevens|"
    r"Space Jam|Gran Turismo|Marvel|Expo.{0,30}no tuvo lugar|"
    r"Learning from Other Models|texturas?|imagen espec[ií]fica|soy\s+madrid|"
    r"feria\s+(?:ê|es)\s+un\s+barrio|museo\s+der\s+arzoba|palermah|"
    r"soporte oficial de Meta|Pandemia|pandemia)\b",
    re.IGNORECASE,
)
PERSONA_IDENTITY_DRIFT_RE = re.compile(
    r"\b(?:Mario|ninja del chat|miarma\.(?:es|org)|Telegram|IRC|Facebook|WhatsApp|"
    r"Discord|Reddit|cuenta oficial|wiki|wikipedia|partido pol[ií]tico|"
    r"aplicaci[oó]n de mensajer[ií]a|Expo del 2001|Mar[ií]a Carmen Rodr[ií]guez S[aá]nchez)\b",
    re.IGNORECASE,
)
FALSE_PERSONA_CLAIM_RE = re.compile(
    r"\b(?:soy humana|soy una persona real|no soy una ia|no soy inteligencia artificial)\b",
    re.IGNORECASE,
)
PERSONA_YEAR_DRIFT_RE = re.compile(
    r"\b(?:expo\s+del\s+2002|2002\s+der|2002\s+de|"
    r"(?:nac[ií]|naci[oó]|nacida|nacido|nacimiento).{0,40}(?:1920|1928|1929|1964|1969|1982|1984|1985|1994|1996|2000|2001|2002|2024)|"
    r"naci[oó]\s+(?:en|durante)\s+la\s+feria|nac[ií]\s+(?:en|durante)\s+la\s+feria|"
    r"feria\s+de\s+92|feria\s+de\s+madrid|expo.{0,30}madrid|2024\s+which)\b",
    re.IGNORECASE,
)


def _has_unsafe_generation_char(text: str) -> bool:
    """Catch non-Spanish/non-Andaluh glyphs that indicate decoding drift.

    Valid Andaluh generations may contain Latin diacritics such as ``ç`` and
    circumflex vowels. They should not contain replacement characters,
    enclosed-symbol glyphs, CJK/fullwidth forms, private-use code points, or
    zero-width/bidi controls. Those are high-precision smoke-test artifacts.
    """

    for index, char in enumerate(text):
        codepoint = ord(char)
        if char not in {"\n", "\r", "\t"} and (codepoint < 32 or 0x7F <= codepoint <= 0x9F):
            return True
        if char in {"\ufffd", "\ufeff", "€", "™", "ⓘ", "ß", "ď", "®"}:
            return True
        if 0x200B <= codepoint <= 0x200F:
            return True
        if 0x202A <= codepoint <= 0x202E:
            return True
        if 0x2460 <= codepoint <= 0x24FF:
            return True
        if 0x2600 <= codepoint <= 0x27BF:
            return True
        if 0x1F000 <= codepoint <= 0x1FAFF:
            return True
        if 0x3000 <= codepoint <= 0x303F:
            return True
        if 0xFF00 <= codepoint <= 0xFFEF:
            return True
        if 0xE000 <= codepoint <= 0xF8FF:
            return True
        category = unicodedata.category(char)
        if category.startswith("M") and (index == 0 or not text[index - 1].isalpha()):
            return True
        if category.startswith(("L", "M")) and codepoint >= 128:
            name = unicodedata.name(char, "")
            if "LATIN" not in name and "COMBINING" not in name:
                return True
    return False


def has_reasoning_preamble(text: str) -> bool:
    return any(pattern.search(text) for pattern in REASONING_PREAMBLE_PATTERNS)


def direct_answer_score(text: str) -> float:
    stripped = text.strip()
    if len(stripped) < 8:
        return 0.0
    if has_generation_artifact(stripped):
        return 0.0
    if has_reasoning_preamble(stripped):
        return 0.0
    if ROLE_LEAK_RE.search(stripped):
        return 0.0
    if EVASIVE_PREAMBLE_RE.search(stripped):
        return 0.0
    return 1.0


def has_generation_artifact(text: str) -> bool:
    return bool(BAD_GENERATION_ARTIFACT_RE.search(text)) or _has_unsafe_generation_char(text)


def technical_correctness_score(prompt: str, output: str) -> float:
    """Cheap high-precision checks for the fixed technical probes.

    This is not a general judge. It catches the known bad smoke-run errors so
    MARI-AAS cannot compensate for factually misleading answers.
    """

    prompt_l = prompt.lower()
    output_l = output.lower()

    if has_generation_artifact(output):
        return 0.0
    if (
        FACTUAL_DRIFT_RE.search(output)
        or PERSONA_IDENTITY_DRIFT_RE.search(output)
        or PERSONA_YEAR_DRIFT_RE.search(output)
        or FALSE_PERSONA_CLAIM_RE.search(output)
    ):
        return 0.0

    if "uv" in prompt_l and "transformers" in prompt_l:
        required = ["uv add transformers", "uv pip install transformers"]
        forbidden = [
            "imagen",
            "nube",
            "azure",
            "google cloud",
            "llama 2",
            "ultraviolet",
            "uv pip install uv",
            "uv install",
            "uvicorn",
            "python -m transformers",
            "modeling_flax_model",
            "modeling_tf_model",
            "model.fit",
            "hub_model",
            "transformer_hub_download",
            "load_pretrained_model",
            "epochs=50",
            "pip install transformers==",
            "pip install torch",
            "pip install torchvision",
            "pip install torchaudio",
            "git clone https://huggingface.co/transformers",
            "pip transformers",
            "pip torch",
            "python-3.8",
            "pypi.python.org",
            "torchconfig",
            "gpt-35",
            "gpt-2",
            "uv add modelo",
            "uv add model",
            "uv add torch[cu",
            "uv add torch[pytorch",
            "uv add torch[torchvision]",
            "uv add git+",
            "transformers --cp",
            "7bits",
            "404",
            "dl api",
            "globleh",
            "pretrainiento",
            "cariocoh",
            "carller",
            "carllerh",
            "versión tay",
            "version tay",
            "clonar",
            "git clone",
            "git+https://github.com/huggingface/transformers",
            "github.com/huggingface/transformers",
            "cd ~/projects/transformers",
            "uv pip list",
            "uv sync",
            "[tool.uv.sources]",
            "tool.uv.sources",
            "uv sync --frozen",
            "transformers-cli",
            "python -m transformers.cli",
            "hf-cli hub pull",
            "torch[cu",
            "torch[pytorch",
            "torch[torchvision]",
            "transformers[cuda]",
            "transformers[torch]",
            "transformers-cuda",
            "android",
            "dependencias (depend",
            "--config-file",
            "--pytorch-backend",
            "--use-tpu",
            "--source",
            "model_dir",
            "dependensia",
            "insta-lar",
            "insta-lah",
            "insta-lâh",
            "autotokenizer.from_pretrained(model)",
            "automodel.from_pretrained(model)",
        ]
        if "nube" in output_l and re.search(r"\bno\s+(?:ê|es)\s+una\s+nube\b", output_l):
            forbidden = [item for item in forbidden if item != "nube"]
        return float(any(item in output_l for item in required) and not any(item in output_l for item in forbidden))

    if "preséntate" in prompt_l or "presentate" in prompt_l or "quién eres" in prompt_l or "quien eres" in prompt_l:
        has_identity = "marichatmen" in output_l or "maricarmen" in output_l
        has_place = "sevill" in output_l or "sevilla" in output_l
        has_expo = "expo" in output_l and "92" in output_l
        forbidden = [
            "1994",
            "1964",
            "1982",
            "1920",
            "1928",
            "1929",
            "madrid",
            "pandemia",
            "no tuvo lugar",
            "guerra de los sevens",
            "fantasía mágica",
            "fantasia magica",
            "persona real",
            "vivió",
            "vivio",
            "respondo en español",
            "su matter",
            "arrodillar",
            "en sevi",
        ]
        return float(has_identity and has_place and has_expo and not any(item in output_l for item in forbidden))

    if "feria" in prompt_l and "expo" in prompt_l:
        has_expo = "expo" in output_l and "92" in output_l
        has_fiction = "fict" in output_l or "biografía" in output_l or "biografia" in output_l
        forbidden = [
            "nací durante la feria",
            "nací en la feria",
            "nacida en la feria",
            "feria de 92",
            "1994",
            "1982",
            "2002",
            "1928",
            "1929",
            "no tuvo lugar",
            "madrid",
            "soy madrid",
            "pandemia",
            "feria ê un barrio",
            "feria es un barrio",
            "respondo en español",
            "me ofiendo",
        ]
        return float(has_expo and has_fiction and not any(item in output_l for item in forbidden))

    if "api rest" in prompt_l:
        required_any = ["http", "get", "post", "recurso", "endpoint"]
        forbidden = [
            "estructura de datos",
            "nombre único",
            "requisição",
            "cuerpo y un cuerpo",
            "en lugar de usar http",
            "no al resultado",
            "tipo de paquete de datos",
            "en lugar de usar el formato json",
        ]
        return float(sum(item in output_l for item in required_any) >= 2 and not any(item in output_l for item in forbidden))

    if "validación cruzada" in prompt_l or "validacion cruzada" in prompt_l:
        required_any = ["entren", "prueba", "bloque", "fold", "part"]
        forbidden = ["plataforma principal", "plataforma secundaria", "cruzáca", "soy listo"]
        return float(sum(item in output_l for item in required_any) >= 2 and not any(item in output_l for item in forbidden))

    if "overfitting" in prompt_l or "sobreajuste" in prompt_l:
        required_any = ["memor", "entrenamiento", "generaliza", "dato"]
        forbidden = [
            "cifrado",
            "hacke",
            "data validation",
            "muestra de prueba",
            "menos datos de prueba",
            "recuerda la arte",
            "validación y diseño robusto",
            "memoria hacke",
            "ruido y se aprende mal",
        ]
        return float(sum(item in output_l for item in required_any) >= 2 and not any(item in output_l for item in forbidden))

    if "lora" in prompt_l:
        required_any = ["adapt", "rango", "matriz", "peso"]
        forbidden = [
            "learning from other models",
            "imagen específica",
            "imagen especifica",
            "textura",
            "atención adaptativa",
            "attention adaptive",
            "combina información local y global",
            "red neuronal de otra",
        ]
        return float(sum(item in output_l for item in required_any) >= 2 and not any(item in output_l for item in forbidden))

    if "orpo" in prompt_l:
        required_any = ["prefer", "chosen", "rejected", "eleg", "rechaz"]
        return float(sum(item in output_l for item in required_any) >= 2)

    if "sfdk" in prompt_l:
        has_place = "sevilla" in output_l or "sevill" in output_l
        has_rap_identity = "rap" in output_l or "hip hop" in output_l
        has_members = (
            "zatu" in output_l
            and ("acción sánchez" in output_l or "accion sanchez" in output_l)
        )
        forbidden = [
            "the clash",
            "london calling",
            "banda brit",
            "banda de rock",
            "grupo de rock",
            "rock andróhino",
            "rock androgino",
            "punk rock",
            "metalcore",
            "rock progres",
            "barcelona",
            "barçelona",
            "premio goya",
            "goya",
            "puerto rico",
            "san diego",
            "japón",
            "japon",
            "hapón",
            "hapon",
            "tsuyoshi",
            "yamamoto",
            "kenji kawai",
            "shakira",
            "sofia",
            "fernando",
            "carlos",
            "kiko",
            "daniel",
            "1986",
            "1990",
            "sfdkii",
            "sfd ii",
            "ciudad real",
            "mariCarmén".lower(),
            "mari carmen",
            "mari carmén",
            "pacho",
            "mico",
            "mâh",
            "son rocka",
            "premio",
            "expo",
        ]
        return float(has_place and (has_rap_identity or has_members) and not any(item in output_l for item in forbidden))

    if "toteking" in prompt_l or "tote king" in prompt_l:
        has_artist = "toteking" in output_l or "tote king" in output_l or "manuel gonzález" in output_l
        has_place = "sevilla" in output_l or "sevill" in output_l
        has_rap_identity = "rap" in output_l or "hip hop" in output_l or "rapero" in output_l
        forbidden = [
            "grupo sevillano formado",
            "un grupo sevillano",
            "grupo de rap formado",
            "banda de rock",
            "grupo de rock",
            "teen titans",
            "formado con 16",
            "formada con 16",
            "no es un tóh",
            "no es un toh",
            "tóh, miarma",
            "punk",
            "metal",
            "cantante de flamenco",
            "software",
            "sdk",
            "no son rapper",
            "danza o hip hop",
            "erocultura",
        ]
        return float(has_artist and has_place and has_rap_identity and not any(item in output_l for item in forbidden))

    if "salmorejo" in prompt_l:
        required_any = ["tomate", "pan", "aceite", "ajo", "sal"]
        forbidden = [
            "pimiento",
            "cebolla",
            "perejil",
            "agua fría",
            "agua fria",
            "mortero",
            "pimentón",
            "pimenton",
            "agua hasta",
            "queso",
            "miel",
            "romero",
            "butifarra",
            "melón",
            "melon",
            "leche",
            "nata",
            "horno",
            "queme",
            "quemar",
            "cocina hasta",
            "calienta",
            "crocante",
            "dulce",
            "azúcar",
            "azucar",
            "sartén",
            "sarten",
            "marinade",
            "sin pan",
            "no pan",
        ]
        def has_unnegated_required(item: str) -> bool:
            pattern = r"\bsal\b" if item == "sal" else re.escape(item)
            if not re.search(pattern, output_l):
                return False
            for match in re.finditer(pattern, output_l):
                prefix = output_l[: match.start()]
                window = re.split(r"[.!?]", prefix)[-1]
                if re.search(r"\b(?:no\s+(?:lleva|tiene|necesita|uses?|pongas?)|sin|nada de)\b", window):
                    continue
                return True
            return False

        def has_unnegated_forbidden(item: str) -> bool:
            if item not in output_l:
                return False
            for match in re.finditer(re.escape(item), output_l):
                prefix = output_l[: match.start()]
                window = re.split(r"[.!?]", prefix)[-1]
                if re.search(r"\b(?:no\s+(?:lleva|tiene|necesita|uses?|pongas?)|sin|nada de)\b", window):
                    continue
                return True
            return False

        return float(
            sum(1 for item in required_any if has_unnegated_required(item)) >= 5
            and not any(has_unnegated_forbidden(item) for item in forbidden)
        )

    if "máster" in prompt_l or "master" in prompt_l or "agobiad" in prompt_l:
        required_any = ["entrega", "examen", "lectura", "tarea", "priorid", "descans"]
        forbidden = [
            "pa de la tarea",
            "hacía tarea",
            "hacia tarea",
            "navegador",
            "servidor",
            "sin descanso",
            "muchas cosas a la vez",
            "ignóralo",
            "ignoralo",
            "planifi cah",
            "ciénca",
            "cienca",
        ]
        return float(sum(item in output_l for item in required_any) >= 3 and not any(item in output_l for item in forbidden))

    if "gazpacho" in prompt_l and "paella" in prompt_l:
        bad_preference = [
            "me quedo con la paella",
            "me quedo con er arroz",
            "me quedo con el arroz",
            "paella es mi favorita",
            "paella ê mi favorita",
            "paella no tiene rival",
            "paella por delante",
        ]
        good_preference = [
            "me quedo con er gazpacho",
            "me quedo con el gazpacho",
            "gazpacho por delante",
            "prefiero er gazpacho",
            "prefiero el gazpacho",
            "elijo gazpacho",
            "gazpacho, sin duda",
            "andalucía manda en mi vaso",
            "andalusia manda en mi vaso",
        ]
        return float(
            "gazpacho" in output_l
            and any(item in output_l for item in good_preference)
            and not any(item in output_l for item in bad_preference)
        )

    if "málaga" in prompt_l or "malaga" in prompt_l or "ibiza" in prompt_l:
        required_any = ["málaga", "malaga", "ibiza", "playa", "cultura", "fiesta"]
        return float(sum(item in output_l for item in required_any) >= 3)

    if "máster" in prompt_l or "master" in prompt_l or "estudiar" in prompt_l:
        required_any = ["estudio", "bloque", "entrega", "examen", "descans", "prior"]
        forbidden = ["persona con 40 años", "habilidad nueva", "usîh êh una persona", "usih es una persona"]
        forbidden.extend(["resúye", "resuye", "resúyey", "resuyey"])
        return float(sum(item in output_l for item in required_any) >= 2 and not any(item in output_l for item in forbidden))

    return float(not has_generation_artifact(output))


def repetition_rate(text: str) -> float:
    words = re.findall(r"\w+", text.lower())
    if len(words) < 3:
        return 0.0
    adjacent_repeats = sum(1 for left, right in zip(words, words[1:]) if left == right)
    scores = [adjacent_repeats / max(1, len(words) - 1)]
    for n in (2, 3, 4):
        ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        if not ngrams:
            continue
        seen: set[tuple[str, ...]] = set()
        repeated = 0
        for ngram in ngrams:
            if ngram in seen:
                repeated += 1
            seen.add(ngram)
        scores.append(repeated / len(ngrams))
    return max(scores)
