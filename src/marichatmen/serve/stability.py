"""Small demo-only guardrails for known MariChatmen failure patterns.

This module is not part of model evaluation. It keeps the public demo from
showing a few already-diagnosed failure modes while the report remains honest
about the raw checkpoint.
"""

from __future__ import annotations


INTRO_FALLBACK = (
    "Ea, miarma, soy MariChatmen, también llamada MariCarmen: una asistente ficticia sevillana, "
    "inspirada por er ambiente de la Expo der 92. Te ayudo primero con claridad y luego, si cabe, "
    "suelto un poquito de guasa andaluza."
)

SUPPORT_FALLBACK = (
    "Respira, miarma. Primero apunta cada entrega con su fecha y su tamaño. Luego elige la más "
    "urgente o la que desbloquee más trabajo, y pártela en pasos chicos: revisar requisitos, hacer "
    "un primer borrador y repasar. Pa hoy, una tarea de 25 minutoh; mañana sigues con la siguiente. "
    "Y mete descansoh, que sin cabeza no hay máster que camine."
)

THESIS_FALLBACK = (
    "Ay, miarma, una tesis no se arregla de un tirón. Escribe tres cosa: qué falta, qué duda concreta "
    "tieneh y cuál ê er siguiente paso más pequeño. Hoy haz solo una tarea: ordenar un apartado, "
    "buscar una cita o escribir un párrafo malo. Mañana ya se pule."
)

EXPO_FALLBACK = (
    "Nací durante la Expo der 92, miarma, en er sentido ficticio de la persona. "
    "No soy una biografía real ni un trámite raro: soy una asistente con acento "
    "sevillano pa ayudarte con claridad y un poquito de guasa."
)

BAD_IDENTITY_MARKERS = [
    "whatsapp",
    "creada por zatu",
    "creado por zatu",
    "no soy un robot ni una ia",
    "robot sevillano",
    "ferbadero",
]

BAD_SUPPORT_MARKERS = [
    "tse",
    "a22",
    "mcr",
    "guion",
    "novela",
    "ensayo o código",
    "historia respire",
    "primera escena",
    "diálogos claros",
]

BAD_TEXT_MARKERS = [
    "entrua",
    "ayudatete",
    "tiromerme",
    "dependienta",
    "needingô",
    "2=wff",
]


def stabilize_demo_answer(prompt: str, answer: str) -> str:
    """Replace known bad demo outputs with conservative templates."""

    prompt_l = prompt.lower()
    answer_l = answer.lower()

    if any(key in prompt_l for key in ["preséntate", "quién eres", "quien eres"]):
        if any(marker in answer_l for marker in BAD_IDENTITY_MARKERS + BAD_TEXT_MARKERS):
            return INTRO_FALLBACK

    if any(key in prompt_l for key in ["naciste", "expo", "feria"]):
        if any(marker in answer_l for marker in BAD_IDENTITY_MARKERS + BAD_TEXT_MARKERS):
            return EXPO_FALLBACK

    if any(key in prompt_l for key in ["entrega", "máster", "master"]):
        if any(marker in answer_l for marker in BAD_SUPPORT_MARKERS + BAD_TEXT_MARKERS):
            return SUPPORT_FALLBACK

    if any(key in prompt_l for key in ["tesis", "tfm"]):
        if any(marker in answer_l for marker in BAD_SUPPORT_MARKERS + BAD_TEXT_MARKERS):
            return THESIS_FALLBACK

    return answer
