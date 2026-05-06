from marichatmen.eval.generation_eval import strip_think_blocks


def test_strip_generation_role_leakage():
    text = "Er modelo contetta en Andalûh.\nuser Otra pregunta\nassistant Otra respuesta"
    assert strip_think_blocks(text) == "Er modelo contetta en Andalûh."


def test_strip_generation_role_leakage_after_sentence():
    text = "Er modelo contetta en Andalûh. user Otra pregunta assistant Otra respuesta"
    assert strip_think_blocks(text) == "Er modelo contetta en Andalûh."


def test_strip_generation_keeps_normal_text():
    text = "Er usuario puede preguntâh, pero la respuesta sigue clara."
    assert strip_think_blocks(text) == text
