from marichatmen.data.build_cpt import _render_training_text


def test_cpt_default_plain_rendering_has_no_chat_template_tokens():
    messages = [
        {"role": "system", "content": "Eres un asistente"},
        {"role": "user", "content": "Explícame qué es el overfitting."},
        {"role": "assistant", "content": "Er overfitting ê cuando memoriza demasiado."},
    ]

    text = _render_training_text(messages, render_mode="plain")

    assert "<|im_start|>" not in text
    assert "Eres un asistente" not in text
    assert "Explícame" in text
    assert "Er overfitting" in text


def test_cpt_chat_rendering_is_available_for_legacy_debugging():
    messages = [
        {"role": "system", "content": "Eres un asistente"},
        {"role": "user", "content": "Hola"},
        {"role": "assistant", "content": "Buenas"},
    ]

    text = _render_training_text(messages, render_mode="chat")

    assert "<|im_start|>system" in text
    assert "<|im_start|>assistant" in text
