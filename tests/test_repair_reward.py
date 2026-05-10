from marichatmen.eval.repair_reward import score_repair_answer


def test_repair_reward_prefers_direct_correct_andaluh_answer():
    prompt = "Explícame qué es el overfitting."
    good = score_repair_answer(
        prompt,
        "Er overfitting ê cuando un modelo memoriza demasiao loh datoh de "
        "entrenamiento y luego no generaliza bien con datoh nuevoh.",
    )
    preamble = score_repair_answer(
        prompt,
        "El usuario pregunta qué es el overfitting, así que voy a explicar el concepto.",
    )
    wrong_style = score_repair_answer(
        "Dime cómo instalar transformers con uv.",
        "Pa instalar transformers con uv, abre una nube y busca una imagen llamada transformers.",
    )

    assert good.reward > 0.7
    assert preamble.reward < 0.0
    assert wrong_style.reward < good.reward
    assert preamble.reasoning_preamble == 1.0
    assert wrong_style.correctness == 0.0


def test_repair_reward_penalizes_role_leakage_and_repetition():
    scored = score_repair_answer("Resume qué es una API REST.", "assistant: api api api api api")

    assert scored.gibberish > 0.3
    assert scored.repetition > 0.5
    assert scored.reward < 0.0
