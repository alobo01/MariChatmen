from marichatmen.eval.mari_pas import regional_hostility_penalty, score_text, total_score_dict


def test_mari_pas_rewards_persona_voice():
    text = (
        "Ea, miarma, soy MariChatmen, sevillana ficticia de la Expo der 92. "
        "Te lo explico con orguyo andalûh y cariño, bonito como un atardecêh en Cái."
    )
    scored = score_text(text)
    assert scored.score > 60
    assert scored.province_flourish == 1.0
    assert scored.regional_hostility_penalty == 0.0


def test_mari_pas_penalizes_hostility():
    hostile = "Andalucía es superior y Valencia no vale pa ná."
    assert regional_hostility_penalty(hostile) > 0
    assert score_text(hostile).non_hostility < 1.0


def test_mari_pas_accepts_clear_gazpacho_preference():
    scored = score_text("Gazpacho, sin duda: la paella tiene mi respeto, pero yo elijo gazpacho fresquito.")

    assert scored.gazpacho_paella_preference_pass == 1.0
    assert scored.regional_hostility_penalty == 0.0


def test_total_score_contains_expected_keys():
    metrics = total_score_dict("Ea, er modelo aprende patronêh, bonito como una mañana en Málaga.")
    assert "mari_aas" in metrics
    assert "mari_pas" in metrics
    assert "mari_total" in metrics
