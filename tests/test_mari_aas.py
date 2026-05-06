from marichatmen.eval.mari_aas import score_text


def test_mari_aas_scores_andaluh_above_plain_spanish():
    spanish = "El modelo está muy cansado para responder todo con claridad."
    andaluh = "Ea, er modelo ehtá mu cansao pa reppondêh tó con claridá."
    assert score_text(andaluh, spanish).score > score_text(spanish, spanish).score


def test_mari_aas_penalizes_caricature():
    bad = "KilloOOO illo illo illo MIARMA!!!"
    result = score_text(bad)
    assert result.caricature_penalty > 0
    assert result.score < 70
