from marichatmen.data.build_orpo import caricature_rejection, mild_andaluh_rejection, standard_spanish_rejection


def test_standard_spanish_rejection_removes_common_andaluh_forms():
    out = standard_spanish_rejection("Ea, er modelo ehtá mu apañao pa contestâh.")
    assert " er " not in f" {out} "
    assert " mu " not in f" {out} "
    assert "muy" in out
    assert "Ea" not in out


def test_mild_andaluh_rejection_is_less_marked():
    out = mild_andaluh_rejection("Miarma, er modelo çabe contestâh mu bien.")
    assert "Miarma" not in out
    assert "ç" not in out


def test_caricature_rejection_is_obvious_negative():
    out = caricature_rejection([{"role": "user", "content": "Explícame una red neuronal."}])
    assert "KilloOOO" in out
    assert "illo illo" in out
