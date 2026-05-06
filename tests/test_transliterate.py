from marichatmen.data.transliterate_andaluh import epa, strip_thinking, to_andaluh


def test_strip_thinking_blocks():
    assert strip_thinking("<think>hidden</think>Respuesta") == "Respuesta"


def test_to_andaluh_changes_standard_spanish():
    out = to_andaluh("El modelo está muy cansado para todo.", informal_strength=1.0, seed=1)
    assert out != "El modelo está muy cansado para todo."
    assert "<think>" not in out


def test_default_variant_prefers_ce_style_sevillian_output():
    out = epa("Sevilla sabe hacer cosas bonitas.")
    assert "Çebiya" in out
    assert "çabe" in out


def test_plain_seseo_variant_remains_available():
    out = epa("Sevilla sabe hacer cosas bonitas.", variant="seseo")
    assert "Sebiya" in out
    assert "çabe" not in out
