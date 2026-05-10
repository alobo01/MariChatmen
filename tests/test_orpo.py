from marichatmen.data.build_orpo import (
    caricature_rejection,
    make_rejected,
    mild_andaluh_rejection,
    parse_args,
    standard_spanish_rejection,
)


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


def test_reasoning_preamble_rejection_contains_visible_planning():
    out = make_rejected(
        [{"role": "system", "content": "Eres un asistente"}, {"role": "user", "content": "Resume qué es una API REST."}],
        "Una API REST ê una forma de comunicâh aplicacionêh usando HTTP.",
        "reasoning_preamble",
        1,
    )
    assert "El usuario pregunta" in out


def test_wrong_answer_rejection_targets_known_technical_failure():
    out = make_rejected(
        [{"role": "system", "content": "Eres un asistente"}, {"role": "user", "content": "Dime cómo instalar transformers con uv."}],
        "Pa instalâh transformers con uv, ejecuta `uv add transformers`.",
        "correct_style_wrong_answer",
        1,
    )
    assert "nube" in out or "navegador" in out


def test_repetition_rejection_is_degenerate():
    out = make_rejected(
        [{"role": "user", "content": "No entiendo la validación cruzada."}],
        "La validación cruzá divide datoh en bloqueh pa probâh generalización.",
        "repetition_collapse",
        1,
    )
    assert "prueba, prueba" in out


def test_default_rejected_types_are_neutral_qwen_andaluh_negatives(monkeypatch):
    monkeypatch.setattr("sys.argv", ["build_orpo"])
    args = parse_args()

    assert "reasoning_preamble" in args.rejected_types
    assert "correct_style_wrong_answer" in args.rejected_types
    assert "caricature" not in args.rejected_types
    assert args.enforce_length_ratio


def test_standard_spanish_leak_prefers_preserved_original_assistant():
    original = "El overfitting ocurre cuando un modelo memoriza los datos de entrenamiento."
    out = make_rejected(
        [{"role": "system", "content": "Eres un asistente"}, {"role": "user", "content": "Explícame qué es el overfitting."}],
        "Er overfitting ê cuando un modelo memoriza loh datoh de entrenamiento.",
        "standard_spanish_leak",
        1,
        original_spanish=original,
    )

    assert out == original
