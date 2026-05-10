from marichatmen.data.build_repair_sft import _keep_row as keep_repair_row
from marichatmen.data.build_repair_sft import _limit_source_rows
from marichatmen.data.filter_cpt import reject_reason as cpt_reject_reason
from marichatmen.data.regularize_sft import _has_bad_answer_shape, _regularize


def _row(user: str, assistant: str):
    return {
        "messages": [
            {"role": "system", "content": "Eres un asistente."},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {},
    }


def test_regularized_sft_rejects_grouped_numeric_loops():
    assert _has_bad_answer_shape("Hay 300 000 000 y 200 000 000 registros repetidos.")


def test_regularized_sft_rejects_prompt_side_artifacts():
    rows = [_row("Explícame esto: iesa iesa", "Er overfitting ê memorizâh demasiao.")]

    assert _regularize(rows, seed=1, max_exact_assistant_repeats=8) == []


def test_repair_sft_rejects_grouped_numeric_loops_anywhere():
    bad = _row(
        "Resume esta tabla con 300 000 000 y 200 000 000 filas raras.",
        "Er resumen ê que la tabla tiene mucho dato.",
    )

    assert not keep_repair_row(bad, max_repetition_rate=1.0)


def test_repair_sft_rejects_mojibake_anywhere():
    bad = _row(
        "Dime cómo instalar transformers con uv.",
        "Pa instalar usa instaÃhenth transformers y luego mira er símbolo ⓘ.",
    )

    assert not keep_repair_row(bad, max_repetition_rate=1.0)


def test_repair_sft_source_limit_zero_disables_orpo_rows():
    rows = [_row("Pregunta", "Reppuêtta") for _ in range(5)]

    assert _limit_source_rows(rows, limit=0, seed=1) == []
    assert len(_limit_source_rows(rows, limit=2, seed=1)) == 2
    assert len(_limit_source_rows(rows, limit=-1, seed=1)) == 5


def test_cpt_filter_rejects_mojibake_artifacts():
    reason = cpt_reject_reason(
        "Ehte texto tiene bastante material normal pa contâh una cosa, pero incluye mojibake Ã y un símbolo ⓘ que no debe entrar en CPT.",
        min_words=1,
        max_words=100,
        max_numeric_tokens=8,
        max_percent_tokens=1,
        max_unique_years=3,
        max_numeric_char_fraction=0.2,
        max_repetition_rate=1.0,
    )

    assert reason == "generation_artifact"
