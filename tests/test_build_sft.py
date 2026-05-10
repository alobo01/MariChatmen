from marichatmen.constants import SYSTEM_PROMPT_BASE
from marichatmen.data.build_sft import _normalize_messages
from marichatmen.data.regularize_sft import _rebalance_user_language


def test_sft_preserves_original_assistant_for_orpo_standard_spanish_negative():
    row = {
        "messages": [
            {"role": "user", "content": "Explícame qué es el overfitting."},
            {
                "role": "assistant",
                "content": "El overfitting ocurre cuando un modelo memoriza los datos de entrenamiento.",
            },
        ],
        "source_data": "fixture",
        "source_license": "apache-2.0",
    }

    out = _normalize_messages(
        row,
        system_prompt=SYSTEM_PROMPT_BASE,
        neutral_system_prompt=SYSTEM_PROMPT_BASE,
        explicit_system_prompt="Eres un asistente que responde en Andalûh.",
        neutral_system_ratio=1.0,
        empty_system_ratio=0.0,
        user_andaluh_ratio=0.0,
        assistant_andaluh_ratio=1.0,
        variant="sevillian_ce",
        informal_strength=0.0,
        seed=1,
        row_index=0,
    )

    assert out is not None
    assert out["metadata"]["original_assistant"].startswith("El overfitting ocurre")
    assistant = out["messages"][-1]["content"]
    assert assistant != out["metadata"]["original_assistant"]


def test_regularize_rebalances_user_andaluh_after_technical_gold_injection():
    rows = [
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_BASE},
                {"role": "user", "content": f"Explícame el concepto {index}."},
                {"role": "assistant", "content": "Er concepto ê una idea importante."},
            ],
            "metadata": {"user_language": "spanish", "user_turns": 1, "user_andaluh_turns": 0},
        }
        for index in range(10)
    ]

    rebalanced, audit = _rebalance_user_language(
        rows,
        target_andaluh_ratio=0.3,
        seed=123,
        variant="sevillian_ce",
        informal_strength=0.0,
    )

    assert audit["final_andaluh_rows"] == 3
    assert sum(row["metadata"]["user_language"] == "andaluh" for row in rebalanced) == 3
