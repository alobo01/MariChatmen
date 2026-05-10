from marichatmen.eval.qwen_gate import needs_repair, pass_thresholds


class Args:
    min_samples = 5
    min_aas = 65.0
    min_clear_andaluh = 0.70
    max_spanish_leak = 0.15
    max_persona_leak = 0.0
    gate_persona_leak = False
    max_reasoning_preamble = 0.05
    max_generation_artifact = 0.0
    min_technical_correctness = 0.70
    min_direct_answer = 0.90
    repair_reasoning_preamble = 0.03
    repair_generation_artifact = 0.0
    repair_spanish_leak = 0.07
    repair_repetition = 0.03
    repair_direct_answer = 0.90


def test_persona_leak_is_diagnostic_unless_explicitly_gated():
    summary = {
        "n_samples": 5.0,
        "mari_aas_mean": 80.0,
        "clear_andaluh_rate": 0.90,
        "spanish_leak_mean": 0.02,
        "persona_leak_rate": 1.0,
        "reasoning_preamble_rate": 0.0,
        "generation_artifact_rate": 0.0,
        "technical_correctness_rate": 1.0,
        "direct_answer_rate": 1.0,
        "repetition_rate": 0.0,
    }

    passed, failures = pass_thresholds(summary, Args())

    assert passed
    assert not failures


def test_repair_gate_ignores_persona_leak_for_neutral_qwen_andaluh():
    summary = {
        "n_samples": 5.0,
        "mari_aas_mean": 80.0,
        "clear_andaluh_rate": 0.90,
        "spanish_leak_mean": 0.02,
        "persona_leak_rate": 1.0,
        "reasoning_preamble_rate": 0.10,
        "generation_artifact_rate": 0.0,
        "technical_correctness_rate": 1.0,
        "direct_answer_rate": 0.80,
        "repetition_rate": 0.0,
    }

    assert needs_repair(summary, Args())
