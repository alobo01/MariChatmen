from marichatmen.data.mask_fragile_spans import mask_and_restore_identity, mask_text, unmask_text
from marichatmen.data.transliterate_andaluh import to_andaluh


def test_mask_round_trip_fragile_spans():
    text = (
        "Instala transformers con `pip install transformers` y usa "
        "Qwen/Qwen3.5-9B en https://huggingface.co."
    )
    assert mask_and_restore_identity(text)


def test_transliteration_preserves_model_and_code_spans():
    text = "Instala transformers con `pip install transformers` y usa Qwen/Qwen3.5-9B."
    out = to_andaluh(text, informal_strength=0.0)
    assert "`pip install transformers`" in out
    assert "Qwen/Qwen3.5-9B" in out


def test_mask_unmask_explicit():
    masked = mask_text("Usa CUDA y data/processed/sft_train.jsonl.")
    assert masked.text != "Usa CUDA y data/processed/sft_train.jsonl."
    assert unmask_text(masked.text, masked.replacements) == "Usa CUDA y data/processed/sft_train.jsonl."


def test_transliteration_preserves_unquoted_identifiers_and_functions():
    text = "Llama a foo_bar() y guarda el resultado en my_variable."
    out = to_andaluh(text, informal_strength=0.0)
    assert "foo_bar()" in out
    assert "my_variable" in out
