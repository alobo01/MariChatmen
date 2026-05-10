from marichatmen.eval.generation_eval import generation_eos_token_ids, trim_to_sentences


class FakeTokenizer:
    eos_token_id = 1

    def convert_tokens_to_ids(self, token: str):
        if token == "<|im_end|>":
            return 2
        return -1


class FakeTokenizerSameEos:
    eos_token_id = 2

    def convert_tokens_to_ids(self, token: str):
        if token == "<|im_end|>":
            return 2
        return -1


def test_generation_stops_on_chatml_im_end_as_well_as_eos():
    assert generation_eos_token_ids(FakeTokenizer()) == [1, 2]


def test_generation_eos_ids_are_deduplicated():
    assert generation_eos_token_ids(FakeTokenizerSameEos()) == 2


def test_trim_to_sentences_caps_and_drops_trailing_question():
    text = "Primera frase. Segunda frase. ¿Qué necesita? Cola incompleta"

    assert (
        trim_to_sentences(text, max_sentences=3, drop_trailing_question=True)
        == "Primera frase. Segunda frase."
    )


def test_trim_to_sentences_keeps_complete_sentences_only():
    text = "Una respuesta buena. Otra respuesta buena. Cola sin cerrar"

    assert trim_to_sentences(text) == "Una respuesta buena. Otra respuesta buena."
