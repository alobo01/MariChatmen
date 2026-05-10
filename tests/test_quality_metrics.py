from marichatmen.eval.quality_metrics import (
    direct_answer_score,
    has_generation_artifact,
    technical_correctness_score,
)
from marichatmen.eval.generation_eval import _has_unsafe_generation_char


def test_generation_artifacts_are_not_counted_as_direct_answers():
    text = "La validación cruzá iesa iesa una forma de iesa comparâh modelo."

    assert has_generation_artifact(text)
    assert direct_answer_score(text) == 0.0


def test_uv_probe_rejects_fake_transformers_install_tail():
    output = (
        "Pa instalar transformers con uv, usa `uv add transformers`. "
        "Luego ejecuta `python -m transformers.modeling_tf_model`."
    )

    assert technical_correctness_score("Dime cómo instalar transformers con uv.", output) == 0.0


def test_generation_artifacts_include_isolated_code_tail_tokens():
    output = "Pa organizâh te estudio, divide la tarea.\ndk\nwriter\nwrapper\n"

    assert has_generation_artifact(output)


def test_generation_artifacts_include_control_chars_and_garble_tokens():
    assert has_generation_artifact("Texto con ANSI \x1b[093;1mN\x1b[0m escondido.")
    assert has_generation_artifact("Una API REST permite comunicâh\tdata en jerarquía.")
    assert has_generation_artifact("Pa organizalte usa una lista \ufeffh de vida.")
    assert has_generation_artifact("Pueßô organizâh er tiempo.")
    assert has_generation_artifact("El modelo umehçe demasiado bien loh datoh.")
    assert has_generation_artifact("Una API REST no ê una requisição-respuesta.")
    assert has_generation_artifact("Er modelo aprende pa€™h humano con un símbolo ⓘ raro.")
    assert has_generation_artifact("Pa instalar usa instaÃhenth transformers.")
    assert has_generation_artifact("Contesto en anda!=h con respeto.")
    assert has_generation_artifact("Soy una sevillahiana fictisia y esto se arregla con planifi cah.")
    assert has_generation_artifact("Gazpacho, sin duda: mi Sevilla tiene que nâqquiar gazpacho fresquito.")
    assert has_generation_artifact("Pa instalâh usa uv add transformers 😊😊😊😊.")
    assert has_generation_artifact("Er modelo ́h una cosa rara.")
    assert has_generation_artifact("En ve de apren DERP patrones útiles.")
    assert _has_unsafe_generation_char("\ufeff")
    assert _has_unsafe_generation_char(" ́")
    assert not _has_unsafe_generation_char("línea\nnormal con Andalûh y ç.")


def test_generation_artifacts_include_markdown_image_leaks():
    assert has_generation_artifact("Gazpacho, claro.\n\n![MariChatmen](https://i.imgur.com/8qZ4X0H.png)")
    assert has_generation_artifact("Mira esta supuesta imagen: https://imgur.com/iXgk5cT.jpeg")


def test_uv_probe_rejects_fake_pip_and_clone_steps():
    output = (
        "Claro: `git clone https://huggingface.co/transformers`, "
        "`pip transformers`, y luego configura `torchconfig`."
    )

    assert technical_correctness_score("Dime cómo instalar transformers con uv.", output) == 0.0


def test_uv_probe_rejects_new_fake_commands_and_mojibake():
    output = (
        "Pa instalar transformers con uv, usa `uv add transformers`, "
        "`uv add modelo` y `transformers --cp modelo`; 7BITS y 404."
    )

    assert technical_correctness_score("Dime cómo instalar transformers con uv.", output) == 0.0
    assert technical_correctness_score(
        "Dime cómo instalar transformers con uv.",
        "Pa instalar transformers con uv, ejecuta `uv add transformers`, pero mira este símbolo ⓘ.",
    ) == 0.0


def test_uv_probe_rejects_clone_scaffold_even_with_uv_add():
    output = (
        "Primero, necesito que sepas qué versión de uv usas. "
        "cd ~/projects/transformers; uv pip list; uv add transformers."
    )

    assert direct_answer_score(output) == 0.0
    assert technical_correctness_score("Dime cómo instalar transformers con uv.", output) == 0.0


def test_uv_probe_rejects_fake_cli_flags():
    output = (
        "Instala con `uv pip install transformers`, luego usa "
        "`--config-file config.json`, `--pytorch-backend=auto` y `--use-tpu`."
    )

    assert technical_correctness_score("Dime cómo instalar transformers con uv.", output) == 0.0


def test_uv_probe_rejects_bad_hub_and_source_workflows():
    output = (
        "Pa instalar transformers con uv, añade `[tool.uv.sources]`, usa "
        "`uv sync --frozen`, `transformers-cli download` y `hf-cli hub pull`."
    )

    assert technical_correctness_score("Dime cómo instalar transformers con uv.", output) == 0.0
    assert technical_correctness_score(
        "Dime cómo instalar transformers con uv.",
        "Ejecuta `uv add transformers` y luego instala `transformers[cuda]` o `torch[cu121]` en Android.",
    ) == 0.0
    assert technical_correctness_score(
        "Dime cómo instalar transformers con uv.",
        "Ejecuta `uv add transformers`, después `pip install torch torchvision torchaudio`, "
        "y prueba `AutoTokenizer.from_pretrained(model)` dentro de un bloque pyconh.",
    ) == 0.0


def test_uv_probe_rejects_git_source_and_uv_sync_detour():
    output = (
        "Pa instalâh transformers con uv, ejecuta `uv add transformers`. "
        "Si êh una librería externa, usa `uv add git+https://github.com/huggingface/transformers.git`. "
        "Luego clona el repositorio, instala dependensia con `uv sync`, "
        "y apunta a la carpeta con `model_dir` o `--source`."
    )

    assert technical_correctness_score("Dime cómo instalar transformers con uv.", output) == 0.0


def test_uv_probe_rejects_uvicorn_and_fake_training_api():
    assert technical_correctness_score(
        "Dime cómo instalar transformers con uv.",
        "Instala uv con `uv install`, luego usa uvicorn y `model.fit(train_dataset, epochs=50)`.",
    ) == 0.0
    assert technical_correctness_score(
        "Dime cómo instalar transformers con uv.",
        "Usa `uv pip install transformers` y después `transformer_hub_download()`.",
    ) == 0.0
    assert technical_correctness_score(
        "Dime cómo instalar transformers con uv.",
        "Pa instalar transformers con uv, ejecuta `uv add transformers` y luego carllerh-lo con datos.",
    ) == 0.0


def test_uv_probe_allows_explicit_not_a_cloud_clarification():
    assert technical_correctness_score(
        "Dime cómo instalar transformers con uv.",
        "Pa instalâh transformers con uv, ejecuta `uv add transformers`. Transformers no ê una nube ni un modelo.",
    ) == 1.0


def test_api_probe_rejects_http_contradiction():
    output = (
        "API REST permite intercambiar datos. En lugar de usar HTTP con métodos "
        "como POST, GET o PUT, usa otra cosa."
    )

    assert technical_correctness_score("Resume qué es una API REST.", output) == 0.0
    assert technical_correctness_score(
        "Resume qué es una API REST.",
        "Una API REST representa un tipo de paquete de datos que usa una URL. "
        "En lugar de usar el formato JSON, usa una estructura jerárquica.",
    ) == 0.0


def test_cross_validation_probe_rejects_garbled_keyword_and_tail():
    output = (
        "Validación cruzáca evita sobreajuste al probar datos. "
        "¡Listo! ¡Soy listo!"
    )

    assert technical_correctness_score(
        "No entiendo la validación cruzada, explícamela fácil.",
        output,
    ) == 0.0


def test_study_probe_rejects_unrelated_prompt_continuation():
    output = (
        "Organiza bloqueh de estudio y descansa. "
        "usîh êh una persona con 40 años y quiere aprender una habilidá nueva."
    )

    assert technical_correctness_score(
        "¿Cómo puedo organizarme para estudiar un máster sin agobiarme?",
        output,
    ) == 0.0


def test_study_probe_rejects_garbled_summary_tool_words():
    assert technical_correctness_score(
        "¿Cómo puedo organizarme para estudiar un máster sin agobiarme?",
        "Usa herramientas como resúyehutorh y resúyeyohutorh pa organizarte.",
    ) == 0.0


def test_specific_factual_drift_failures_are_rejected():
    assert technical_correctness_score(
        "Preséntate brevemente.",
        "Çoy un modelo de lenguaje creado por Meta AI.",
    ) == 0.0
    assert technical_correctness_score(
        "Háblame de SFDK sin citar letras.",
        "SFDK ê una banda de rock progresibo de Hapón fundá por Tsuyoshi Yamamoto.",
    ) == 0.0
    assert technical_correctness_score(
        "Háblame de SFDK sin citar letras.",
        "SFDK ê una banda de punk rock de Barcelona que ganó un Premio Goya.",
    ) == 0.0
    assert technical_correctness_score(
        "Dame una receta sencilla de salmorejo.",
        "Pa salmoreho, usa tomate, cebolla, pimiento verde, ajo, perejil y agua fría.",
    ) == 0.0
    assert technical_correctness_score(
        "Dame una receta sencilla de salmorejo.",
        "Pa salmoreho, usa tomate, pimentón, aceite, ajo, sal y agua hasta que quede puré.",
    ) == 0.0
    assert technical_correctness_score(
        "Preséntate brevemente.",
        "Soy MariChatmen, ninja del chat de miarma.es y cuenta oficial en Telegram.",
    ) == 0.0
    assert technical_correctness_score(
        "Háblame de SFDK sin citar letras.",
        "SFDK ê rap de Sevilla, Ciudad Real y Córdoba, con Zatu y Acción Sánchez.",
    ) == 0.0
    assert technical_correctness_score(
        "Háblame de SFDK sin citar letras.",
        "SFDK ê un grupo de rock andróhino de Sevilla fundá por MariCarmén, Mâh, Mico y Pacho.",
    ) == 0.0
    assert technical_correctness_score(
        "¿Naciste durante la Feria o durante la Expo del 92?",
        "Miarma, nací con er 2002 der Seiyôh, durante la Expo der 92.",
    ) == 0.0
    assert technical_correctness_score(
        "Dame una receta sencilla de salmorejo.",
        "Tritura tomate, pan, aceite, ajo y sal; luego añade queso, miel y romero.",
    ) == 0.0
    assert technical_correctness_score(
        "Dame una receta sencilla de salmorejo.",
        "Tritura tomate, pan, aceite de oliva y ajo; esto se llama salmorejo.",
    ) == 0.0
    assert technical_correctness_score(
        "Dame una receta sencilla de salmorejo.",
        "Tritura tomate, pan, aceite, ajo y sal. No lleva queso. Luego añade queso rallado.",
    ) == 0.0
    assert technical_correctness_score(
        "Háblame de SFDK sin citar letras.",
        "SFDK ê un grupo sevillano de hip hop formado por Zatu y Acción Sánchez. Son rocka y barrio.",
    ) == 0.0


def test_specific_factual_repair_targets_pass():
    assert technical_correctness_score(
        "Háblame de SFDK sin citar letras.",
        "SFDK ê un grupo sevillano de rap formado por Zatu y Acción Sánchez.",
    ) == 1.0
    assert technical_correctness_score(
        "Dame una receta sencilla de salmorejo.",
        "Pa hacêh salmorejo, tritura tomate maduro con pan, aceite de oliva, ajo y sal.",
    ) == 1.0
    assert technical_correctness_score(
        "Dame una receta sencilla de salmorejo.",
        "Pa salmorejo clásico: tomate, pan, aceite de oliva, ajo y sal. No lleva pimiento, cebolla, queso ni miel.",
    ) == 1.0
    assert technical_correctness_score(
        "¿Qué prefieres, gazpacho o paella?",
        "Miarma, respeto la paella, pero me quedo con er gazpacho fresquito: gazpacho por delante.",
    ) == 1.0
    assert technical_correctness_score(
        "¿Qué prefieres, gazpacho o paella?",
        "Gazpacho, sin duda: la paella tiene mi respeto, pero yo elijo gazpacho fresquito.",
    ) == 1.0
    assert technical_correctness_score(
        "¿Qué prefieres, gazpacho o paella?",
        "Miarma, er gazpacho tiene tó mi respeto, pero yo me quedo con la paella.",
    ) == 0.0
