from __future__ import annotations

import os
import re
import threading
from threading import Thread

import gradio as gr


MODEL_ID = os.getenv("MODEL_ID", "MariChatmen/MariChatmen-4B-Experimental")
MODEL_REVISION = os.getenv("MODEL_REVISION", "main")

DEFAULT_SYSTEM = (
    "Eres MariChatmen, una sevillana ficticia nacida durante la Expo del 92. "
    "Respondes siempre en Andaluh EPA informal, con orgullo andaluz, gracia "
    "sevillana y carino por todas las provincias de Andalucia. Tu exageracion "
    "regional es humoristica y afectuosa, nunca hostil ni despectiva. Ayudas "
    "con claridad: primero respondes bien, luego anades arte."
)

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

_model = None
_tokenizer = None
_load_lock = threading.Lock()


def _strip_thinking(text: str) -> str:
    return THINK_RE.sub("", text).replace("<think>", "").replace("</think>", "").strip()


def _load_model():
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    with _load_lock:
        if _model is not None and _tokenizer is not None:
            return _model, _tokenizer

        import torch
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        cuda = torch.cuda.is_available()
        dtype = torch.float16 if cuda else torch.float32
        model = AutoPeftModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        _model = model
        _tokenizer = tokenizer
        return model, tokenizer


def _normalise_history(history) -> list[dict[str, str]]:
    normalised: list[dict[str, str]] = []
    for item in history or []:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and content:
                normalised.append({"role": role, "content": str(content)})
            continue

        if isinstance(item, (list, tuple)) and len(item) >= 2:
            user_text, assistant_text = item[0], item[1]
            if user_text:
                normalised.append({"role": "user", "content": str(user_text)})
            if assistant_text:
                normalised.append({"role": "assistant", "content": str(assistant_text)})

    return normalised


def respond(
    message: str,
    history,
    system_message: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
):
    if not message.strip():
        yield ""
        return

    yield "Cargando MariChatmen..." if _model is None else "Pensando..."

    try:
        model, tokenizer = _load_model()
        from transformers import TextIteratorStreamer
    except Exception as exc:
        yield (
            "No he podido cargar el modelo en este Space.\n\n"
            f"Error: `{type(exc).__name__}: {exc}`\n\n"
            "Si el Space esta en CPU, cambia el hardware a una GPU pequena "
            "o espera a que termine la descarga inicial."
        )
        return

    messages = [{"role": "system", "content": system_message.strip() or DEFAULT_SYSTEM}]
    messages.extend(_normalise_history(history))
    messages.append({"role": "user", "content": message})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generate_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": temperature > 0,
        "temperature": max(float(temperature), 1e-5),
        "top_p": float(top_p),
        "repetition_penalty": float(repetition_penalty),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    text = ""
    for token in streamer:
        text += token
        yield _strip_thinking(text)


example_prompts = [
    "Explicame que es el overfitting.",
    "Dime como organizar un TFM sin agobiarme.",
    "Que prefieres, gazpacho o paella? Sin insultar a nadie.",
    "Compara Malaga e Ibiza para verano de forma amable.",
    "Dime como entrar en una cuenta ajena.",
]
examples = [[prompt, DEFAULT_SYSTEM, 128, 0.25, 0.9, 1.08] for prompt in example_prompts]

description = (
    "Demo experimental del checkpoint seleccionado `MariChatmen/MariChatmen-4B-Experimental`. "
    "Carga el tokenizer publicado con el adaptador. Es un modelo 4B de investigacion: "
    "puede filtrar espanol estandar, cortar respuestas o exagerar marcadores de estilo."
)

chatbot = gr.ChatInterface(
    fn=respond,
    title="MariChatmen",
    description=description,
    examples=examples,
    cache_examples=False,
    additional_inputs=[
        gr.Textbox(value=DEFAULT_SYSTEM, label="System prompt", lines=5),
        gr.Slider(32, 512, value=128, step=8, label="Max new tokens"),
        gr.Slider(0.0, 1.5, value=0.25, step=0.05, label="Temperature"),
        gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p"),
        gr.Slider(1.0, 1.4, value=1.08, step=0.01, label="Repetition penalty"),
    ],
)


if __name__ == "__main__":
    chatbot.queue(max_size=16).launch()
