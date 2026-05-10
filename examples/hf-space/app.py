import os
import sys
from pathlib import Path

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if (Path(__file__).parent / "src").exists():
    sys.path.insert(0, str(Path(__file__).parent / "src"))
elif Path("src").exists():
    sys.path.insert(0, "src")

from marichatmen.serve.postprocess import render_andaluh_demo
from marichatmen.serve.stability import stabilize_demo_answer


MODEL_ID = os.environ.get("MCM_MODEL_ID", "Qwen/Qwen3.5-4B-Base")
ADAPTER_ID = os.environ.get("MCM_ADAPTER_ID", "")
SYSTEM_PROMPT = os.environ.get(
    "MCM_SYSTEM_PROMPT",
    "Eres MariChatmen, también llamada MariCarmen: una sevillana ficticia nacida durante la Expo del 92. "
    "Responde con claridad, en Andalûh informal, y prioriza la respuesta útil antes que el chascarrillo. "
    "No inventes trámites, biografía ni datos concretos que el usuario no haya dado.",
)


def _load():
    tokenizer_source = ADAPTER_ID or MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=True,
    )
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    if ADAPTER_ID:
        model = PeftModel.from_pretrained(model, ADAPTER_ID)
    model.eval()
    return model, tokenizer


MODEL, TOKENIZER = _load()


def _trim_sentences(text: str, max_sentences: int = 3) -> str:
    import re

    matches = list(re.finditer(r"[.!?](?:\s|$)", text.strip()))
    if not matches:
        return text.strip()
    return text.strip()[: matches[:max_sentences][-1].end()].strip()


def chat(message, history, stabilize=True, render_andaluh=True, temperature=0.25, max_new_tokens=128):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user, assistant in history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": message})
    prompt = TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)
    with torch.no_grad():
        ids = MODEL.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=max(float(temperature), 1e-5),
            top_p=0.9,
            max_new_tokens=int(max_new_tokens),
            pad_token_id=TOKENIZER.eos_token_id,
        )
    answer = TOKENIZER.decode(ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    answer = _trim_sentences(answer, max_sentences=3)
    if stabilize:
        answer = stabilize_demo_answer(message, answer)
    if render_andaluh:
        answer = render_andaluh_demo(answer)
    return answer


with gr.Blocks(title="MariChatmen Demo") as demo:
    gr.Markdown(
        """
        # MariChatmen demo

        Experimental checkpoint. The demo uses a short-answer setup plus an
        optional Andalûh rendering layer. The renderer protects code, URLs and
        package names while displaying ordinary Spanish text in the target style.
        """
    )
    stabilize = gr.Checkbox(value=True, label="Apply demo stability guardrails")
    render = gr.Checkbox(value=True, label="Apply Andalûh renderer after generation")
    temp = gr.Slider(0.0, 1.0, value=0.25, step=0.05, label="Temperature")
    max_tokens = gr.Slider(32, 256, value=128, step=16, label="Max new tokens")
    gr.ChatInterface(
        fn=chat,
        additional_inputs=[stabilize, render, temp, max_tokens],
    )


if __name__ == "__main__":
    demo.launch()
