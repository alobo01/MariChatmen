"""Simple local CLI chat for a base model plus optional adapter."""

from __future__ import annotations

import argparse

from marichatmen.constants import SYSTEM_PROMPT_BASE
from marichatmen.eval.generation_eval import generate_response, load_causal_model, load_tokenizer
from marichatmen.serve.postprocess import render_andaluh_demo
from marichatmen.serve.stability import stabilize_demo_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--tokenizer_name", default="")
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--system_prompt", default=SYSTEM_PROMPT_BASE)
    parser.add_argument("--max_new_tokens", type=int, default=72)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--trim_to_sentence", action="store_true")
    parser.add_argument("--max_sentences", type=int, default=3)
    parser.add_argument("--drop_trailing_question", action="store_true")
    parser.add_argument(
        "--postprocess_andaluh",
        action="store_true",
        help="Render generated text through the protected Andaluh demo postprocessor.",
    )
    parser.add_argument(
        "--stabilize_demo",
        action="store_true",
        help="Apply demo-only fallbacks for known diagnosed failure patterns.",
    )
    parser.add_argument("--no_4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(
        args.model_name,
        args.tokenizer_name or None,
        args.adapter_path or None,
    )
    model = load_causal_model(
        args.model_name,
        args.adapter_path or None,
        load_in_4bit=not args.no_4bit,
        tokenizer_len=len(tokenizer),
        tokenizer=tokenizer,
        tokenizer_name=args.tokenizer_name or args.model_name,
    )
    messages = [{"role": "system", "content": args.system_prompt}]
    print("MariChatmen/Qwen-Andaluh CLI. Ctrl-D to exit.")
    while True:
        try:
            user = input("User> ").strip()
        except EOFError:
            print()
            return
        if not user:
            continue
        messages.append({"role": "user", "content": user})
        answer = generate_response(
            model,
            tokenizer,
            messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            trim_to_sentence=args.trim_to_sentence,
            max_sentences=args.max_sentences,
            drop_trailing_question=args.drop_trailing_question,
        )
        if args.stabilize_demo:
            answer = stabilize_demo_answer(user, answer)
        if args.postprocess_andaluh:
            answer = render_andaluh_demo(answer)
        messages.append({"role": "assistant", "content": answer})
        print(f"MariChatmen> {answer}")


if __name__ == "__main__":
    main()
