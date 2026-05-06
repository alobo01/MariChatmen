"""Simple local CLI chat for a base model plus optional adapter."""

from __future__ import annotations

import argparse

from marichatmen.constants import SYSTEM_PROMPT_INFERENCE
from marichatmen.eval.generation_eval import generate_response, load_causal_model, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--no_4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.model_name)
    model = load_causal_model(args.model_name, args.adapter_path or None, load_in_4bit=not args.no_4bit)
    messages = [{"role": "system", "content": SYSTEM_PROMPT_INFERENCE}]
    print("MariChatmen CLI. Ctrl-D to exit.")
    while True:
        try:
            user = input("User> ").strip()
        except EOFError:
            print()
            return
        if not user:
            continue
        messages.append({"role": "user", "content": user})
        answer = generate_response(model, tokenizer, messages, max_new_tokens=args.max_new_tokens)
        messages.append({"role": "assistant", "content": answer})
        print(f"MariChatmen> {answer}")


if __name__ == "__main__":
    main()
