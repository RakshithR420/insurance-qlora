"""Ask the fine-tuned model a question from the command line.

Run:  python src/inference.py "Does homeowners insurance cover a burst pipe?"
      python src/inference.py            # interactive mode
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import generate_answer, load_config, load_finetuned_model, load_tokenizer  # noqa: E402


def main() -> None:
    cfg = load_config()
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_finetuned_model(cfg)
    sys_prompt = cfg["data"]["system_prompt"]

    if len(sys.argv) > 1:
        print(generate_answer(model, tokenizer, " ".join(sys.argv[1:]), sys_prompt))
        return

    print("Insurance assistant ready. Type a question (or 'quit').")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"quit", "exit", ""}:
            break
        print(f"\nAssistant: {generate_answer(model, tokenizer, q, sys_prompt)}")


if __name__ == "__main__":
    main()
