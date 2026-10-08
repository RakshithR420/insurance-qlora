"""Step 5 — Gradio demo: ask insurance questions in a browser.

Run:  python app.py      (in Colab it prints a public *.gradio.live link)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from utils import generate_answer, load_config, load_finetuned_model, load_tokenizer  # noqa: E402

import gradio as gr  # noqa: E402

cfg = load_config()
tokenizer = load_tokenizer(cfg["base_model"])
model = load_finetuned_model(cfg)
SYSTEM = cfg["data"]["system_prompt"]


def answer(question: str) -> str:
    if not question.strip():
        return "Please type a question."
    return generate_answer(model, tokenizer, question, SYSTEM, max_new_tokens=300)


demo = gr.Interface(
    fn=answer,
    inputs=gr.Textbox(lines=3, label="Your insurance question"),
    outputs=gr.Textbox(lines=10, label="Answer"),
    title="Insurance Assistant (QLoRA fine-tuned)",
    description=f"{cfg['base_model']} fine-tuned with QLoRA on InsuranceQA. Educational demo, not financial or legal advice.",
    examples=[
        "What is the difference between term and whole life insurance?",
        "Does renters insurance cover my guests if they get hurt?",
        "What does a collision deductible mean?",
    ],
)

if __name__ == "__main__":
    demo.launch(share=True)
