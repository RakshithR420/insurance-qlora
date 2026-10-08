"""Shared helpers: config loading, model loading, prompt building."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | os.PathLike = ROOT / "configs" / "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_messages(question: str, system_prompt: str, answer: str | None = None) -> list[dict]:
    """Chat-format example. The tokenizer's chat template turns this into the
    exact prompt format the base model was trained on."""
    msgs = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": question.strip()},
    ]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer.strip()})
    return msgs


def compute_dtype():
    """T4 GPUs don't support bfloat16, so fall back to float16 there."""
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def bnb_config(cfg: dict):
    from transformers import BitsAndBytesConfig

    q = cfg["quantization"]
    return BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype(),
    )


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_base_model_4bit(cfg: dict):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb_config(cfg),
        device_map="auto",
        torch_dtype=compute_dtype(),
        trust_remote_code=True,
    )


def load_finetuned_model(cfg: dict, adapter_path: str | None = None):
    """Base model in 4-bit + trained LoRA adapter on top."""
    from peft import PeftModel

    adapter_path = adapter_path or cfg["training"]["output_dir"]
    base = load_base_model_4bit(cfg)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model


def generate_answer(model, tokenizer, question: str, system_prompt: str, max_new_tokens: int = 256) -> str:
    import torch

    msgs = build_messages(question, system_prompt)
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
