"""Step 1 — Download and clean InsuranceQA, then save chat-format JSONL files.

Run:  python src/prepare_data.py

What it does (simple terms):
  1. Downloads the dataset from Hugging Face (question -> expert answer).
  2. Fixes tokenizer artifacts like "-LRB-" and "do n't" so text reads naturally.
  3. Keeps ONE good answer per question (many questions have several answers).
  4. Drops answers that are too short or too long.
  5. Saves train / eval / test files in the chat "messages" format.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ROOT, build_messages, load_config  # noqa: E402

# ---------- text cleaning ----------
_PTB = {
    "-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]",
    "-LCB-": "{", "-RCB-": "}", "``": '"', "''": '"',
}


def clean_text(text: str) -> str:
    for k, v in _PTB.items():
        text = text.replace(k, v)
    # contractions split by the tokenizer: "do n't" -> "don't", "it 's" -> "it's"
    text = re.sub(r"\s+n't\b", "n't", text)
    text = re.sub(r"\s+'(s|re|ve|ll|d|m)\b", r"'\1", text)
    # space before punctuation: "word ," -> "word,"
    text = re.sub(r"\s+([,.;:!?%)\]}])", r"\1", text)
    text = re.sub(r"([(\[{$])\s+", r"\1", text)
    text = re.sub(r'"\s+(.*?)\s+"', r'"\1"', text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(text.split())


def pick_one_answer_per_question(rows, min_w: int, max_w: int) -> list[dict]:
    """Group answers by question and keep the longest answer inside the length limits.
    Longer answers in this dataset tend to be more complete explanations."""
    best: dict[str, str] = {}
    for r in rows:
        q = clean_text(r["input"])
        a = clean_text(r["output"])
        n = word_count(a)
        if not (min_w <= n <= max_w):
            continue
        if "WEBSITELINK" in a:  # placeholder links add noise
            continue
        if q not in best or n > word_count(best[q]):
            best[q] = a
    return [{"question": q, "answer": a} for q, a in best.items()]


def save_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main() -> None:
    from datasets import load_dataset

    cfg = load_config()
    d = cfg["data"]
    random.seed(cfg["training"]["seed"])
    out_dir = ROOT / d["processed_dir"]

    print(f"Downloading {d['hf_dataset']} ...")
    ds = load_dataset(d["hf_dataset"])
    print(ds)

    split_map = {"train": "train", "eval": "validation", "test": "test"}
    limits = {
        "train": d["max_train_samples"],
        "eval": d["max_eval_samples"],
        "test": d["max_test_samples"],
    }

    for name, hf_split in split_map.items():
        pairs = pick_one_answer_per_question(ds[hf_split], d["min_answer_words"], d["max_answer_words"])
        random.shuffle(pairs)
        if limits[name]:
            pairs = pairs[: limits[name]]

        if name == "test":
            # test keeps plain question/reference so evaluation can compare answers
            save_jsonl(out_dir / "test.jsonl", pairs)
        else:
            chat = [{"messages": build_messages(p["question"], d["system_prompt"], p["answer"])} for p in pairs]
            save_jsonl(out_dir / f"{name}.jsonl", chat)
        print(f"  {name:5s}: {len(pairs):5d} examples -> {out_dir / (name + '.jsonl')}")

    print("\nSample cleaned example:")
    print(json.dumps(pairs[0], indent=2)[:800])


if __name__ == "__main__":
    main()
