"""Step 3 — Compare the base model vs. the fine-tuned model on held-out questions.

Run:  python src/eval_compare.py

What it does (simple terms):
  1. Takes N test questions the model never saw during training.
  2. Asks the BASE model and the FINE-TUNED model the same questions.
  3. Scores each answer against the expert reference with ROUGE-L
     (word-overlap similarity, 0 to 1 — higher means closer to the expert).
  4. Saves a side-by-side CSV and a Markdown report you can put in your README.
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ROOT, generate_answer, load_base_model_4bit, load_config, load_finetuned_model, load_tokenizer  # noqa: E402


def run_model(model, tokenizer, questions, system_prompt, max_new_tokens, label):
    answers = []
    for i, q in enumerate(questions, 1):
        answers.append(generate_answer(model, tokenizer, q, system_prompt, max_new_tokens))
        print(f"  [{label}] {i}/{len(questions)}", end="\r")
    print()
    return answers


def free(model):
    import torch

    del model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    import pandas as pd
    from rouge_score import rouge_scorer

    cfg = load_config()
    e = cfg["eval"]
    sys_prompt = cfg["data"]["system_prompt"]
    test_file = ROOT / cfg["data"]["processed_dir"] / "test.jsonl"
    rows = [json.loads(l) for l in open(test_file, encoding="utf-8")][: e["num_samples"]]
    questions = [r["question"] for r in rows]
    refs = [r["answer"] for r in rows]

    tokenizer = load_tokenizer(cfg["base_model"])
    tokenizer.padding_side = "left"

    print("Generating with BASE model ...")
    base = load_base_model_4bit(cfg)
    base_ans = run_model(base, tokenizer, questions, sys_prompt, e["max_new_tokens"], "base")
    free(base)

    print("Generating with FINE-TUNED model ...")
    ft = load_finetuned_model(cfg)
    ft_ans = run_model(ft, tokenizer, questions, sys_prompt, e["max_new_tokens"], "fine-tuned")
    free(ft)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    base_scores = [scorer.score(r, a)["rougeL"].fmeasure for r, a in zip(refs, base_ans)]
    ft_scores = [scorer.score(r, a)["rougeL"].fmeasure for r, a in zip(refs, ft_ans)]

    out_dir = ROOT / e["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "question": questions,
        "reference": refs,
        "base_answer": base_ans,
        "finetuned_answer": ft_ans,
        "base_rougeL": base_scores,
        "finetuned_rougeL": ft_scores,
    })
    df.to_csv(out_dir / "comparison.csv", index=False)

    b, f = df["base_rougeL"].mean(), df["finetuned_rougeL"].mean()
    wins = int((df["finetuned_rougeL"] > df["base_rougeL"]).sum())
    report = [
        "# Evaluation: base vs. fine-tuned",
        "",
        f"- Test questions: **{len(df)}** (never seen in training)",
        f"- Base model: `{cfg['base_model']}`",
        "",
        "| Model | Avg ROUGE-L |",
        "|---|---|",
        f"| Base | {b:.4f} |",
        f"| Fine-tuned (QLoRA) | {f:.4f} |",
        "",
        f"- Relative improvement: **{(f - b) / max(b, 1e-9) * 100:+.1f}%**",
        f"- Fine-tuned scored higher on **{wins}/{len(df)}** questions",
        "",
        "## Sample answers",
    ]
    for _, r in df.head(3).iterrows():
        report += [
            "", f"**Q:** {r.question}", "",
            f"**Base:** {r.base_answer[:600]}", "",
            f"**Fine-tuned:** {r.finetuned_answer[:600]}", "",
            "---",
        ]
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")

    print(f"\nAvg ROUGE-L  base={b:.4f}  fine-tuned={f:.4f}  ({wins}/{len(df)} wins)")
    print(f"Saved: {out_dir / 'comparison.csv'} and {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
