# Insurance LLM — Domain Fine-tuning with QLoRA

Fine-tune an open 7B LLM (`Qwen2.5-7B-Instruct`) into an **insurance Q&A assistant** using **QLoRA**, on a single free 16 GB GPU (Colab / Kaggle T4).

## What this project shows
- **QLoRA end to end:** 4-bit NF4 quantization + LoRA adapters, so only ~0.5% of the parameters are trained.
- **Real data work:** the InsuranceQA dataset (~28k expert answers) is cleaned of tokenizer artifacts, cut down to one answer per question, and length-filtered.
- **Answer-only loss:** prompt tokens are masked, so the model learns to *answer* instead of repeating questions.
- **Measured results:** the base model and the fine-tuned model answer the same held-out questions, scored with ROUGE-L.
- **Shippable:** a CLI, a Gradio demo, and a Hugging Face Hub upload.

## How it works (simple version)
```
InsuranceQA (HF)  ──►  prepare_data.py  ──►  train / eval / test JSONL
                                                   │
Qwen2.5-7B (4-bit, frozen) + LoRA adapters  ◄──────┘  train.py
                         │
                         ├──► eval_compare.py  → base vs fine-tuned report
                         ├──► inference.py / app.py  → ask questions
                         └──► merge_and_push.py  → Hugging Face Hub
```

1. **Quantize:** load the 7B model in 4-bit. That's about 5 GB instead of about 15 GB, so it fits on a T4.
2. **Freeze + adapt:** freeze the original weights and insert small LoRA matrices (rank 16) into attention and MLP layers.
3. **Train:** only the LoRA matrices learn, using an effective batch of 16, LR 2e-4, a cosine schedule, and paged 8-bit AdamW.
4. **Save:** the output is a small adapter of about 150 MB that sits on top of the base model.

## Project structure
```
configs/config.yaml          all settings (model, data, LoRA, training) in one place
src/prepare_data.py          download + clean + split the dataset
src/train.py                 QLoRA training with TRL SFTTrainer
src/eval_compare.py          base vs fine-tuned comparison (ROUGE-L + samples)
src/inference.py             ask questions from the terminal
src/merge_and_push.py        upload adapter (or merged model) to the HF Hub
src/utils.py                 shared loading / prompting helpers
app.py                       Gradio web demo
notebooks/insurance_qlora_colab.ipynb   run everything on Colab
```

## Quick start (Colab / Kaggle)
1. Push this folder to GitHub, or zip it and upload it.
2. Open `notebooks/insurance_qlora_colab.ipynb` in Colab and set the runtime to **T4 GPU**.
3. Run the cells in order.

## Quick start (local GPU, ≥16 GB VRAM)
```bash
pip install -r requirements.txt
python src/prepare_data.py      # ~1 min
python src/train.py             # ~1.5–3 h on a T4 with defaults
python src/eval_compare.py      # base vs fine-tuned report → outputs/eval/
python src/inference.py "What is a collision deductible?"
python app.py                   # web demo
```

**Smoke test first:** set `base_model: Qwen/Qwen2.5-1.5B-Instruct` and `max_train_samples: 500` in the config. The whole pipeline then runs in minutes.

## Results
Fill this in after running `eval_compare.py` (copy from `outputs/eval/report.md`).

| Model | Avg ROUGE-L |
|---|---|
| Qwen2.5-7B-Instruct (base) | _tbd_ |
| + QLoRA insurance adapter | _tbd_ |

## Tuning tips
| Problem | Fix |
|---|---|
| CUDA out of memory | lower `per_device_train_batch_size` to 1 and raise `gradient_accumulation_steps` to 16; or lower `max_seq_length` to 768 |
| eval loss rising while train loss falls (overfitting) | fewer epochs, `lora.dropout: 0.1`, or more data |
| answers too generic | more epochs or more samples (`max_train_samples: null`) |
| training too slow | start with the 1.5B model or 2,000 samples |

## Use another domain
Change `data.hf_dataset` to any dataset with question/answer columns, update the column names in `prepare_data.py`, and rewrite `system_prompt`. Everything else stays the same.

## Data & license
- Dataset: [deccan-ai/insuranceQA-v2](https://huggingface.co/datasets/deccan-ai/insuranceQA-v2), from Feng et al., *Applying Deep Learning to Answer Selection* (IEEE ASRU 2015). Answers are mostly US-centric.
- This is an educational project and does not give financial, legal, or insurance advice.
