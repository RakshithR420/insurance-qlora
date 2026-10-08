"""Step 2 — QLoRA fine-tuning.

Run:  python src/train.py

What happens (simple terms):
  1. Load the base model in 4-bit  -> it fits on a 16 GB GPU.   (the "Q" in QLoRA)
  2. Freeze all original weights and add small trainable LoRA adapters. (the "LoRA")
  3. Train only the adapters on our insurance Q&A (~0.5% of parameters).
  4. Loss is computed only on the assistant's answer, not on the question.
  5. Save just the adapter (~100-200 MB) instead of the full 15 GB model.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ROOT, compute_dtype, load_base_model_4bit, load_config, load_tokenizer  # noqa: E402


def make_sft_config(cfg: dict):
    """Build SFTConfig while tolerating small API renames across TRL versions."""
    import torch
    from trl import SFTConfig

    t = cfg["training"]
    params = inspect.signature(SFTConfig.__init__).parameters
    use_bf16 = compute_dtype() == torch.bfloat16

    kwargs = dict(
        output_dir=str(ROOT / t["output_dir"]),
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        logging_steps=t["logging_steps"],
        eval_steps=t["eval_steps"],
        save_steps=t["save_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        bf16=use_bf16,
        fp16=not use_bf16,
        seed=t["seed"],
        report_to=t["report_to"],
    )
    # renamed args across TRL / transformers versions
    kwargs["max_length" if "max_length" in params else "max_seq_length"] = t["max_seq_length"]
    kwargs["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "steps"
    kwargs["save_strategy"] = "steps"
    # train only on the assistant answer (prompt tokens are masked out of the loss)
    if "completion_only_loss" in params:
        kwargs["completion_only_loss"] = True
    return SFTConfig(**{k: v for k, v in kwargs.items() if k in params})


def to_prompt_completion(example, tokenizer):
    """Split each chat into prompt (system+user) and completion (assistant answer),
    so the loss is only computed on the answer tokens."""
    msgs = example["messages"]
    prompt = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(msgs, tokenize=False)
    return {"prompt": prompt, "completion": full[len(prompt):]}


def main() -> None:
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import SFTTrainer

    cfg = load_config()
    data_dir = ROOT / cfg["data"]["processed_dir"]
    train_file, eval_file = data_dir / "train.jsonl", data_dir / "eval.jsonl"
    if not train_file.exists():
        sys.exit("No processed data found. Run: python src/prepare_data.py")

    tokenizer = load_tokenizer(cfg["base_model"])
    ds = load_dataset("json", data_files={"train": str(train_file), "eval": str(eval_file)})
    ds = ds.map(lambda ex: to_prompt_completion(ex, tokenizer), remove_columns=["messages"])

    print(f"Loading {cfg['base_model']} in 4-bit ...")
    model = load_base_model_4bit(cfg)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    lc = cfg["lora"]
    peft_config = LoraConfig(
        r=lc["r"],
        lora_alpha=lc["alpha"],
        lora_dropout=lc["dropout"],
        target_modules=lc["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer_kwargs = dict(
        model=model,
        args=make_sft_config(cfg),
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        peft_config=peft_config,
    )
    tparams = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs["processing_class" if "processing_class" in tparams else "tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.model.print_trainable_parameters()

    trainer.train()

    out = ROOT / cfg["training"]["output_dir"]
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    metrics = trainer.evaluate()
    print(f"\nFinal eval loss: {metrics.get('eval_loss'):.4f}")
    print(f"Adapter saved to: {out}")


if __name__ == "__main__":
    main()
