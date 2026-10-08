"""Step 4 (optional) — Publish your work to the Hugging Face Hub.

Run:
  huggingface-cli login                       # once
  python src/merge_and_push.py                # push adapter only (small, recommended)
  python src/merge_and_push.py --merge        # also merge into a full fp16 model (needs ~30 GB RAM)

Adapter-only is what most people share: anyone can load it on top of the base model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ROOT, load_config, load_tokenizer  # noqa: E402


def main() -> None:
    import torch
    from peft import AutoPeftModelForCausalLM, PeftModel
    from transformers import AutoModelForCausalLM

    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true", help="merge adapter into base weights")
    ap.add_argument("--repo", default=None, help="override hub repo id")
    args = ap.parse_args()

    cfg = load_config()
    adapter_dir = ROOT / cfg["training"]["output_dir"]
    repo = args.repo or cfg["hub"]["adapter_repo"]
    tokenizer = load_tokenizer(cfg["base_model"])

    if not args.merge:
        model = AutoPeftModelForCausalLM.from_pretrained(str(adapter_dir), device_map="cpu")
        model.push_to_hub(repo)
        tokenizer.push_to_hub(repo)
        print(f"Adapter pushed: https://huggingface.co/{repo}")
        return

    # Merge needs the base model in full precision (not 4-bit).
    base = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=torch.float16, device_map="cpu")
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    out = ROOT / "outputs" / "merged-model"
    merged.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))
    merged.push_to_hub(repo + "-merged")
    tokenizer.push_to_hub(repo + "-merged")
    print(f"Merged model pushed: https://huggingface.co/{repo}-merged")


if __name__ == "__main__":
    main()
