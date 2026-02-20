import os
import json
import yaml
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

import sys
sys.path.append("/kaggle/working/CaseStudy/src")
from format_noc import build_prompt  # only need prompt for testing

# -----------------------------
# Paths
# -----------------------------
TEST_FP = "/kaggle/input/datasets/haehyunlee/noc-stage1/data/step1_full/test.jsonl"

# -----------------------------
# Dataset formatter
# -----------------------------
def make_test_text(ex):
    row = json.loads(ex["text"])
    spec = row["spec"]
    return {"text": build_prompt(spec)}

# -----------------------------
# Find last checkpoint
# -----------------------------
def find_last_checkpoint(output_dir):
    if not os.path.exists(output_dir):
        return None

    ckpts = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]

    if not ckpts:
        return None

    # Sort by step number
    ckpts = sorted(ckpts, key=lambda x: int(x.split("-")[-1]))
    return ckpts[-1]

# -----------------------------
# Main testing function
# -----------------------------
def main(cfg_path):
    # Load config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Load dataset
    ds = load_dataset("text", data_files={"test": TEST_FP})
    ds = ds.map(make_test_text, remove_columns=["text"])

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Quantization config
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=cfg["load_in_4bit"],
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
    )

    # Load base model
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=quant_cfg,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Load LoRA adapter
    ckpt = find_last_checkpoint(cfg["output_dir"])
    if ckpt is None:
        raise RuntimeError(f"No checkpoints found in {cfg['output_dir']}")
    print("Using checkpoint:", ckpt)

    model = PeftModel.from_pretrained(base, ckpt)
    model.eval()
    model.config.use_cache = True

    # Generate predictions
    predictions = []
    for example in ds["test"]:
        inp = tok(example["text"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=256)
            txt = tok.decode(out[0], skip_special_tokens=True)
            # Remove the prompt from output
            predictions.append(txt[len(example["text"]):].strip())

    # Save predictions
    out_file = "test_predictions.jsonl"
    with open(out_file, "w") as f:
        for pred in predictions:
            f.write(json.dumps({"prediction": pred}) + "\n")

    print(f"Predictions saved to {out_file}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/llama7b.yaml")
    args = p.parse_args()

    main(args.config)
