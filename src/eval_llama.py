import os
import json
import yaml
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import PeftModel
from trl import SFTTrainer

import sys
sys.path.append("/kaggle/working/CaseStudy/src")

from format_noc import build_prompt, build_label


# Paths
TRAIN_FP = "/kaggle/input/datasets/haehyunlee/noc-stage1/data/step1_full/train.jsonl"
VALID_FP = "/kaggle/input/datasets/haehyunlee/noc-stage1/data/step1_full/valid.jsonl"


# Dataset formatter
def make_text_from_line(ex):
    row = json.loads(ex["text"])
    spec = row["spec"]
    switches = row["switches"]
    return {"text": build_prompt(spec) + build_label(switches)}


# Find last checkpoint
def find_last_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None

    ckpts = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-")
    ]

    if not ckpts:
        return None

    return sorted(ckpts, key=lambda x: int(x.split("-")[-1]))[-1]


def main(cfg_path):

    # Load config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Load dataset
    ds = load_dataset(
        "text",
        data_files={"validation": VALID_FP},
    )

    ds = ds.map(make_text_from_line, remove_columns=["text"])

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Quant config
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=cfg["load_in_4bit"],
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
    )

    # Base model
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=quant_cfg,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Load LoRA adapter
    ckpt = find_last_checkpoint(cfg["output_dir"])
    print("Using checkpoint:", ckpt)

    if ckpt is None:
        raise ValueError(f"No checkpoint found in {cfg['output_dir']}")

    model = PeftModel.from_pretrained(base, ckpt)
    model.eval()
    model.config.use_cache = True

    # Training args (minimal)
    args = TrainingArguments(
        seed=42,
        output_dir="./eval_tmp",
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        report_to="none",
        fp16=True,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        eval_dataset=ds["validation"],
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
        args=args,
    )

    # Evaluate
    metrics = trainer.evaluate()

    print("\n==== VALIDATION RESULTS ====")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # Perplexity
    if "eval_loss" in metrics:
        ppl = torch.exp(torch.tensor(metrics["eval_loss"]))
        print("Perplexity:", ppl.item())


if __name__ == "__main__":

    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/llama7b_qlora_sft.yaml")
    args = p.parse_args()

    main(args.config)
