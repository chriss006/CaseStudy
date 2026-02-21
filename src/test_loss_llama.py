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


# ----------------
# Paths
# ----------------
TEST_FP = "/kaggle/input/datasets/haehyunlee/noc-stage1/data/step1_full/test.jsonl"


# ----------------
# Format
# ----------------
def make_text(ex):

    row = json.loads(ex["text"])

    spec = row["spec"]
    switches = row["switches"]

    return {
        "text": build_prompt(spec) + build_label(switches)
    }


# ----------------
# Find ckpt
# ----------------
def find_last_checkpoint(out_dir):

    ckpts = []

    for d in os.listdir(out_dir):
        if d.startswith("checkpoint-"):
            step = int(d.split("-")[-1])
            ckpts.append((step, os.path.join(out_dir, d)))

    if not ckpts:
        return None

    ckpts.sort()
    return ckpts[-1][1]


# ----------------
# Main
# ----------------
def main(cfg_path):

    # Config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)


    # Dataset
    ds = load_dataset("text", data_files={"test": TEST_FP})
    ds = ds.map(make_text, remove_columns=["text"])


    # Tokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"


    # Quant
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


    # Load adapter
    ckpt = find_last_checkpoint(cfg["output_dir"])

    if ckpt is None:
        raise RuntimeError("No checkpoint found")

    print("Using:", ckpt)

    model = PeftModel.from_pretrained(base, ckpt)
    model.eval()


    # Trainer args
    args = TrainingArguments(
        output_dir="./test_eval",
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        fp16=True,
        report_to="none",
    )


    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        eval_dataset=ds["test"],
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
        args=args,
    )


    # Evaluate
    metrics = trainer.evaluate()


    print("\n==== TEST RESULTS ====")

    for k, v in metrics.items():
        print(f"{k}: {v}")


    if "eval_loss" in metrics:

        ppl = torch.exp(torch.tensor(metrics["eval_loss"]))
        print("Perplexity:", ppl.item())


# ----------------
if __name__ == "__main__":

    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/llama7b_qlora_sft.yaml")
    args = p.parse_args()

    main(args.config)
