import os
import json
import yaml
import torch
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

import sys
sys.path.append("/kaggle/working/CaseStudy/src")
from format_noc import build_prompt


# ----------------
# Paths
# ----------------
TEST_FP = "/kaggle/input/datasets/haehyunlee/noc-stage1/data/step1_full/test.jsonl"


def make_test(ex):
    row = json.loads(ex["text"])
    return {"text": build_prompt(row["spec"])}


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

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)


    # Dataset
    ds = load_dataset("text", data_files={"test": TEST_FP})
    ds = ds.map(make_test, remove_columns=["text"])


    # Tokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # IMPORTANT for batching


    # Quant
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=cfg["load_in_4bit"],
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
    )


    # Model
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=quant_cfg,
        device_map="auto",
        torch_dtype=torch.float16,
    )


    ckpt = find_last_checkpoint(cfg["output_dir"])
    print("Using:", ckpt)

    model = PeftModel.from_pretrained(base, ckpt)
    model.eval()
    model.config.use_cache = True


    # ----------------
    # Batched generate
    # ----------------
    BATCH = 16        # try 4 / 8 / 16 depending on VRAM
    MAX_NEW = 256


    preds = []

    for i in tqdm(range(0, len(ds["test"]), BATCH)):

        batch = ds["test"][i:i+BATCH]["text"]

        tok_out = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg["max_seq_length"]
        ).to(model.device)


        with torch.no_grad():

            out = model.generate(
                **tok_out,
                max_new_tokens=MAX_NEW,
                do_sample=False,        # faster
                num_beams=1,            # faster
            )


        texts = tok.batch_decode(out, skip_special_tokens=True)


        # Remove prompts
        for t, p in zip(texts, batch):
            preds.append(t[len(p):].strip())


    # Save
    with open("test_predictions.jsonl", "w") as f:
        for p in preds:
            f.write(json.dumps({"prediction": p}) + "\n")


    print("Saved: test_predictions.jsonl")


if __name__ == "__main__":

    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/llama7b.yaml")
    args = p.parse_args()

    main(args.config)
