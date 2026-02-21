import os
import json
import yaml
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from torch.utils.data import DataLoader

import sys
sys.path.append("/kaggle/working/CaseStudy/src")
from format_noc import build_prompt


TEST_FP = "/kaggle/input/datasets/haehyunlee/noc-stage1/data/step1_full/test.jsonl"


def make_test_text(ex):
    row = json.loads(ex["text"])
    return {"text": build_prompt(row["spec"])}


def find_last_checkpoint(output_dir):
    ckpts = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-")
    ]
    return sorted(ckpts, key=lambda x: int(x.split("-")[-1]))[-1]


def main(cfg_path):

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Dataset
    ds = load_dataset("text", data_files={"test": TEST_FP})
    ds = ds.map(make_test_text, remove_columns=["text"])

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Quant
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=cfg["load_in_4bit"],
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
    )

    torch.cuda.set_device(0)

    # Base
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=quant_cfg,
        device_map={"": 0},
        torch_dtype=torch.float16,
    )

    # Load LoRA
    ckpt = find_last_checkpoint(cfg["output_dir"])
    model = PeftModel.from_pretrained(base, ckpt)
    model.eval()

    # Dataloader
    loader = DataLoader(ds["test"], batch_size=4)

    preds = []

    for i, batch in enumerate(loader):

        inputs = tok(
            batch["text"],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        with torch.no_grad():
            outs = model.generate(
                **inputs,
                max_new_tokens=256,
            )

        texts = tok.batch_decode(outs, skip_special_tokens=True)

        for inp, out in zip(batch["text"], texts):
            preds.append(out[len(inp):].strip())

        if i % 50 == 0:
            print(f"Processed {i*4}/{len(ds['test'])}")

    # Save
    with open("test_predictions.jsonl", "w") as f:
        for p in preds:
            f.write(json.dumps({"prediction": p}) + "\n")

    print("Saved test_predictions.jsonl")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config")
    args = p.parse_args()

    main(args.config)
