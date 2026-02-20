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

from format_noc import build_prompt, build_label


# -----------------------------
# Paths
# -----------------------------
TRAIN_FP = "/kaggle/input/datasets/haehyunlee/noc-stage1/data/step1_full/train.jsonl"
VALID_FP = "/kaggle/input/datasets/haehyunlee/noc-stage1/data/step1_full/valid.jsonl"


# Dataset formatter

def load_valid_data():

    rows = []

    with open(VALID_FP) as f:
        for line in f:
            row = json.loads(line)

            spec = row["spec"]
            switches = row["switches"]

            prompt = build_prompt(spec)
            label = build_label(switches)

            rows.append({
                "prompt": prompt,
                "label": label.strip()
            })

    return rows

# Find checkpoint

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

    return sorted(
        ckpts,
        key=lambda x: int(x.split("-")[-1])
    )[-1]

# Main

def main(cfg_path):

    # Load config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Load validation samples
    data = load_valid_data()

    print("Validation samples:", len(data))


    # Tokenizer
  
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"


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


    # Load LoRA
  
    ckpt = find_last_checkpoint(cfg["output_dir"])

    if ckpt is None:
        raise RuntimeError("No checkpoint found!")

    print("Using checkpoint:", ckpt)

    model = PeftModel.from_pretrained(base, ckpt)
    model.eval()

  
    # Testing loop
  
    correct = 0
    total = 0

    examples = []

    for item in tqdm(data):

        prompt = item["prompt"]
        gold = item["label"]

        inputs = tok(
            prompt,
            return_tensors="pt"
        ).to(model.device)


        with torch.no_grad():

            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=tok.eos_token_id,
            )


        decoded = tok.decode(
            outputs[0],
            skip_special_tokens=True
        )


        # Extract generated part
        pred = decoded[len(prompt):].strip()


        # Normalize
        pred_norm = " ".join(pred.split())
        gold_norm = " ".join(gold.split())


        match = pred_norm == gold_norm


        if match:
            correct += 1

        total += 1


        # Save some examples
        if len(examples) < 10:

            examples.append({
                "prompt": prompt[:300],
                "gold": gold,
                "pred": pred,
                "correct": match
            })


    acc = correct / total * 100

    print("\n================ TEST RESULTS ================\n")
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {acc:.2f}%\n")


    print("=============== EXAMPLES ===============\n")

    for i, ex in enumerate(examples):

        print(f"Example {i+1}")
        print("Correct:", ex["correct"])
        print("Gold:", ex["gold"])
        print("Pred:", ex["pred"])
        print("-" * 50)


if __name__ == "__main__":

    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default="configs/llama7b_qlora_sft.yaml"
    )

    args = p.parse_args()

    main(args.config)
