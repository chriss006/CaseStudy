import os
import json
import yaml
import torch
from tqdm.auto import tqdm

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

import sys
sys.path.append("/kaggle/working/CaseStudy/src")

from fewshot_format_noc import build_fewshot_full_prompt


# =====================================================
# CONFIG
# =====================================================

TEST_FP = "/kaggle/working/step2_full/test.jsonl"

CFG_PATH = "/kaggle/working/CaseStudy/configs/llama7b.yaml"

OUT_DIR = "/kaggle/working/CaseStudy/outputs/fewshot_validation"

CKPT_DIR = "/kaggle/input/datasets/chetana092004/llama7b-stage1-ckpt/v2/checkpoint-3200"


# How many samples to test
N_SAMPLES = 500

# GPU batch size
BATCH_SIZE = 4

# Generation length
MAX_NEW_TOKENS = 256


PRED_PATH = os.path.join(OUT_DIR, "predictions.jsonl")
STATS_PATH = os.path.join(OUT_DIR, "stats.json")


os.makedirs(OUT_DIR, exist_ok=True)


# =====================================================
# UTILS
# =====================================================

def clean_output(txt):
    """
    Extract first valid JSON object from model output
    """

    start = txt.find("{")
    end = txt.rfind("}")

    if start == -1 or end == -1:
        return None

    return txt[start:end+1]


def safe_json_load(txt):

    try:
        return json.loads(txt)
    except:
        return None


def batchify(lst, n):

    for i in range(0, len(lst), n):
        yield lst[i:i+n]


# =====================================================
# LOAD CONFIG
# =====================================================

with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)


# =====================================================
# TOKENIZER
# =====================================================

tok = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)

if tok.pad_token is None:
    tok.pad_token = tok.eos_token

tok.padding_side = "left"


# =====================================================
# QUANTIZATION
# =====================================================

quant_cfg = BitsAndBytesConfig(
    load_in_4bit=cfg["load_in_4bit"],
    bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
    bnb_4bit_compute_dtype=getattr(torch, cfg["bnb_4bit_compute_dtype"]),
    bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
)


# =====================================================
# LOAD MODEL
# =====================================================

print("Loading base model...")

base = AutoModelForCausalLM.from_pretrained(
    cfg["model_name"],
    quantization_config=quant_cfg,
    device_map="auto",
)

base.config.use_cache = True


print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(base, CKPT_DIR)

model.eval()


# =====================================================
# LOAD DATA
# =====================================================

print("Loading dataset...")

raw = load_dataset("text", data_files={"test": TEST_FP})["test"]

raw = raw.select(range(N_SAMPLES))

data = [json.loads(x["text"]) for x in raw]


# =====================================================
# STATS
# =====================================================

n_total = 0
n_json_ok = 0
n_has_switches = 0


# =====================================================
# INFERENCE
# =====================================================

print("Starting inference...")

with open(PRED_PATH, "w", encoding="utf-8") as fout:

    for batch in tqdm(list(batchify(data, BATCH_SIZE))):

        # Build few-shot prompts
        prompts = [
            build_fewshot_full_prompt(r["spec"])
            for r in batch
        ]

        # Tokenize
        inputs = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg["max_seq_length"],
        ).to(model.device)


        # Generate
        with torch.no_grad():

            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )


        # Decode
        texts = tok.batch_decode(outputs, skip_special_tokens=True)


        # Process outputs
        for row, prompt, full in zip(batch, prompts, texts):

            # Remove prompt
            pred = full[len(prompt):].strip()


            # Clean JSON
            cleaned = clean_output(pred)

            pred_json = None

            if cleaned:
                pred_json = safe_json_load(cleaned)


            # Stats
            ok_json = pred_json is not None
            ok_switch = ok_json and "switches" in pred_json


            n_total += 1
            n_json_ok += int(ok_json)
            n_has_switches += int(ok_switch)


            # Save
            fout.write(json.dumps({
                "pred_text": pred,
                "cleaned_json": cleaned,
                "pred_json": pred_json,
                "gt_switches": row.get("switches"),
            }, ensure_ascii=False) + "\n")


# =====================================================
# SAVE STATS
# =====================================================

stats = {

    "n_total": n_total,

    "json_ok": n_json_ok,
    "json_ok_rate": round(n_json_ok / n_total, 4),

    "has_switches": n_has_switches,
    "has_switches_rate": round(n_has_switches / n_total, 4),

    "pred_file": PRED_PATH,
}


with open(STATS_PATH, "w") as f:
    json.dump(stats, f, indent=2)


print("\n===== FEW-SHOT VALIDATION =====")
print(json.dumps(stats, indent=2))

print("\nSaved predictions to:", PRED_PATH)
print("Saved stats to:", STATS_PATH)
