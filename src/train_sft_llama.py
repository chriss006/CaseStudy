import os
import yaml
import torch
import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from format_noc import build_prompt, build_label

# Paths to your pre-split dataset
TRAIN_FP = "/kaggle/input/datasets/haehyunlee/noc-stage1/data/step1_full/train.jsonl"
VALID_FP = "/kaggle/input/datasets/haehyunlee/noc-stage1/data/step1_full/valid.jsonl"

# Convert a JSONL row to the text the model will train on
def make_text_from_line(ex):
    row = json.loads(ex["text"])
    spec = row["spec"]
    switches = row["switches"]
    return {"text": build_prompt(spec) + build_label(switches)}

# Find last checkpoint for resume
def find_last_checkpoint(output_dir: str):
    if not os.path.isdir(output_dir):
        return None
    candidates = []
    for name in os.listdir(output_dir):
        if name.startswith("checkpoint-"):
            p = os.path.join(output_dir, name)
            if os.path.isdir(p):
                candidates.append(p)
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: int(p.split("-")[-1]))[-1]

# Format function for TRL SFTTrainer (new API)
def format_fn(example):
    return example["text"]

# Main function
def main(cfg_path: str, resume: bool = False):
    # Load YAML config
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg["output_dir"], exist_ok=True)

    # Load datasets
    ds = load_dataset(
        "text",
        data_files={"train": TRAIN_FP, "validation": VALID_FP},
    )
    ds = ds.map(make_text_from_line, remove_columns=['text'])

    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'right'

    # Quantization config for 4-bit training
    #quant_cfg = BitsAndBytesConfig(
   #     load_in_4bit=cfg["load_in_4bit"],
    #    bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
      #  bnb_4bit_compute_dtype=getattr(torch, cfg["bnb_4bit_compute_dtype"]),
     #   bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
    #)

    # Load LLaMA base model
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],            # LLaMA-7B checkpoint
        #quantization_config=quant_cfg,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False

    # LoRA config
    lora_cfg = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg["lora_target_modules"],  # ["q_proj","v_proj"] for LLaMA
    )
    model = get_peft_model(model, lora_cfg)

    # Training arguments
    args = TrainingArguments(
    output_dir=cfg["output_dir"],
    num_train_epochs=cfg["num_train_epochs"],
    per_device_train_batch_size=cfg["per_device_train_batch_size"],
    per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
    gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
    learning_rate=cfg["learning_rate"],
    warmup_steps=cfg["warmup_ratio"],
    weight_decay=cfg["weight_decay"],
    lr_scheduler_type=cfg["lr_scheduler_type"],
    logging_dir=os.path.join(cfg["output_dir"], "logs"),
    logging_first_step=True,
    logging_strategy="steps",
    logging_steps=cfg["logging_steps"],
    #evaluation_strategy="steps",
    eval_steps=cfg["eval_steps"],           # replace evaluation_strategy
    save_strategy="steps",
    save_steps=cfg["save_steps"],
    save_total_limit=cfg["save_total_limit"],
    optim="adamw_torch",
    report_to="none",
    seed=cfg["seed"],
    bf16=False,
    fp16=True,
)


    # Trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        formatting_func=format_fn,
        #max_seq_length=cfg["max_seq_length"],
        packing=False,
        args=args,
    )

    # Resume or start training
    if resume:
        ckpt = find_last_checkpoint(cfg["output_dir"])
        print("Resume from:", ckpt)
        if ckpt is None:
            trainer.train()
        else:
            trainer.train(resume_from_checkpoint=ckpt)
    else:
        trainer.train()

    # Save everything
    trainer.save_model(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])
    model.save_pretrained(os.path.join(cfg["output_dir"], "adapter"))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/llama7b_qlora_sft.yaml")
    p.add_argument("--resume", action="store_true", help="resume from last checkpoint")
    a = p.parse_args()
    main(a.config, resume=a.resume)
