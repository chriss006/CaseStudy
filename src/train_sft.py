import json
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

def make_text(example):
    return {"text": example["spec"] + example["output"]}

def main():
    # Configuration
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    output_dir = "outputs/mistral7b-noc-switch-qlora"
    
    # QLoRA configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # LoRA configuration
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Load tokenizer and model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        offload_buffers=True,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    
    # Load datasets
    print("Loading datasets...")
    train_data = []
    with open("data/processed_str_output/train.jsonl", "r") as f:
        for line in f:
            train_data.append(json.loads(line))
    
    valid_data = []
    with open("data/processed_str_output/valid.jsonl", "r") as f:
        for line in f:
            valid_data.append(json.loads(line))
    
    train_dataset = Dataset.from_list(train_data).map(make_text, remove_columns=["id","spec", "output"])
    valid_dataset = Dataset.from_list(valid_data).map(make_text, remove_columns=["id", "spec", "output"])
   
    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")

        #num_train_epochs=3,
        #per_device_train_batch_size=1,
        #gradient_accumulation_steps=16,

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        max_steps=5,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=True,
        optim="paged_adamw_8bit",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        max_seq_length=1024,
        dataset_text_field="text",
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()
