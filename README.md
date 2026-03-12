# Network on Chip Generation with LLMs
This repository contains the code and experiments for our case study on generating Network-on-Chip (NoC) architectures using Large Language Models (LLMs).
We explore different approaches including fine-tuning, few-shot prompting, and single-stage prediction for generating switch placements and routing structures.

## Repository Structure
```
    configs/
        llama7b.yaml
        mistral7b_qlora_sft.yaml
        Configuration files for model training.
    
    data/
        Training and evaluation datasets.
    
    notebooks/
        llama7b-singleshot.ipynb
        mistral7b.ipynb
        validation-fewshot.ipynb
        validation-of-noc-architecture.ipynb
        Experimental notebooks for running inference and validation.
    
    src/
        mistral7b/
            build_dataset.py
            format_noc.py
            train_sft.py
            Scripts for dataset preparation and fine-tuning with Mistral7B.
    
        openllama/
            3shot_noc_prompt.py
            fewshot_noc_text.py
            fewshot_validate_stage2.py
            fewshot_validate_text.py
            oneshot_noc_prompt.py
            Scripts for few-shot and one-shot prompting experiments.
    
    outputs/
        checkpoints from experiments.
```
## Methods
We explored multiple approaches for generating NoC architectures:

1. Stage 1 Prediction

    - switch placement prediction using structured JSON outputs.

2. Single-Stage Prediction

    - Simultaneous generation of switch placements and routing paths.

3. Few-shot Prompting

    - Providing examples to guide the model to generate structured outputs.

4. Fine-tuning

    - Using QLoRA-based supervised fine-tuning to adapt the model for the NoC generation task.

## Evaluation
Generated architectures are evaluated using several structural validity checks:

- Parsed Output Rate – percentage of outputs that can be parsed into valid JSON

- Switch Placement Validity – correctness of generated switch coordinates, ensuring that routing does not intersect with forbidden regions

- Path Element Validity – correctness of routing path elements

- Cycle-Free Routing – verification that routing paths do not contain cycles

## Run Experiments

The experiments can be reproduced using the following notebooks.

- **llama7b-singleshot.ipynb**  
  Runs one-shot inference experiments using OpenLLaMA for NoC switch prediction.

- **mistral7b.ipynb**  
  Performs inference using the fine-tuned Mistral7B model.

- **validation-fewshot.ipynb**  
  Evaluates few-shot prediction outputs and computes structural validity metrics.

- **validation-of-noc-architecture.ipynb**  
  Validates generated NoC architectures, including routing correctness and structural constraints.

## Requirements
Main dependencies include:

- Python 3.10+

- PyTorch

- Transformers

- PEFT

- BitsAndBytes

- HuggingFace Datasets

## Authors
- Sophie-Anna
- Haehyun Lee
- Lakshmi
- Ezhil

Case study project for MSc Data Science & AI UNiCA

