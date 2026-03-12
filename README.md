NoC Architecture Generation with LLMs

This repository contains the code and experiments for our case study on generating Network-on-Chip (NoC) architectures using Large Language Models (LLMs).
We explore different approaches including fine-tuning, few-shot prompting, and single-stage prediction for generating switch placements and routing structures.

Repository Structure
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
    Generated outputs from experiments.
Methods

We explored multiple approaches for generating NoC architectures:

Stage 1 Prediction

Switch placement prediction using structured JSON outputs.

Single-Stage Prediction

Simultaneous generation of switch placements and routing paths.

Few-shot Prompting

Providing examples to guide the model to generate structured outputs.

Fine-tuning

Using QLoRA-based supervised fine-tuning to adapt the model for the NoC generation task.

Evaluation

Generated architectures are evaluated using several structural validity checks:

Parsed Output Rate – percentage of outputs that can be parsed into valid JSON

Switch Placement Validity – correctness of generated switch coordinates

Path Element Validity – correctness of routing path elements

Cycle-Free Routing – verification that routing paths do not contain cycles

Blockage Validity – ensuring that routing does not intersect with forbidden regions

Running Experiments

Example workflow:

Dataset Preparation
python src/mistral7b/build_dataset.py
Fine-tuning
python src/mistral7b/train_sft.py --config configs/mistral7b_qlora_sft.yaml
Few-shot Prediction

Use the scripts in:

src/openllama/

Example:

python src/openllama/3shot_noc_prompt.py
Requirements

Main dependencies include:

Python 3.10+

PyTorch

Transformers

PEFT

BitsAndBytes

HuggingFace Datasets

Install dependencies:

pip install -r requirements.txt
Authors

Sophie-Anna

Haehyun Lee

Lakshmi

Ezhil

Case study project on LLM-based NoC architecture generation.
