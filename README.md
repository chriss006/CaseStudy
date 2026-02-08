# NoC Switch Placement Optimization using Mistral 7B Fine-tuning

## Project Overview

This project fine-tunes the Mistral 7B language model using QLoRA (Quantized Low-Rank Adaptation) to automatically optimize Network-on-Chip (NoC) switch placement. The goal is to train a model that can take a NoC architecture specification (including initiators, targets, connectivity requirements, and blockages) and generate optimal switch placements that satisfy design constraints.

## Project Structure

```
├── configs/                          # Configuration files
│   └── mistral7b_qlora_sft.yaml      # Training configuration (hyperparameters, model settings)
├── data/                             # Data directory
│   ├── raw/                          # Raw test network outputs (testnetworks_out_*.txt files)
│   ├── processed/                    # Processed data ready for training
│   │   ├── train.jsonl               # Training dataset (90% of samples)
│   │   ├── valid.jsonl               # Validation dataset (10% of samples)
│   │   └── split.json                # Data split metadata
│   └── processed_str/                # Stringified version for training
│       ├── train.jsonl               # Training data with JSON as strings
│       └── valid.jsonl               # Validation data with JSON as strings
├── notebooks/                        # Jupyter notebooks and documentation
│   └── mistral7b.ipynb              # Training and analysis notebook
├── outputs/                          # Training outputs and checkpoints
│   └── mistral7b-noc-switch-qlora/   # Fine-tuned model outputs
├── src/                              # Python source code
│   ├── build_dataset.py              # Dataset building and preprocessing script
│   ├── format_noc.py                 # NoC prompt and label formatting utilities
│   └── train_sft.py                  # Supervised fine-tuning training script
└── README.md                         # This file
```

## Data Processing Pipeline

### 1. **Raw Data** (`data/raw/`)
- Contains 150+ text files from test network outputs
- Each file includes:
  - NoC architecture specifications (initiators, targets, connectivity, floorplan dimensions, blockages)
  - Synthesized switch placements (ground truth solutions)

### 2. **Data Extraction** (`build_dataset.py`)
- Parses raw text files to extract:
  - **Specification**: Architecture constraints including:
    - `inits`: Initiator locations and identifiers
    - `targets`: Target locations and identifiers
    - `connectivity`: Required routes between initiators and targets
    - `floorplan_dim`: Chip dimensions
    - `blockages`: Physical obstacles that switches must avoid
  - **Switches**: Optimal switch placements (x, y coordinates)
- Outputs JSONL format for easy processing
- Splits data: 90% training, 10% validation (stratified random split)

### 3. **Prompt Formatting** (`format_noc.py`)
- **Prompt Construction**: Creates structured instructions for the model
  - Specifies the role: "You are an expert NoC physical designer"
  - Provides clear output format expectations (JSON only)
  - Lists physical constraints and rules
  - Includes the architecture specification as context
  
- **Label Creation**: Formats ground truth switch placements as JSON responses

## Training Configuration

**Model**: Mistral-7B-Instruct-v0.2

**Optimization Technique**: QLoRA (Quantized Low-Rank Adaptation)
- 4-bit quantization for memory efficiency
- LoRA rank: 8
- LoRA alpha: 32
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

**Training Hyperparameters**:
- Epochs: 10
- Batch size: 1 (per device)
- Gradient accumulation steps: 16 (effective batch size: 16)
- Learning rate: 2.0e-4 (cosine scheduler with warmup)
- Max sequence length: 1024 tokens
- Evaluation: Every 25 steps
- Checkpoint saving: Every 25 steps (keep 2 latest)

**Hardware Optimization**:
- Uses bfloat16 precision
- 8-bit AdamW optimizer for memory efficiency
- Automatic device mapping

## Training Script (`train_sft.py`)

The training pipeline:
1. **Loads configuration** from YAML file
2. **Prepares dataset** from JSONL files using HuggingFace `datasets` library
3. **Initializes tokenizer** with proper padding token handling
4. **Quantizes model** using BitsAndBytes for 4-bit inference
5. **Configures LoRA** adapters on specified modules
6. **Trains using SFTTrainer** from TRL library with full training callbacks
7. **Saves checkpoints** periodically for resumable training

## Usage

### Dataset Preparation
```bash
python src/build_dataset.py
```
Processes raw data and creates training/validation splits.

### Model Fine-tuning
```bash
python src/train_sft.py configs/mistral7b_qlora_sft.yaml
```
Starts the supervised fine-tuning process. Training progress is saved with periodic checkpoints.

### Resume Training
The script supports resuming from the latest checkpoint by default.

## Key Features

- **Efficient Fine-tuning**: Uses QLoRA to reduce memory footprint while maintaining model quality
- **Structured Output**: Model trained to output valid JSON with switch coordinates
- **Design Constraints**: Incorporates physical constraints (floorplan bounds, blockages) in training
- **Validation**: Continuous evaluation on held-out validation set
- **Reproducibility**: Fixed random seeds and YAML configuration for full reproducibility

## Model Output Format

The fine-tuned model generates responses in the following format:
```json
{"switches": {"s_0": {"x": 701, "y": 670}, "s_1": {"x": 535, "y": 670}, ...}}
```

Each switch placement includes:
- Switch ID (s_0, s_1, etc.)
- X coordinate (must be within floorplan bounds)
- Y coordinate (must be within floorplan bounds and avoid blockages)

## Research Applications

This approach demonstrates how large language models can be specialized for physical design optimization problems:
- **Automated placement**: Reduces manual design effort
- **Constraint satisfaction**: Learns to respect physical constraints
- **Scalability**: Can be applied to different NoC topologies and sizes
- **Expert simulation**: Models trained on expert-designed solutions

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- HuggingFace transformers and datasets
- PEFT (Parameter-Efficient Fine-Tuning)
- TRL (Transformer Reinforcement Learning)
- BitsAndBytes
- PyYAML

## Outputs

Training outputs are saved to `outputs/mistral7b-noc-switch-qlora/` including:
- Model checkpoints at specified intervals
- Training logs and metrics
- Evaluation results on validation set
