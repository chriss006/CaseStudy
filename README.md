# NoC Switch Placement Optimization using Mistral 7B Fine-tuning

## Project Overview

This project fine-tunes the Mistral 7B language model using QLoRA (Quantized Low-Rank Adaptation) to automatically optimize Network-on-Chip (NoC) switch placement. The goal is to train a model that can take a NoC architecture specification (including initiators, targets, connectivity requirements, and blockages) and generate optimal switch placements that satisfy design constraints.

## Project Structure

```
├── arteris_docs/                     # Documentation and design notes
│   ├── LLM_network_gen_Arteris.md   # Network generation approach
│   ├── NOC_Arteris_Requirement.md   # NoC requirements
│   └── Next_Steps.md                # Future improvements
├── configs/                          # Configuration files
│   ├── mistral7b_qlora_sft.yaml      # Full training configuration
│   └── mistral7b_qlora_sft_quick.yaml # Quick testing configuration
├── data/                             # Data processing and storage
│   ├── raw/                          # Raw test network outputs (150+ files)
│   ├── processed/                    # Processed training-ready data
│   │   ├── train.jsonl               # Training dataset (90%)
│   │   ├── valid.jsonl               # Validation dataset (10%)
│   │   └── split.json                # Data split metadata
│   ├── processed_backup/             # Backup of original processed data
│   ├── processed_str/                # Stringified JSON version
│   │   ├── train.jsonl               # Training data (JSON as strings)
│   │   └── valid.jsonl               # Validation data (JSON as strings)
│   └── processed_str_output/         # Output format variants
├── notebooks/                        # Jupyter notebooks and analysis
│   ├── mistral7b.ipynb              # Model training and evaluation
│   ├── validate_model_colab.ipynb   # Google Colab validation pipeline
│   ├── architecture_generation_for_noc.ipynb  # Architecture generation demo
│   ├── train_noc_with_routing_paths.ipynb     # Routing path training
│   └── notebook.md                  # Notebook documentation
├── outputs/                          # Model checkpoints and results
│   ├── mistral7b-noc-switch-qlora/   # Fine-tuned LoRA adapter weights
│   ├── result_*.json                 # Inference results (0-4)
│   └── research_outputs/             # Detailed analysis results
├── test_specs/                       # Test cases and ground truth
│   ├── test_spec_*.json              # Test specifications (0-4)
│   └── test_ground_truth_*.json      # Expected outputs
├── src/                              # Python source code utilities
│   ├── build_dataset.py              # Data extraction and preprocessing
│   ├── format_noc.py                 # Prompt and label formatting
│   ├── train_sft.py                  # Supervised fine-tuning trainer
│   ├── validate_architecture.py      # Constraint validation module
│   ├── generate_architecture.py      # Inference and generation
│   ├── evaluate_model.py             # Model evaluation metrics
│   ├── batch_generate.py             # Batch inference processing
│   ├── create_stringified_data.py    # JSON stringification utility
│   ├── extract_test_specs.py         # Test case extraction
│   ├── compare_with_ground_truth.py  # Result comparison tool
│   └── test_*.py                     # Unit tests for components
├── test_complete_pipeline.sh         # End-to-end testing script
├── training_log.txt                  # Training execution log
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

### 1. Dataset Preparation
```bash
# Extract and process raw data
python src/build_dataset.py

# Create stringified JSON variants if needed
python src/create_stringified_data.py
```

### 2. Train the Model
```bash
# Full training (recommended)
python src/train_sft.py configs/mistral7b_qlora_sft.yaml

# Quick test/debug run
python src/train_sft.py configs/mistral7b_qlora_sft_quick.yaml
```

### 3. Generate Architectures
```bash
# Single inference example
python src/generate_architecture.py --spec <spec_json> --model <model_path>

# Batch inference on test set
python src/batch_generate.py --input data/processed_str/valid.jsonl --output outputs/results.jsonl
```

### 4. Validate & Evaluate
```bash
# Validate architecture constraints
python src/validate_architecture.py <spec.json> <output.json>

# Comprehensive evaluation with metrics
python src/evaluate_model.py --predictions outputs/results.jsonl --ground-truth data/test_specs/

# Compare with ground truth
python src/compare_with_ground_truth.py --results outputs/results.jsonl
```

### 5. Run Complete Pipeline
```bash
# Execute end-to-end testing
bash test_complete_pipeline.sh
```

### 6. Use Jupyter Notebooks
```bash
# Training and analysis
jupyter notebook notebooks/mistral7b.ipynb

# Model validation on Google Colab
jupyter notebook notebooks/validate_model_colab.ipynb

# Architecture generation demo
jupyter notebook notebooks/architecture_generation_for_noc.ipynb

# Routing path analysis
jupyter notebook notebooks/train_noc_with_routing_paths.ipynb
```

## Key Features

- **Efficient Fine-tuning**: Uses QLoRA to reduce memory footprint while maintaining model quality
- **Structured Output**: Model trained to output valid JSON with switch placements and routing paths
- **Multi-level Validation**: 
  - Constraint satisfaction (floorplan bounds, blockage avoidance)
  - Path connectivity verification
  - Cycle detection in routing paths
- **Comprehensive Evaluation**:
  - Accuracy metrics against ground truth
  - Constraint violation analysis
  - Performance benchmarking
- **Batch Processing**: Efficient inference on large test datasets
- **Reproducibility**: Fixed random seeds and YAML configuration for full reproducibility
- **Google Colab Support**: Ready to deploy validation on cloud GPUs

## Validation Framework

The project includes a comprehensive validation module (`ArchitectureValidator`) that checks:

1. **Switch Placement Validation**
   - Coordinates within floorplan bounds
   - No overlapping with blockages
   - Valid coordinate ranges

2. **Path Element Validation**
   - All nodes in paths exist in spec
   - Path data structure integrity
   - Node identifier consistency

3. **Route Connectivity Validation**
   - All required routes are present
   - Routes connect correct initiators to targets
   - Path endpoints match connectivity requirements

4. **Cycle Detection**
   - No repeating nodes in routing paths
   - Acyclic routing guarantees

## Model Output Format

The fine-tuned model generates responses in the following format:

```json
{
  "switches": {
    "s_0": {"x": 701, "y": 670},
    "s_1": {"x": 535, "y": 670},
    "s_2": {"x": 535, "y": 946}
  },
  "routing_paths": {
    "r_0": ["i_0", "s_0", "s_1", "t_0"],
    "r_1": ["i_0", "s_0", "t_1"],
    "r_2": ["i_1", "s_1", "t_0"]
  }
}
```

**Components**:
- **switches**: Dictionary mapping switch IDs to (x, y) coordinates
  - Must be within floorplan bounds
  - Must not overlap with blockages
- **routing_paths**: Dictionary mapping route IDs to node sequences
  - Starts with an initiator (i_N)
  - Ends with a target (t_N)
  - May traverse through switches (s_N) for multi-hop routes
  - Must not contain cycles

## Research Applications

This approach demonstrates how large language models can be specialized for physical design optimization problems:

- **Automated Placement**: Reduces manual design effort and improves design turnaround time
- **Constraint Satisfaction**: Learns to consistently respect physical and connectivity constraints
- **Scalability**: Can be applied to different NoC topologies, sizes, and complexity levels
- **Expert Simulation**: Models trained on expert-designed solutions capture domain knowledge
- **Design Space Exploration**: Can sample multiple valid solutions for optimization

## Project Artifacts

### Documentation
- `arteris_docs/LLM_network_gen_Arteris.md`: Detailed approach and methodology
- `arteris_docs/NOC_Arteris_Requirement.md`: NoC requirements and specifications
- `arteris_docs/Next_Steps.md`: Future improvements and extensions
- `training_log.txt`: Training execution details and metrics

### Benchmarks
- 150+ test network specifications from various topology configurations
- Ground truth solutions for validation and comparison
- Pre-computed test cases in `test_specs/`

### Trained Models
- Fine-tuned Mistral-7B with LoRA adapters
- 4-bit quantized for efficient inference
- Compatible with HuggingFace PEFT and transformers libraries

## Citation

If you use this project in your research, please cite:

```bibtex
@project{noc_mistral_qlora,
  title={NoC Switch Placement Optimization using Mistral 7B Fine-tuning},
  year={2024},
  organization={Arteris},
  description={Specialized LLM for automated Network-on-Chip design optimization}
}
```

## License

[Your License Here]

## Contact & Support

For questions, issues, or suggestions, please refer to the documentation in `arteris_docs/` or the project's issue tracker.

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- HuggingFace transformers and datasets
- PEFT (Parameter-Efficient Fine-Tuning)
- TRL (Transformer Reinforcement Learning)
- BitsAndBytes
- PyYAML
- Matplotlib (for visualization)

## Data Format Details

### Input Specification (NoC Architecture)
```json
{
  "inits": {"i_0": {"x": 438, "y": 74}, ...},
  "targets": {"t_0": {"x": 439, "y": 988}, ...},
  "connectivity": {"r_0": ["i_0", "t_0"], ...},
  "floorplan_dim": [1000, 1000],
  "blockages": {"b_0": {"x": 226, "y": 76, "width": 248, "height": 9}, ...}
}
```

### Output Format (Generated Architecture)
```json
{
  "switches": {"s_0": {"x": 701, "y": 670}, ...},
  "routing_paths": {"r_0": ["i_0", "s_0", "t_0"], ...}
}
```

### Dataset Format (JSONL)
Each line contains:
```json
{"id": "testnetworks_out_586", "spec": "...", "output": "..."}
```

## Testing

### Unit Tests
Run individual module tests:
```bash
python src/test_validation.py
python src/test_evaluation.py
python src/test_inference_components.py
python src/test_routing_paths.py
```

### Test Specifications
Pre-made test cases in `test_specs/` directory:
- `test_spec_*.json`: Input specifications
- `test_ground_truth_*.json`: Expected model outputs

### Complete Pipeline Test
```bash
bash test_complete_pipeline.sh
```

## Outputs

Training and inference outputs are organized as follows:

- **Model Checkpoints**: `outputs/mistral7b-noc-switch-qlora/`
  - LoRA adapter weights for Mistral-7B
  - Training logs and metrics
  - Checkpoint states for resumable training

- **Inference Results**: `outputs/result_*.json`
  - Sample predictions and validation results
  - Performance metrics and error analysis

- **Documentation**: `arteris_docs/`
  - Network generation approach documentation
  - NoC requirements and specifications
  - Future improvements and next steps
