# Project Gap Analysis: NoC Generation with LLMs

## Overview
This document outlines the gap between the **Arteris IP Requirements** (NOC_Arteris_Requirement.md) and the **Current Implementation** (README.md), identifying what has been completed and what remains pending.

---

## ✅ Completed Components

### 1. **Dataset Processing Pipeline**
- ✅ Raw data parsing from classical solver outputs
- ✅ Extraction of specifications (inits, targets, connectivity, floorplan, blockages)
- ✅ JSONL format conversion
- ✅ Train/validation split (90/10)
- ✅ File: `src/build_dataset.py`

### 2. **Prompt Engineering**
- ✅ Structured prompt formatting for model input
- ✅ JSON-based output specification
- ✅ Role definition ("expert NoC physical designer")
- ✅ File: `src/format_noc.py`

### 3. **Training Infrastructure**
- ✅ QLoRA fine-tuning implementation
- ✅ Mistral-7B-Instruct-v0.2 model integration
- ✅ Training script with checkpoint management
- ✅ Configuration management (YAML)
- ✅ File: `src/train_sft.py`

---

## ⚠️ Partially Completed / Incomplete Components

### 1. **Output Format** (Partially Complete)
**Current State:**
- ✅ Switches output format implemented
- ❌ **Routing paths generation NOT implemented**

**Requirement:**
```json
{
  "switches": {"s_0": {"x": 701, "y": 670}, ...},
  "routing_paths": {"r_0": ["i_0", "s_1", "s_2", "t_0"], ...}
}
```

**Gap:** Model currently only generates switches. Routing paths must be generated as part of the output to complete the requirement.

**Action Items:**
- [ ] Modify `format_noc.py` to include routing paths in labels
- [ ] Update training to include routing path prediction
- [ ] Extend model output parsing to extract routing paths
- [ ] Validate routing paths in post-processing

---

## ❌ Not Started / Missing Components

### 1. **Constraint Validation Module**
**Requirement:** Validate that generated architectures satisfy hard constraints
- Routes are connected (all initiator-target pairs have valid paths)
- Switches not placed in blockage zones
- Network is deadlock-free
- Switch coordinates within floorplan bounds

**Gap:** No validation logic implemented

**Action Items:**
- [ ] Create `src/validate_architecture.py` with functions:
  - `validate_switch_placement()` - check floorplan bounds and blockage avoidance
  - `validate_route_connectivity()` - verify all required routes exist
  - `validate_deadlock_free()` - detect cycles in routing graph
  - `validate_complete_network()` - comprehensive validation
- [ ] Integrate validation into evaluation pipeline
- [ ] Add validation metrics to training/evaluation reports

### 2. **Inference/Generation Module**
**Requirement:** Ability to generate new architectures from unseen specifications

**Gap:** No inference script or module for using the trained model

**Action Items:**
- [ ] Create `src/generate_architecture.py` with:
  - Model loading and configuration
  - Input specification parsing
  - Inference with appropriate prompts
  - Output JSON parsing and validation
  - Error handling for invalid outputs
- [ ] Add command-line interface for batch generation
- [ ] Include JSON schema validation for inputs/outputs

### 3. **Evaluation Metrics & Analysis**
**Requirement:** Evaluate based on:
- Training loss
- Output validity (constraints satisfied)
- Wirelength analysis (soft constraint)
- Route-length analysis (soft constraint)
- Generalization to unseen architectures

**Gap:** Only basic training loss tracking; no comprehensive evaluation framework

**Action Items:**
- [ ] Create `src/evaluate_model.py` with metrics:
  - Constraint satisfaction rate (%)
  - Wirelength metrics (total vs optimal)
  - Route-length metrics (average hops)
  - Comparison to ground truth solutions
- [ ] Generate evaluation reports and visualizations
- [ ] Add metrics tracking during training
- [ ] Benchmark against classical solver solutions

### 4. **Data Representation Exploration** (Secondary Goal)
**Requirement:** Explore alternative data representations that may improve training/generalization

**Current Implementation:**
- Simple JSON text format (as provided in dataset)

**Gap:** No exploration of alternative representations

**Action Items:**
- [ ] Implement alternative formats:
  - Graph-based representation (adjacency lists)
  - Distance matrices (pre-computed distances between all points)
  - Floorplan constraint encoding (visual/spatial representation)
  - Normalized coordinates and relative distances
  - Pre-computed network statistics (degree, centrality, etc.)
- [ ] Comparative analysis of representations
- [ ] Training with alternative formats
- [ ] Document impact on model performance

### 5. **Optimization for Soft Constraints**
**Requirement:** Minimize wirelength and route-length (cost-performance tradeoff)

**Gap:** Model trained only to reproduce ground truth; no explicit optimization for these metrics

**Action Items:**
- [ ] Analyze wirelength in training data
- [ ] Explore loss function modifications:
  - Multi-objective loss combining validity and wire/route metrics
  - Reward-based training for optimization
- [ ] Implement wirelength calculation module
- [ ] Fine-tune model variants optimizing for different criteria
- [ ] Compare solutions against Pareto frontier

### 6. **Scalability Testing**
**Requirement:** Handle datasets with ~1k networks; potential for larger networks

**Gap:** No testing on different network sizes or complexity levels

**Action Items:**
- [ ] Analyze current dataset size distribution
- [ ] Test model on larger networks (if available)
- [ ] Profile memory and inference time
- [ ] Document scalability limitations
- [ ] Plan for dataset expansion if needed

### 7. **Multi-Step/Iterative Generation** (Implicit Requirement)
**Requirement:** For logical constraints, "modifying data representation, or having inference being done in multiple steps might be a good candidate"

**Gap:** Single-pass generation implemented; no multi-step refinement

**Action Items:**
- [ ] Design multi-step inference pipeline:
  - Step 1: Generate switch placement
  - Step 2: Validate and refine
  - Step 3: Generate routing paths
  - Step 4: Final validation and optimization
- [ ] Implement iterative refinement with constraint feedback
- [ ] Test improvement from multi-step approach

### 8. **Visualization & Debugging Tools**
**Gap:** No tools to visualize generated architectures and compare with ground truth

**Action Items:**
- [ ] Create visualization module for:
  - Floorplan with switches, initiators, targets
  - Routing paths visualization
  - Blockage overlay
  - Comparison plots (generated vs ground truth)
- [ ] Generate HTML reports for architecture review
- [ ] Create debugging tools for failed validations

### 9. **Documentation & Reporting**
**Gap:** Missing detailed documentation on architecture validation, evaluation results

**Action Items:**
- [ ] Document constraint validation algorithms
- [ ] Create evaluation report templates
- [ ] Document data format specifications
- [ ] Write inference usage guide
- [ ] Create troubleshooting guide

---

## Summary Table

| Component | Status | Priority | Est. Effort |
|-----------|--------|----------|------------|
| Dataset Processing | ✅ Complete | - | - |
| Prompt Engineering | ✅ Complete | - | - |
| Training Infrastructure | ✅ Complete | - | - |
| Routing Paths Output | ⚠️ Incomplete | **Critical** | Medium |
| Constraint Validation | ❌ Missing | **Critical** | High |
| Inference Module | ❌ Missing | **High** | Medium |
| Evaluation Framework | ⚠️ Partial | **High** | Medium |
| Data Representation Exploration | ❌ Missing | Medium | High |
| Soft Constraint Optimization | ❌ Missing | Medium | High |
| Multi-Step Generation | ❌ Missing | Medium | High |
| Visualization Tools | ❌ Missing | Low | Medium |
| Documentation | ⚠️ Partial | Medium | Low |

---

## Critical Path (Minimum Viable Product)

To achieve the core project goal, these items must be completed:

1. **Add Routing Paths to Model Output**
2. **Implement Constraint Validation Module**
3. **Create Inference Pipeline**
4. **Build Comprehensive Evaluation Framework**
5. **Validate Model on Test Set**

Estimated effort: **2-3 weeks**

---

## Extended Roadmap (Full Project)

After core completion:

1. **Data Representation Exploration** (weeks 3-4)
2. **Soft Constraint Optimization** (weeks 4-5)
3. **Multi-Step Generation** (weeks 5-6)
4. **Visualization & Tools** (week 6)
5. **Final Testing & Documentation** (week 7)

Estimated full project effort: **7-8 weeks**

---

## Dependency Graph

```
Dataset Processing (✅)
    ↓
Prompt Engineering (✅)
    ↓
Training Infrastructure (✅)
    ├→ Add Routing Paths (⚠️) 
    │   ├→ Constraint Validation (❌) ← CRITICAL
    │   └→ Evaluation Framework (⚠️) ← CRITICAL
    │       └→ Inference Module (❌) ← CRITICAL
    │
    ├→ Data Representation (❌)
    │   └→ Comparative Analysis
    │
    └→ Soft Constraint Optimization (❌)
        └→ Multi-Step Generation (❌)
```

---

## Next Steps

**Immediate Priority:**
1. Review routing paths in dataset - understand expected format
2. Modify training labels to include routing paths
3. Start constraint validation implementation
4. Create test suite for validation functions

**Recommendation:**
Begin with routing paths implementation and constraint validation in parallel, as they are prerequisites for the evaluation framework.
