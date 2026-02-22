#!/bin/bash
# Complete pipeline test for case study
# NoC Generation - End-to-End Evaluation

set -e  # Exit on error

echo "======================================"
echo "NoC Generation Pipeline - Full Test"
echo "======================================"

# Configuration - UPDATE THESE PATHS
MODEL_PATH="${1:-outputs/mistral7b-noc-switch-qlora}"  # Default or first argument
NUM_SAMPLES="${2:-20}"  # Default to 20 samples or second argument
VALIDATION_DATA="${3:-data/processed/valid.jsonl}"  # Default validation data
OUTPUT_DIR="outputs/generated"
REPORT_DIR="outputs/reports"

echo -e "\nConfiguration:"
echo "  Model: $MODEL_PATH"
echo "  Validation data: $VALIDATION_DATA"
echo "  Number of samples: $NUM_SAMPLES"
echo "  Output directory: $OUTPUT_DIR"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "\n❌ ERROR: Model not found at $MODEL_PATH"
    echo "Please train the model first or provide correct model path as first argument:"
    echo "  ./test_complete_pipeline.sh <model_path> [num_samples] [validation_data]"
    exit 1
fi

# Check if validation data exists
if [ ! -f "$VALIDATION_DATA" ]; then
    echo -e "\n❌ ERROR: Validation data not found at $VALIDATION_DATA"
    exit 1
fi

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$REPORT_DIR"
mkdir -p test_specs

echo -e "\n======================================"
echo "[1/6] Extracting test specifications..."
echo "======================================"

if [ -f "src/extract_test_specs.py" ]; then
    python src/extract_test_specs.py
    echo "✅ Test specifications extracted"
else
    echo "⚠️  extract_test_specs.py not found, skipping..."
fi

echo -e "\n======================================"
echo "[2/6] Running batch generation..."
echo "======================================"

if [ -f "src/batch_generate.py" ]; then
    python src/batch_generate.py \
        "$VALIDATION_DATA" \
        --model "$MODEL_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --num-samples "$NUM_SAMPLES" \
        --temperature 0.7
    
    echo "✅ Batch generation complete"
else
    echo "❌ ERROR: batch_generate.py not found"
    echo "Please ensure src/batch_generate.py exists"
    exit 1
fi

echo -e "\n======================================"
echo "[3/6] Evaluating results..."
echo "======================================"

if [ -f "src/evaluate_model.py" ]; then
    python src/evaluate_model.py \
        "$OUTPUT_DIR" \
        --output "$REPORT_DIR/evaluation_report.md"
    
    echo "✅ Evaluation complete"
else
    echo "❌ ERROR: evaluate_model.py not found"
    exit 1
fi

echo -e "\n======================================"
echo "[4/6] Comparing with ground truth..."
echo "======================================"

if [ -f "src/compare_with_ground_truth.py" ] && [ -d "test_specs" ]; then
    # Try batch comparison if we have ground truth files
    GT_COUNT=$(ls test_specs/test_ground_truth_*.json 2>/dev/null | wc -l)
    
    if [ "$GT_COUNT" -gt 0 ]; then
        echo "Found $GT_COUNT ground truth files, running batch comparison..."
        python src/compare_with_ground_truth.py \
            --batch \
            --results-dir "$OUTPUT_DIR" \
            --gt-dir test_specs \
            --output "$REPORT_DIR/comparison_report.md"
        echo "✅ Batch comparison complete"
    else
        # Single comparison with first result
        if [ -f "$OUTPUT_DIR/result_000.json" ] && [ -f "test_specs/test_ground_truth_00.json" ]; then
            echo "Running single comparison (sample 0)..."
            python src/compare_with_ground_truth.py \
                "$OUTPUT_DIR/result_000.json" \
                test_specs/test_ground_truth_00.json
            echo "✅ Single comparison complete"
        else
            echo "⚠️  No ground truth files found for comparison"
        fi
    fi
else
    echo "⚠️  Skipping ground truth comparison (files not found)"
fi

echo -e "\n======================================"
echo "[5/6] Generating summary statistics..."
echo "======================================"

# Count results
TOTAL_RESULTS=$(ls "$OUTPUT_DIR"/result_*.json 2>/dev/null | wc -l)
echo "Total results generated: $TOTAL_RESULTS"

# Parse success count
if command -v jq &> /dev/null; then
    PARSED_COUNT=$(jq -s '[.[] | select(.parsed_successfully == true)] | length' "$OUTPUT_DIR"/result_*.json 2>/dev/null || echo "0")
    VALID_COUNT=$(jq -s '[.[] | select(.is_valid == true)] | length' "$OUTPUT_DIR"/result_*.json 2>/dev/null || echo "0")
    
    echo "Successfully parsed: $PARSED_COUNT / $TOTAL_RESULTS"
    echo "Valid architectures: $VALID_COUNT / $TOTAL_RESULTS"
else
    echo "⚠️  Install 'jq' for detailed JSON statistics"
fi

echo -e "\n======================================"
echo "[6/6] Displaying reports..."
echo "======================================"

# Show evaluation report
if [ -f "$REPORT_DIR/evaluation_report.md" ]; then
    echo -e "\n📊 EVALUATION REPORT:"
    echo "======================================"
    cat "$REPORT_DIR/evaluation_report.md"
else
    echo "⚠️  Evaluation report not found"
fi

# Show comparison report if available
if [ -f "$REPORT_DIR/comparison_report.md" ]; then
    echo -e "\n📊 COMPARISON REPORT:"
    echo "======================================"
    cat "$REPORT_DIR/comparison_report.md"
fi

echo -e "\n======================================"
echo "✅ Pipeline test complete!"
echo "======================================"

echo -e "\nGenerated files:"
echo "  📁 Generation results: $OUTPUT_DIR/"
echo "  📄 Evaluation report: $REPORT_DIR/evaluation_report.md"
echo "  📄 Metrics (JSON): $REPORT_DIR/evaluation_report.json"

if [ -f "$REPORT_DIR/comparison_report.md" ]; then
    echo "  📄 Comparison report: $REPORT_DIR/comparison_report.md"
fi

echo -e "\nNext steps:"
echo "  1. Review evaluation_report.md for performance metrics"
echo "  2. Check individual results in $OUTPUT_DIR/"
echo "  3. Analyze constraint violations for improvements"
echo "  4. Update case study documentation with findings"

echo -e "\n======================================"
