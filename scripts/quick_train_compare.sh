#!/bin/bash

# Quick Start Script for Training and Comparing All Models
# This script automates the entire process from data preparation to visualization

set -e  # Exit on error

echo "=================================="
echo "OBJECT DETECTION MODEL COMPARISON"
echo "Quick Start Script"
echo "=================================="
echo ""

# Configuration
EPOCHS=${1:-20}
DATA_ROOT="data/raw/yolov8"
COCO_DIR="data/processed/coco_format"
EXPERIMENTS_DIR="experiments/model_comparison"

echo "Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Data root: $DATA_ROOT"
echo "  COCO annotations: $COCO_DIR"
echo "  Experiments dir: $EXPERIMENTS_DIR"
echo ""

# Step 1: Check if data exists
echo "Step 1/3: Checking data..."
if [ ! -d "$DATA_ROOT/train/images" ]; then
    echo "Error: Training data not found at $DATA_ROOT/train/images"
    echo "Please ensure your data is in the correct location."
    exit 1
fi
echo "✓ Data found"
echo ""

# Step 2: Prepare COCO format annotations
echo "Step 2/3: Preparing COCO format annotations..."
if [ ! -f "$COCO_DIR/train_coco.json" ]; then
    echo "Converting YOLO format to COCO format..."
    python scripts/prepare_coco_format.py \
        --data-root "$DATA_ROOT" \
        --output-dir "$COCO_DIR"
else
    echo "✓ COCO format annotations already exist"
fi
echo ""

# Step 3: Train and compare all models
echo "Step 3/3: Training and comparing all models..."
echo "This will take several hours depending on your GPU."
echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5

python scripts/train_and_compare_all_models.py \
    --epochs "$EPOCHS" \
    --data-root "$DATA_ROOT" \
    --coco-annotations "$COCO_DIR" \
    --experiments-dir "$EXPERIMENTS_DIR"

# Done
echo ""
echo "=================================="
echo "COMPARISON COMPLETE!"
echo "=================================="
echo ""
echo "Results saved to: $EXPERIMENTS_DIR"
echo ""
echo "View your results:"
echo "  - Comparison plots: $EXPERIMENTS_DIR/model_comparison.png"
echo "  - Training curves: $EXPERIMENTS_DIR/training_curves.png"
echo "  - Detailed report: $EXPERIMENTS_DIR/comparison_report.txt"
echo "  - JSON results: $EXPERIMENTS_DIR/all_results.json"
echo ""
echo "To view the comparison plot:"
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "  open $EXPERIMENTS_DIR/model_comparison.png"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "  xdg-open $EXPERIMENTS_DIR/model_comparison.png"
else
    echo "  start $EXPERIMENTS_DIR/model_comparison.png"
fi
echo ""
