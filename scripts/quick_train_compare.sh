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

# Step 1: Download dataset from Roboflow (if not exists)
echo "Step 1/4: Checking/downloading dataset..."
if [ ! -d "$DATA_ROOT/train/images" ]; then
    echo "Dataset not found. Downloading from Roboflow..."
    echo "Note: This requires ROBOFLOW_API_KEY in .env file"

    # Check if .env file exists
    if [ ! -f ".env" ]; then
        echo "Error: .env file not found"
        echo "Please create .env file with your ROBOFLOW_API_KEY"
        echo "See .env.example for reference"
        exit 1
    fi

    # Download dataset using Python script
    python -c "
from src.utils.roboflow_loader import RoboflowDataLoader
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('ROBOFLOW_API_KEY')

if not api_key:
    print('Error: ROBOFLOW_API_KEY not found in .env file')
    exit(1)

loader = RoboflowDataLoader(
    api_key=api_key,
    workspace='giaothong-t5tdy',
    project='phat_hien_bien_bao-zsswb',
    version=1
)

# Download in YOLOv8 format
yolo_path = loader.download_dataset(format='yolov8')
print(f'Dataset downloaded to: {yolo_path}')
"

    if [ $? -ne 0 ]; then
        echo "Error: Failed to download dataset"
        exit 1
    fi
else
    echo "✓ Dataset already exists"
fi
echo ""

# Step 2: Prepare COCO format annotations
echo "Step 2/4: Preparing COCO format annotations..."
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
echo "Step 3/4: Training and comparing all models..."
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
