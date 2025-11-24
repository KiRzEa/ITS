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

    # Download dataset using Python script (avoid importing project modules)
    python << 'ENDPYTHON'
import os
import sys
from pathlib import Path

# Ensure we're in the correct directory
print(f'Current working directory: {os.getcwd()}')

# Create data/raw directory if it doesn't exist
data_raw = Path('data/raw')
data_raw.mkdir(parents=True, exist_ok=True)
print(f'Created/verified directory: {data_raw.absolute()}')

# Load environment variables manually
env_file = Path('.env')
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

api_key = os.getenv('ROBOFLOW_API_KEY')
if not api_key:
    print('Error: ROBOFLOW_API_KEY not found in .env file')
    sys.exit(1)

# Import only Roboflow - no project imports
from roboflow import Roboflow

# Download using Roboflow API directly
print('Connecting to Roboflow...')
rf = Roboflow(api_key=api_key)
project = rf.workspace('giaothong-t5tdy').project('phat_hien_bien_bao-zsswb')
print('Downloading dataset...')

# Roboflow downloads to: location/project-name-version
# So if location='data/raw', it creates: data/raw/phat_hien_bien_bao-zsswb-1/
dataset = project.version(1).download('yolov8', location='data/raw')

# Get actual download location
download_location = dataset.location
print(f'Dataset downloaded to: {download_location}')

# Check if we need to move/rename the directory
downloaded_dir = Path(download_location).absolute()
target_dir = Path('data/raw/yolov8').absolute()

if downloaded_dir != target_dir:
    if downloaded_dir.exists():
        print(f'Moving from {downloaded_dir} to {target_dir}')
        if target_dir.exists():
            import shutil
            shutil.rmtree(target_dir)
        # Ensure parent directory exists
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        # Use shutil.move for cross-filesystem compatibility
        import shutil
        shutil.move(str(downloaded_dir), str(target_dir))
        print(f'✓ Dataset moved to: {target_dir}')
    else:
        print(f'⚠️  Download location does not exist: {downloaded_dir}')
else:
    print(f'✓ Dataset is already at correct location: {target_dir}')

# Verify the download
train_images = target_dir / 'train' / 'images'
if train_images.exists():
    num_images = len(list(train_images.glob('*.*')))
    print(f'✓ Verified: Found {num_images} training images')
else:
    print(f'⚠️  Warning: Training images not found at {train_images}')
    print(f'   Directory contents:')
    if target_dir.exists():
        for item in target_dir.iterdir():
            print(f'     - {item.name}')

ENDPYTHON

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
echo "$PWD"
sleep 5

python train_and_compare_all_models.py \
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
