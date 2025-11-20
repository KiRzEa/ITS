#!/usr/bin/env python
"""
Quick Start Script for Traffic Sign Detection
Downloads dataset and runs a simple YOLOv11 training demo
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from utils.roboflow_loader import RoboflowDataLoader
from modern.yolo.trainer import YOLOTrainer
import torch

def main():
    print("="*70)
    print("Traffic Sign Detection - Quick Start")
    print("="*70)

    # Check GPU
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ No GPU detected - training will be slow!")

    # Step 1: Download dataset
    print("\n" + "="*70)
    print("Step 1: Downloading dataset from Roboflow")
    print("="*70)

    loader = RoboflowDataLoader(
        api_key="gbb6oBUEJlstBEF0CruH",
        workspace="GiaoThong",
        project="phat_hien_bien_bao-zsswb",
        version=1,
        data_dir="data/raw"
    )

    # Download YOLOv8 format
    dataset_path = loader.download_dataset(format="yolov8")

    # Get dataset info
    info = loader.get_dataset_info()
    print(f"\n✓ Dataset downloaded successfully!")
    print(f"  Location: {dataset_path}")
    print(f"  Classes ({info['num_classes']}): {', '.join(info['class_names'])}")

    # Step 2: Train YOLOv11
    print("\n" + "="*70)
    print("Step 2: Training YOLOv11 (Demo - 10 epochs)")
    print("="*70)
    print("\nNote: For full training, increase epochs to 100+ in the notebooks")

    # Initialize trainer
    trainer = YOLOTrainer(
        model_size='n',  # nano - fastest for demo
        img_size=640,
        device='auto'
    )

    # Train for just 10 epochs as demo
    data_yaml = Path(dataset_path) / 'data.yaml'

    print("\nStarting training...")
    results = trainer.train(
        data_yaml=str(data_yaml),
        epochs=10,  # Small number for demo
        batch_size=16,
        patience=5,
        save_dir='experiments/quick_start',
        name='demo'
    )

    # Step 3: Validate
    print("\n" + "="*70)
    print("Step 3: Validating model")
    print("="*70)

    val_results = trainer.validate(data_yaml=str(data_yaml), split='val')

    print(f"\n✓ Training completed!")
    print(f"  mAP@0.5: {val_results.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {val_results.box.map:.4f}")
    print(f"  Precision: {val_results.box.mp:.4f}")
    print(f"  Recall: {val_results.box.mr:.4f}")

    # Step 4: Export model
    print("\n" + "="*70)
    print("Step 4: Exporting model to ONNX")
    print("="*70)

    onnx_path = trainer.export(format='onnx')
    print(f"\n✓ Model exported to: {onnx_path}")

    # Summary
    print("\n" + "="*70)
    print("Quick Start Completed!")
    print("="*70)
    print("\nNext steps:")
    print("1. Run notebooks for detailed experiments:")
    print("   - notebooks/01_data_exploration.ipynb")
    print("   - notebooks/02_traditional_methods.ipynb")
    print("   - notebooks/03_modern_methods.ipynb")
    print("\n2. Train with more epochs for better results:")
    print("   - Modify epochs=100 in YOLOTrainer.train()")
    print("\n3. Try different model sizes:")
    print("   - YOLOv11n (fastest)")
    print("   - YOLOv11s (balanced)")
    print("   - YOLOv11m (more accurate)")
    print("\n4. Compare with traditional methods:")
    print("   - HOG+SVM: Fast, CPU-friendly")
    print("   - Color+Shape: No training needed")
    print("\nHappy detecting! 🚦")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
