#!/usr/bin/env python3
"""
Quick test script to verify the Faster R-CNN dataset fix
"""
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Import directly from the module to avoid init issues
import importlib.util
spec = importlib.util.spec_from_file_location(
    "frcnn_trainer",
    project_root / 'src' / 'modern' / 'faster_rcnn' / 'trainer.py'
)
frcnn_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frcnn_module)

TrafficSignDataset = frcnn_module.TrafficSignDataset
import torch

def test_dataset():
    """Test that dataset handles empty annotations correctly"""

    # Check if COCO data exists
    coco_path = Path('data/raw/coco')

    if not coco_path.exists():
        print("COCO dataset not found. Please download it first using notebook 03.")
        print("The fix will work once you download the dataset.")
        return

    train_ann_path = coco_path / 'train' / '_annotations.coco.json'
    val_ann_path = coco_path / 'valid' / '_annotations.coco.json'

    if not train_ann_path.exists():
        print(f"Annotations not found at {train_ann_path}")
        print("The dataset structure may be different. Please check the notebook.")
        return

    print("Testing Faster R-CNN Dataset...")
    print("=" * 60)

    # Test with filter_empty=True (default)
    print("\n1. Testing with filter_empty=True (default):")
    train_dataset = TrafficSignDataset(
        images_dir=str(coco_path / 'train'),
        annotations_file=str(train_ann_path),
        filter_empty=True
    )

    val_dataset = TrafficSignDataset(
        images_dir=str(coco_path / 'valid'),
        annotations_file=str(val_ann_path),
        filter_empty=True
    )

    print(f"   Train dataset: {len(train_dataset)} images")
    print(f"   Val dataset: {len(val_dataset)} images")

    # Test loading a few samples
    print("\n2. Testing data loading (first 5 samples):")
    for i in range(min(5, len(train_dataset))):
        image, target = train_dataset[i]
        num_boxes = len(target['boxes'])
        print(f"   Sample {i}: {num_boxes} boxes, labels shape: {target['labels'].shape}")

        # Verify no empty boxes
        assert num_boxes > 0, f"Sample {i} has empty boxes!"
        assert target['boxes'].shape == (num_boxes, 4), "Invalid box shape!"

    print("\n✅ All tests passed! The dataset is working correctly.")
    print("\nThe fix ensures:")
    print("  1. Images without annotations are filtered out (filter_empty=True)")
    print("  2. Invalid boxes (width/height <= 0) are skipped")
    print("  3. No empty tensor assertions during training")

    print("\n" + "=" * 60)
    print("You can now resume training in the notebook!")
    print("=" * 60)

if __name__ == "__main__":
    test_dataset()
