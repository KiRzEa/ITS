"""
Example script demonstrating model validation for all detection approaches
Shows how to compute precision, recall, F1, and mAP metrics
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import torch
from modern.retinanet.trainer import RetinaNetTrainer, TrafficSignDataset as RetinaNetDataset
from modern.ssd.trainer import SSDTrainer, TrafficSignDataset as SSDDataset
from modern.faster_rcnn.trainer import FasterRCNNTrainer, TrafficSignDataset as FasterRCNNDataset
from modern.yolo.trainer import YOLOTrainer


def validate_retinanet_example():
    """Example: Validate a trained RetinaNet model"""
    print("\n" + "="*80)
    print("EXAMPLE 1: RetinaNet Validation")
    print("="*80 + "\n")

    # Initialize trainer
    trainer = RetinaNetTrainer(
        num_classes=5,  # Your number of classes
        backbone='resnet50',
        pretrained=False,
        device='auto'
    )

    # Load trained checkpoint
    checkpoint_path = 'experiments/retinanet/best_model.pth'
    if Path(checkpoint_path).exists():
        trainer.load_checkpoint(checkpoint_path)
    else:
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print("   Please train a model first or adjust the path")
        return

    # Create validation dataset
    val_dataset = RetinaNetDataset(
        images_dir='data/raw/yolov8/valid/images',
        annotations_file='data/processed/coco_format/valid_coco.json',
        transforms=None,
        filter_empty=True
    )

    # Run validation
    metrics = trainer.validate(
        val_dataset=val_dataset,
        batch_size=4,
        conf_threshold=0.25,
        save_results=True,
        save_dir='experiments/retinanet/validation'
    )

    # Access metrics
    print("\n📊 RetinaNet Results:")
    print(f"   mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")
    print(f"   mAP@0.5:     {metrics['mAP@0.5']:.4f}")
    print(f"   Precision:   {metrics['precision']:.4f}")
    print(f"   Recall:      {metrics['recall']:.4f}")
    print(f"   F1 Score:    {metrics['f1_score']:.4f}")


def validate_ssd_example():
    """Example: Validate a trained SSD model"""
    print("\n" + "="*80)
    print("EXAMPLE 2: SSD Validation")
    print("="*80 + "\n")

    # Initialize trainer
    trainer = SSDTrainer(
        num_classes=5,
        backbone='vgg16',  # or 'mobilenet_v3'
        pretrained=False,
        device='auto'
    )

    # Load trained checkpoint
    checkpoint_path = 'experiments/ssd/best_model.pth'
    if Path(checkpoint_path).exists():
        trainer.load_checkpoint(checkpoint_path)
    else:
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print("   Please train a model first or adjust the path")
        return

    # Create validation dataset
    val_dataset = SSDDataset(
        images_dir='data/raw/yolov8/valid/images',
        annotations_file='data/processed/coco_format/valid_coco.json',
        transforms=None,
        filter_empty=True
    )

    # Run validation
    metrics = trainer.validate(
        val_dataset=val_dataset,
        batch_size=8,
        conf_threshold=0.25,
        save_results=True,
        save_dir='experiments/ssd/validation'
    )

    # Access metrics
    print("\n📊 SSD Results:")
    print(f"   mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")
    print(f"   mAP@0.5:     {metrics['mAP@0.5']:.4f}")
    print(f"   Precision:   {metrics['precision']:.4f}")
    print(f"   Recall:      {metrics['recall']:.4f}")
    print(f"   F1 Score:    {metrics['f1_score']:.4f}")


def validate_faster_rcnn_example():
    """Example: Validate a trained Faster R-CNN model"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Faster R-CNN Validation")
    print("="*80 + "\n")

    # Initialize trainer
    trainer = FasterRCNNTrainer(
        num_classes=5,
        backbone='resnet50',
        pretrained=False,
        device='auto'
    )

    # Load trained checkpoint
    checkpoint_path = 'experiments/faster_rcnn/best_model.pth'
    if Path(checkpoint_path).exists():
        trainer.load_checkpoint(checkpoint_path)
    else:
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print("   Please train a model first or adjust the path")
        return

    # Create validation dataset
    val_dataset = FasterRCNNDataset(
        images_dir='data/raw/yolov8/valid/images',
        annotations_file='data/processed/coco_format/valid_coco.json',
        transforms=None,
        filter_empty=True
    )

    # Run validation
    metrics = trainer.validate(
        val_dataset=val_dataset,
        batch_size=4,
        conf_threshold=0.25,
        save_results=True,
        save_dir='experiments/faster_rcnn/validation'
    )

    # Access metrics
    print("\n📊 Faster R-CNN Results:")
    print(f"   mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")
    print(f"   mAP@0.5:     {metrics['mAP@0.5']:.4f}")
    print(f"   Precision:   {metrics['precision']:.4f}")
    print(f"   Recall:      {metrics['recall']:.4f}")
    print(f"   F1 Score:    {metrics['f1_score']:.4f}")


def validate_yolo_example():
    """Example: Validate a trained YOLO model"""
    print("\n" + "="*80)
    print("EXAMPLE 4: YOLO Validation")
    print("="*80 + "\n")

    # Initialize trainer
    trainer = YOLOTrainer(
        model_version='v11',
        model_size='n',
        img_size=640,
        device='auto'
    )

    # Load trained checkpoint
    checkpoint_path = 'experiments/yolo/traffic_sign_detection/weights/best.pt'
    if Path(checkpoint_path).exists():
        trainer.load_checkpoint(checkpoint_path)
    else:
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print("   Please train a model first or adjust the path")
        return

    # Run validation (YOLO uses data.yaml)
    results = trainer.validate(
        data_yaml='data/raw/yolov8/data.yaml',
        split='val'
    )

    # Access metrics (YOLO returns different structure)
    print("\n📊 YOLO Results:")
    print(f"   mAP@0.5:     {results.box.map50:.4f}")
    print(f"   mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"   Precision:   {results.box.mp:.4f}")
    print(f"   Recall:      {results.box.mr:.4f}")


def compare_all_models():
    """Compare all models side by side"""
    print("\n" + "="*80)
    print("MODEL COMPARISON - Validation Metrics")
    print("="*80 + "\n")

    results = {}

    # Validate each model if checkpoint exists
    models = [
        ('RetinaNet', 'experiments/retinanet/best_model.pth'),
        ('SSD', 'experiments/ssd/best_model.pth'),
        ('Faster R-CNN', 'experiments/faster_rcnn/best_model.pth'),
        ('YOLO', 'experiments/yolo/traffic_sign_detection/weights/best.pt'),
    ]

    for model_name, checkpoint_path in models:
        if Path(checkpoint_path).exists():
            print(f"✓ Found {model_name} checkpoint")
        else:
            print(f"✗ Missing {model_name} checkpoint")

    print("\n" + "-"*80)
    print(f"{'Model':<20} {'mAP@0.5':<12} {'mAP@0.5:0.95':<15} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-"*80)

    # Print results (you would fill this in with actual validation results)
    print(f"{'RetinaNet':<20} {'TBD':<12} {'TBD':<15} {'TBD':<12} {'TBD':<12} {'TBD':<12}")
    print(f"{'SSD':<20} {'TBD':<12} {'TBD':<15} {'TBD':<12} {'TBD':<12} {'TBD':<12}")
    print(f"{'Faster R-CNN':<20} {'TBD':<12} {'TBD':<15} {'TBD':<12} {'TBD':<12} {'TBD':<12}")
    print(f"{'YOLO':<20} {'TBD':<12} {'TBD':<15} {'TBD':<12} {'TBD':<12} {'TBD':<12}")
    print("-"*80)

    print("\n💡 Tip: Run individual validation functions to get actual metrics")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Validate object detection models')
    parser.add_argument(
        '--model',
        type=str,
        choices=['retinanet', 'ssd', 'faster_rcnn', 'yolo', 'all'],
        default='all',
        help='Which model to validate'
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("OBJECT DETECTION MODEL VALIDATION")
    print("="*80)

    if args.model == 'retinanet':
        validate_retinanet_example()
    elif args.model == 'ssd':
        validate_ssd_example()
    elif args.model == 'faster_rcnn':
        validate_faster_rcnn_example()
    elif args.model == 'yolo':
        validate_yolo_example()
    elif args.model == 'all':
        compare_all_models()

    print("\n" + "="*80)
    print("Validation Complete!")
    print("="*80 + "\n")
