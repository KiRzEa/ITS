"""
Recompute validation metrics for already trained models
This script loads saved model checkpoints and recomputes metrics
with the corrected background class handling
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
from torch.utils.data import DataLoader

from src.modern.retinanet.trainer import RetinaNetTrainer, TrafficSignDataset as RetinaNetDataset
from src.modern.ssd.trainer import SSDTrainer, TrafficSignDataset as SSDDataset
from src.modern.faster_rcnn.trainer import FasterRCNNTrainer, TrafficSignDataset as FasterRCNNDataset
from src.utils.detection_validator import validate_model


def collate_fn(batch):
    """Custom collate function for detection"""
    return tuple(zip(*batch))


def recompute_model_metrics(model_dir: Path, model_type: str, backbone: str, num_classes: int, data_root: Path, coco_annotations: Path):
    """
    Recompute metrics for a single model

    Args:
        model_dir: Directory containing the saved model
        model_type: Type of model (retinanet, ssd, faster_rcnn)
        backbone: Backbone architecture
        num_classes: Number of classes (including background)
        data_root: Root directory for data
        coco_annotations: Directory with COCO annotations
    """
    print(f"\n{'='*70}")
    print(f"Recomputing metrics for {model_type.upper()}-{backbone}")
    print(f"{'='*70}")

    # Find best model checkpoint
    best_model_path = model_dir / 'best_model.pth'
    if not best_model_path.exists():
        print(f"⚠️  No checkpoint found at {best_model_path}")
        return None

    print(f"Loading model from: {best_model_path}")

    try:
        # Create validation dataset
        if model_type == 'retinanet':
            val_dataset = RetinaNetDataset(
                images_dir=str(data_root / 'valid' / 'images'),
                annotations_file=str(coco_annotations / 'valid_coco.json'),
                filter_empty=True
            )
            trainer = RetinaNetTrainer(
                num_classes=num_classes,
                backbone=backbone,
                pretrained=False,
                device='cuda'
            )
        elif model_type == 'ssd':
            val_dataset = SSDDataset(
                images_dir=str(data_root / 'valid' / 'images'),
                annotations_file=str(coco_annotations / 'valid_coco.json'),
                filter_empty=True
            )
            trainer = SSDTrainer(
                num_classes=num_classes,
                backbone=backbone,
                pretrained=False,
                device='cuda'
            )
        elif model_type == 'faster_rcnn':
            val_dataset = FasterRCNNDataset(
                images_dir=str(data_root / 'valid' / 'images'),
                annotations_file=str(coco_annotations / 'valid_coco.json'),
                filter_empty=True
            )
            trainer = FasterRCNNTrainer(
                num_classes=num_classes,
                backbone=backbone,
                pretrained=False,
                device='cuda'
            )
        else:
            print(f"Unknown model type: {model_type}")
            return None

        # Load checkpoint
        checkpoint = torch.load(best_model_path, map_location='cuda')
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.model.eval()

        print(f"Model loaded successfully")
        print(f"Validation dataset: {len(val_dataset)} images")

        # Create dataloader
        val_loader = DataLoader(
            val_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn
        )

        # Compute metrics with corrected background handling
        print("\nComputing metrics (with corrected background handling)...")
        metrics = validate_model(
            model=trainer.model,
            dataloader=val_loader,
            device=trainer.device,
            num_classes=num_classes,  # Will be handled correctly inside validate_model
            conf_threshold=0.25,
            verbose=True
        )

        # Save corrected metrics
        corrected_metrics_path = model_dir / 'validation' / 'corrected_metrics.json'
        corrected_metrics_path.parent.mkdir(parents=True, exist_ok=True)

        with open(corrected_metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\n✓ Corrected metrics saved to: {corrected_metrics_path}")

        return {
            'model': f"{model_type}-{backbone}",
            'mAP@0.5': metrics['mAP@0.5'],
            'mAP@0.5:0.95': metrics['mAP@0.5:0.95'],
            'mAP@0.75': metrics['mAP@0.75'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score']
        }

    except Exception as e:
        print(f"✗ Failed to recompute metrics: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Recompute metrics for all trained models"""

    # Configuration
    num_classes = 6  # 5 traffic sign classes + 1 background
    data_root = Path('data/raw/yolov8')
    coco_annotations = Path('data/processed/coco_format')
    experiments_dir = Path('experiments/model_comparison')

    print("\n" + "="*80)
    print("RECOMPUTING METRICS WITH CORRECTED BACKGROUND HANDLING")
    print("="*80)
    print(f"\nThis will recompute metrics for all models in: {experiments_dir}")
    print("The corrected metrics will exclude background class from calculations.")
    print("="*80 + "\n")

    # Define models to recompute
    models_to_check = [
        # RetinaNet models
        ('retinanet', 'resnet50', 'retinanet_resnet50'),
        ('retinanet', 'resnet34', 'retinanet_resnet34'),
        ('retinanet', 'mobilenet_v3', 'retinanet_mobilenet_v3'),

        # SSD models
        ('ssd', 'vgg16', 'ssd_vgg16'),
        ('ssd', 'mobilenet_v3', 'ssd_mobilenet_v3'),

        # Faster R-CNN models
        ('faster_rcnn', 'resnet50', 'faster_rcnn_resnet50'),
        ('faster_rcnn', 'resnet34', 'faster_rcnn_resnet34'),
        ('faster_rcnn', 'mobilenet_v3_large', 'faster_rcnn_mobilenet_v3_large'),
    ]

    results = []

    for model_type, backbone, dir_name in models_to_check:
        model_dir = experiments_dir / dir_name

        if not model_dir.exists():
            print(f"\n⚠️  Skipping {model_type}-{backbone}: directory not found")
            continue

        result = recompute_model_metrics(
            model_dir=model_dir,
            model_type=model_type,
            backbone=backbone,
            num_classes=num_classes,
            data_root=data_root,
            coco_annotations=coco_annotations
        )

        if result:
            results.append(result)

    # Print summary
    if results:
        print("\n" + "="*80)
        print("CORRECTED METRICS SUMMARY")
        print("="*80)
        print(f"\n{'Model':<30} {'mAP@0.5':>10} {'mAP@0.5:0.95':>14} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-"*90)

        for r in results:
            print(f"{r['model']:<30} {r['mAP@0.5']:>10.4f} {r['mAP@0.5:0.95']:>14.4f} "
                  f"{r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1_score']:>10.4f}")

        print("="*80)

        # Save summary
        summary_path = experiments_dir / 'corrected_metrics_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Summary saved to: {summary_path}")
        print("\nNote: These corrected metrics exclude background class (class 0)")
        print("and should be higher than the previously reported metrics.\n")
    else:
        print("\n⚠️  No models found to recompute metrics")


if __name__ == '__main__':
    main()
