"""
Test Set Evaluation - Final Performance Assessment
Evaluate all trained models on the test set (holdout data)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
from torch.utils.data import DataLoader
import argparse

from src.modern.retinanet.trainer import RetinaNetTrainer, TrafficSignDataset as RetinaNetDataset
from src.modern.ssd.trainer import SSDTrainer, TrafficSignDataset as SSDDataset
from src.modern.faster_rcnn.trainer import FasterRCNNTrainer, TrafficSignDataset as FasterRCNNDataset
from src.utils.detection_validator import validate_model


def collate_fn(batch):
    """Custom collate function for detection"""
    return tuple(zip(*batch))


def evaluate_on_test_set(
    model_dir: Path,
    model_type: str,
    backbone: str,
    num_classes: int,
    data_root: Path,
    coco_annotations: Path,
    batch_size: int = 8
):
    """
    Evaluate a trained model on the test set

    Args:
        model_dir: Directory containing the saved model
        model_type: Type of model (retinanet, ssd, faster_rcnn)
        backbone: Backbone architecture
        num_classes: Number of classes (including background)
        data_root: Root directory for data
        coco_annotations: Directory with COCO annotations
        batch_size: Batch size for evaluation
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING {model_type.upper()}-{backbone} ON TEST SET")
    print(f"{'='*80}")

    # Find best model checkpoint
    best_model_path = model_dir / 'best_model.pth'
    if not best_model_path.exists():
        print(f"⚠️  No checkpoint found at {best_model_path}")
        return None

    print(f"Loading model from: {best_model_path}")

    try:
        # Create test dataset
        if model_type == 'retinanet':
            test_dataset = RetinaNetDataset(
                images_dir=str(data_root / 'test' / 'images'),
                annotations_file=str(coco_annotations / 'test_coco.json'),
                filter_empty=True
            )
            trainer = RetinaNetTrainer(
                num_classes=num_classes,
                backbone=backbone,
                pretrained=False,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        elif model_type == 'ssd':
            test_dataset = SSDDataset(
                images_dir=str(data_root / 'test' / 'images'),
                annotations_file=str(coco_annotations / 'test_coco.json'),
                filter_empty=True
            )
            trainer = SSDTrainer(
                num_classes=num_classes,
                backbone=backbone,
                pretrained=False,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        elif model_type == 'faster_rcnn':
            test_dataset = FasterRCNNDataset(
                images_dir=str(data_root / 'test' / 'images'),
                annotations_file=str(coco_annotations / 'test_coco.json'),
                filter_empty=True
            )
            trainer = FasterRCNNTrainer(
                num_classes=num_classes,
                backbone=backbone,
                pretrained=False,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            print(f"Unknown model type: {model_type}")
            return None

        # Load checkpoint
        checkpoint = torch.load(best_model_path, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.model.eval()

        print(f"✓ Model loaded successfully")
        print(f"✓ Test dataset: {len(test_dataset)} images")
        print(f"✓ Device: {trainer.device}")

        # Create dataloader
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn
        )

        # Evaluate on test set
        print(f"\n{'='*60}")
        print("RUNNING TEST SET EVALUATION")
        print(f"{'='*60}")

        metrics = validate_model(
            model=trainer.model,
            dataloader=test_loader,
            device=trainer.device,
            num_classes=num_classes,
            conf_threshold=0.25,
            verbose=True
        )

        # Save test results
        test_results_path = model_dir / 'test_results.json'
        with open(test_results_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\n✓ Test results saved to: {test_results_path}")

        return {
            'model': f"{model_type}-{backbone}",
            'model_type': model_type,
            'backbone': backbone,
            'test_images': len(test_dataset),
            'mAP@0.5': metrics['mAP@0.5'],
            'mAP@0.5:0.95': metrics['mAP@0.5:0.95'],
            'mAP@0.75': metrics['mAP@0.75'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'per_class': metrics['per_class']
        }

    except Exception as e:
        print(f"✗ Failed to evaluate: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models on test set')
    parser.add_argument('--data-root', type=str, default='data/raw/yolov8',
                        help='Root directory for YOLO format data')
    parser.add_argument('--coco-annotations', type=str, default='data/processed/coco_format',
                        help='Directory with COCO format annotations')
    parser.add_argument('--experiments-dir', type=str, default='experiments/model_comparison',
                        help='Directory containing trained models')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Specific models to evaluate (e.g., retinanet_resnet50)')

    args = parser.parse_args()

    # Configuration
    num_classes = 6  # 5 traffic sign classes + 1 background
    data_root = Path(args.data_root)
    coco_annotations = Path(args.coco_annotations)
    experiments_dir = Path(args.experiments_dir)

    print("\n" + "="*80)
    print("TEST SET EVALUATION - FINAL PERFORMANCE ASSESSMENT")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data root: {data_root}")
    print(f"  COCO annotations: {coco_annotations}")
    print(f"  Experiments dir: {experiments_dir}")
    print(f"  Batch size: {args.batch_size}")
    print("="*80 + "\n")

    # Check if test set exists
    test_images_path = data_root / 'test' / 'images'
    test_annotations_path = coco_annotations / 'test_coco.json'

    if not test_images_path.exists():
        print(f"❌ Error: Test images not found at {test_images_path}")
        print("   Please ensure the test set is available.")
        return

    if not test_annotations_path.exists():
        print(f"❌ Error: Test annotations not found at {test_annotations_path}")
        print("   Please run prepare_coco_format.py to generate test annotations.")
        return

    print(f"✓ Test set found: {test_images_path}")
    print(f"✓ Test annotations found: {test_annotations_path}\n")

    # Define models to evaluate
    all_models = [
        ('retinanet', 'resnet50', 'retinanet_resnet50'),
        ('retinanet', 'resnet34', 'retinanet_resnet34'),
        ('retinanet', 'mobilenet_v3', 'retinanet_mobilenet_v3'),
        ('ssd', 'vgg16', 'ssd_vgg16'),
        ('ssd', 'mobilenet_v3', 'ssd_mobilenet_v3'),
        ('faster_rcnn', 'resnet50', 'faster_rcnn_resnet50'),
        ('faster_rcnn', 'resnet34', 'faster_rcnn_resnet34'),
        ('faster_rcnn', 'mobilenet_v3_large', 'faster_rcnn_mobilenet_v3_large'),
    ]

    # Filter models if specified
    if args.models:
        all_models = [(t, b, d) for t, b, d in all_models if d in args.models]

    results = []

    for model_type, backbone, dir_name in all_models:
        model_dir = experiments_dir / dir_name

        if not model_dir.exists():
            print(f"\n⚠️  Skipping {model_type}-{backbone}: directory not found")
            continue

        if not (model_dir / 'best_model.pth').exists():
            print(f"\n⚠️  Skipping {model_type}-{backbone}: no trained model found")
            continue

        result = evaluate_on_test_set(
            model_dir=model_dir,
            model_type=model_type,
            backbone=backbone,
            num_classes=num_classes,
            data_root=data_root,
            coco_annotations=coco_annotations,
            batch_size=args.batch_size
        )

        if result:
            results.append(result)

    # Print summary
    if results:
        print("\n\n" + "="*80)
        print("TEST SET EVALUATION SUMMARY")
        print("="*80)
        print(f"\n{'Model':<30} {'mAP@0.5':>10} {'mAP@0.75':>10} {'mAP@0.5:0.95':>14} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-"*100)

        for r in results:
            print(f"{r['model']:<30} {r['mAP@0.5']:>10.4f} {r['mAP@0.75']:>10.4f} "
                  f"{r['mAP@0.5:0.95']:>14.4f} {r['precision']:>10.4f} "
                  f"{r['recall']:>10.4f} {r['f1_score']:>10.4f}")

        print("="*100)

        # Best model
        best_model = max(results, key=lambda x: x['mAP@0.5:0.95'])
        print(f"\n🏆 Best Model on Test Set: {best_model['model']}")
        print(f"   mAP@0.5:     {best_model['mAP@0.5']:.4f}")
        print(f"   mAP@0.75:    {best_model['mAP@0.75']:.4f}")
        print(f"   mAP@0.5:0.95: {best_model['mAP@0.5:0.95']:.4f}")
        print(f"   Precision:   {best_model['precision']:.4f}")
        print(f"   Recall:      {best_model['recall']:.4f}")
        print(f"   F1 Score:    {best_model['f1_score']:.4f}")

        # Save summary
        summary_path = experiments_dir / 'test_set_results_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Summary saved to: {summary_path}")

        print("\n" + "="*80)
        print("📌 IMPORTANT NOTES:")
        print("="*80)
        print("1. These are the FINAL results on the holdout test set")
        print("2. Metrics exclude background class (corrected computation)")
        print("3. Use these numbers for reporting final model performance")
        print("4. Compare with validation results to check for overfitting")
        print("="*80 + "\n")
    else:
        print("\n⚠️  No models were evaluated")
        print("   Make sure models are trained and checkpoints exist.")


if __name__ == '__main__':
    main()
