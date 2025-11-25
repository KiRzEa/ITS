"""
Detection Validation Module
Compute precision, recall, F1, and mAP for object detection models
Compatible with RetinaNet, SSD, Faster R-CNN, and other PyTorch detection models
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
from pathlib import Path
import json


class DetectionValidator:
    """
    Validate object detection models with standard metrics
    Computes precision, recall, F1, mAP@0.5, and mAP@0.5:0.95
    """

    def __init__(
        self,
        num_classes: int,
        iou_thresholds: List[float] = None,
        conf_threshold: float = 0.001
    ):
        """
        Initialize validator

        Args:
            num_classes: Number of classes (excluding background)
            iou_thresholds: IoU thresholds for mAP calculation
                           Default: [0.5:0.95:0.05] for COCO-style mAP
            conf_threshold: Confidence threshold for filtering predictions
        """
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold

        if iou_thresholds is None:
            # COCO-style: 0.5 to 0.95 with step 0.05
            self.iou_thresholds = np.linspace(0.5, 0.95, 10).tolist()
        else:
            self.iou_thresholds = iou_thresholds

        # Storage for predictions and ground truth
        self.reset()

    def reset(self):
        """Reset internal state"""
        self.predictions = []  # List of (boxes, labels, scores)
        self.ground_truths = []  # List of (boxes, labels)

    def add_batch(
        self,
        pred_boxes: List[np.ndarray],
        pred_labels: List[np.ndarray],
        pred_scores: List[np.ndarray],
        gt_boxes: List[np.ndarray],
        gt_labels: List[np.ndarray]
    ):
        """
        Add a batch of predictions and ground truth

        Args:
            pred_boxes: List of predicted boxes [N, 4] for each image
            pred_labels: List of predicted labels [N] for each image
            pred_scores: List of predicted scores [N] for each image
            gt_boxes: List of ground truth boxes [M, 4] for each image
            gt_labels: List of ground truth labels [M] for each image
        """
        for pb, pl, ps, gb, gl in zip(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
            # Filter by confidence threshold
            if len(ps) > 0:
                keep = ps >= self.conf_threshold
                pb = pb[keep]
                pl = pl[keep]
                ps = ps[keep]

            self.predictions.append((pb, pl, ps))
            self.ground_truths.append((gb, gl))

    def compute_metrics(self, iou_threshold: float = 0.5) -> Dict:
        """
        Compute detection metrics at a specific IoU threshold

        Args:
            iou_threshold: IoU threshold for matching predictions to ground truth

        Returns:
            Dictionary containing metrics
        """
        # Collect all predictions and ground truths per class
        all_pred_boxes = {c: [] for c in range(self.num_classes)}
        all_pred_scores = {c: [] for c in range(self.num_classes)}
        all_gt_boxes = {c: [] for c in range(self.num_classes)}
        all_image_ids = {c: [] for c in range(self.num_classes)}

        # Organize by class
        for img_id, ((pred_boxes, pred_labels, pred_scores), (gt_boxes, gt_labels)) in \
                enumerate(zip(self.predictions, self.ground_truths)):

            # Add predictions
            for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                all_pred_boxes[label].append(box)
                all_pred_scores[label].append(score)
                all_image_ids[label].append(img_id)

            # Add ground truths
            for box, label in zip(gt_boxes, gt_labels):
                all_gt_boxes[label].append((img_id, box))

        # Compute AP for each class
        aps = []
        precisions = []
        recalls = []

        for class_id in range(self.num_classes):
            pred_boxes = np.array(all_pred_boxes[class_id])
            pred_scores = np.array(all_pred_scores[class_id])
            image_ids = np.array(all_image_ids[class_id])
            gt_boxes = all_gt_boxes[class_id]

            if len(gt_boxes) == 0:
                # No ground truth for this class
                aps.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
                continue

            if len(pred_boxes) == 0:
                # No predictions for this class
                aps.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
                continue

            # Sort by confidence
            sort_idx = np.argsort(-pred_scores)
            pred_boxes = pred_boxes[sort_idx]
            image_ids = image_ids[sort_idx]
            pred_scores = pred_scores[sort_idx]

            # Match predictions to ground truth
            tp = np.zeros(len(pred_boxes))
            fp = np.zeros(len(pred_boxes))

            # Track which ground truths have been matched
            gt_matched = {i: False for i in range(len(gt_boxes))}

            for pred_idx, (pred_box, img_id) in enumerate(zip(pred_boxes, image_ids)):
                # Get ground truths for this image
                img_gt_boxes = [gt_box for gt_img_id, gt_box in gt_boxes if gt_img_id == img_id]
                img_gt_indices = [i for i, (gt_img_id, _) in enumerate(gt_boxes) if gt_img_id == img_id]

                if len(img_gt_boxes) == 0:
                    fp[pred_idx] = 1
                    continue

                # Compute IoU with all ground truths in this image
                ious = np.array([self._compute_iou(pred_box, gt_box) for gt_box in img_gt_boxes])

                # Find best matching ground truth
                max_iou_idx = np.argmax(ious)
                max_iou = ious[max_iou_idx]

                gt_idx = img_gt_indices[max_iou_idx]

                if max_iou >= iou_threshold and not gt_matched[gt_idx]:
                    tp[pred_idx] = 1
                    gt_matched[gt_idx] = True
                else:
                    fp[pred_idx] = 1

            # Compute precision and recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)

            recalls_curve = tp_cumsum / len(gt_boxes)
            precisions_curve = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

            # Compute AP using 11-point interpolation
            ap = self._compute_ap(recalls_curve, precisions_curve)
            aps.append(ap)

            # Store final precision and recall
            if len(precisions_curve) > 0:
                precisions.append(precisions_curve[-1])
                recalls.append(recalls_curve[-1])
            else:
                precisions.append(0.0)
                recalls.append(0.0)

        # Aggregate metrics
        mAP = np.mean(aps)
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)

        # Compute F1 score
        if mean_precision + mean_recall > 0:
            f1 = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall)
        else:
            f1 = 0.0

        # Per-class metrics
        per_class_metrics = {}
        for class_id in range(self.num_classes):
            per_class_metrics[f'class_{class_id}'] = {
                'ap': aps[class_id],
                'precision': precisions[class_id],
                'recall': recalls[class_id]
            }

        return {
            'mAP': mAP,
            'precision': mean_precision,
            'recall': mean_recall,
            'f1_score': f1,
            'per_class': per_class_metrics
        }

    def compute_coco_metrics(self) -> Dict:
        """
        Compute COCO-style metrics (mAP@0.5:0.95)

        Returns:
            Dictionary with COCO metrics
        """
        # Compute mAP at different IoU thresholds
        maps = []
        for iou_thresh in self.iou_thresholds:
            metrics = self.compute_metrics(iou_threshold=iou_thresh)
            maps.append(metrics['mAP'])

        # Compute metrics at IoU=0.5 (standard)
        metrics_50 = self.compute_metrics(iou_threshold=0.5)

        # Compute metrics at IoU=0.75
        metrics_75 = self.compute_metrics(iou_threshold=0.75)

        return {
            'mAP@0.5:0.95': np.mean(maps),
            'mAP@0.5': metrics_50['mAP'],
            'mAP@0.75': metrics_75['mAP'],
            'precision': metrics_50['precision'],
            'recall': metrics_50['recall'],
            'f1_score': metrics_50['f1_score'],
            'per_class': metrics_50['per_class']
        }

    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Compute IoU between two boxes

        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]

        Returns:
            IoU value
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

    @staticmethod
    def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
        """
        Compute Average Precision using 11-point interpolation

        Args:
            recalls: Recall values
            precisions: Precision values

        Returns:
            Average Precision
        """
        # Add sentinel values at the beginning and end
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([0.0], precisions, [0.0]))

        # Compute the precision envelope
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])

        # Compute AP using 11-point interpolation
        recall_thresholds = np.linspace(0, 1, 11)
        ap = 0.0

        for thresh in recall_thresholds:
            # Find precision at this recall threshold
            idx = np.where(recalls >= thresh)[0]
            if len(idx) > 0:
                ap += precisions[idx[0]]

        return ap / 11.0


def validate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    conf_threshold: float = 0.001,
    verbose: bool = True
) -> Dict:
    """
    Validate a PyTorch detection model

    Args:
        model: Detection model (RetinaNet, SSD, Faster R-CNN, etc.)
        dataloader: Validation dataloader
        device: Device to use
        num_classes: Number of classes (including background class 0)
        conf_threshold: Confidence threshold
        verbose: Print progress

    Returns:
        Dictionary with validation metrics
    """
    model.eval()

    # For PyTorch detection models, predictions include background class (class 0)
    # but ground truth doesn't. We need to:
    # 1. Filter out class 0 predictions
    # 2. Remap class labels to exclude background (1->0, 2->1, etc.)
    # 3. Use num_classes-1 for validator (excluding background)

    validator = DetectionValidator(num_classes=num_classes - 1, conf_threshold=conf_threshold)

    iterator = tqdm(dataloader, desc="Validating") if verbose else dataloader

    with torch.no_grad():
        for images, targets in iterator:
            images = [img.to(device) for img in images]

            # Get predictions
            predictions = model(images)

            # Extract predictions and ground truth
            pred_boxes = []
            pred_labels = []
            pred_scores = []
            gt_boxes = []
            gt_labels = []

            for pred, target in zip(predictions, targets):
                # Predictions - filter out background class (class 0) and remap
                boxes_np = pred['boxes'].cpu().numpy()
                labels_np = pred['labels'].cpu().numpy()
                scores_np = pred['scores'].cpu().numpy()

                # Filter out background class (class 0)
                non_bg_mask = labels_np > 0
                boxes_np = boxes_np[non_bg_mask]
                labels_np = labels_np[non_bg_mask]
                scores_np = scores_np[non_bg_mask]

                # Remap labels: 1->0, 2->1, 3->2, etc. (subtract 1)
                labels_np = labels_np - 1

                pred_boxes.append(boxes_np)
                pred_labels.append(labels_np)
                pred_scores.append(scores_np)

                # Ground truth - remap labels to 0-indexed (COCO format uses 1-indexed categories)
                gt_boxes.append(target['boxes'].cpu().numpy())
                gt_labels_np = target['labels'].cpu().numpy()
                # Remap labels: 1->0, 2->1, 3->2, etc. (subtract 1)
                gt_labels_np = gt_labels_np - 1
                gt_labels.append(gt_labels_np)

            # Add to validator
            validator.add_batch(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels)

    # Compute metrics
    metrics = validator.compute_coco_metrics()

    if verbose:
        print(f"\n{'='*60}")
        print("Validation Results")
        print(f"{'='*60}")
        print(f"mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")
        print(f"mAP@0.5:     {metrics['mAP@0.5']:.4f}")
        print(f"mAP@0.75:    {metrics['mAP@0.75']:.4f}")
        print(f"Precision:   {metrics['precision']:.4f}")
        print(f"Recall:      {metrics['recall']:.4f}")
        print(f"F1 Score:    {metrics['f1_score']:.4f}")
        print(f"{'='*60}")

        print("\nPer-class AP@0.5 (excluding background):")
        for class_id in range(num_classes - 1):
            ap = metrics['per_class'][f'class_{class_id}']['ap']
            print(f"  Class {class_id}: {ap:.4f}")
        print(f"{'='*60}\n")

    return metrics


def save_metrics(metrics: Dict, save_path: str):
    """
    Save metrics to JSON file

    Args:
        metrics: Metrics dictionary
        save_path: Path to save JSON file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    metrics_serializable = convert_numpy(metrics)

    with open(save_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)

    print(f"Metrics saved to: {save_path}")
