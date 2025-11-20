"""
Evaluation metrics for object detection
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import torch


class DetectionMetrics:
    """Calculate detection metrics"""

    def __init__(self, num_classes: int, iou_threshold: float = 0.5):
        """
        Initialize metrics calculator

        Args:
            num_classes: Number of classes
            iou_threshold: IoU threshold for matching predictions to ground truth
        """
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.tp = defaultdict(int)  # True positives per class
        self.fp = defaultdict(int)  # False positives per class
        self.fn = defaultdict(int)  # False negatives per class
        self.scores = defaultdict(list)  # Confidence scores per class
        self.all_predictions = []
        self.all_ground_truths = []

    @staticmethod
    def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate IoU between two boxes

        Args:
            box1: First box [x1, y1, x2, y2]
            box2: Second box [x1, y1, x2, y2]

        Returns:
            IoU score
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - intersection

        return intersection / (union + 1e-6)

    def update(
        self,
        pred_boxes: np.ndarray,
        pred_classes: np.ndarray,
        pred_scores: np.ndarray,
        gt_boxes: np.ndarray,
        gt_classes: np.ndarray
    ):
        """
        Update metrics with predictions and ground truths

        Args:
            pred_boxes: Predicted boxes [N, 4] (x1, y1, x2, y2)
            pred_classes: Predicted classes [N]
            pred_scores: Prediction confidence scores [N]
            gt_boxes: Ground truth boxes [M, 4]
            gt_classes: Ground truth classes [M]
        """
        self.all_predictions.append({
            'boxes': pred_boxes,
            'classes': pred_classes,
            'scores': pred_scores
        })
        self.all_ground_truths.append({
            'boxes': gt_boxes,
            'classes': gt_classes
        })

        # Track which ground truths have been matched
        matched_gt = set()

        # Sort predictions by confidence (highest first)
        if len(pred_scores) > 0:
            sorted_idx = np.argsort(pred_scores)[::-1]
            pred_boxes = pred_boxes[sorted_idx]
            pred_classes = pred_classes[sorted_idx]
            pred_scores = pred_scores[sorted_idx]

        # Match predictions to ground truths
        for pred_box, pred_class, pred_score in zip(pred_boxes, pred_classes, pred_scores):
            best_iou = 0
            best_gt_idx = -1

            # Find best matching ground truth
            for gt_idx, (gt_box, gt_class) in enumerate(zip(gt_boxes, gt_classes)):
                if gt_idx in matched_gt:
                    continue

                if pred_class != gt_class:
                    continue

                iou = self.calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # Check if match is valid
            if best_iou >= self.iou_threshold and best_gt_idx != -1:
                self.tp[pred_class] += 1
                matched_gt.add(best_gt_idx)
                self.scores[pred_class].append((pred_score, 1))  # (score, is_tp)
            else:
                self.fp[pred_class] += 1
                self.scores[pred_class].append((pred_score, 0))  # (score, is_fp)

        # Count unmatched ground truths as false negatives
        for gt_idx, gt_class in enumerate(gt_classes):
            if gt_idx not in matched_gt:
                self.fn[gt_class] += 1

    def compute_precision_recall(self, class_id: int) -> Tuple[float, float]:
        """
        Compute precision and recall for a class

        Args:
            class_id: Class ID

        Returns:
            (precision, recall)
        """
        tp = self.tp[class_id]
        fp = self.fp[class_id]
        fn = self.fn[class_id]

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)

        return precision, recall

    def compute_f1_score(self, class_id: int) -> float:
        """Compute F1 score for a class"""
        precision, recall = self.compute_precision_recall(class_id)
        return 2 * (precision * recall) / (precision + recall + 1e-6)

    def compute_ap(self, class_id: int) -> float:
        """
        Compute Average Precision for a class

        Args:
            class_id: Class ID

        Returns:
            Average Precision
        """
        if class_id not in self.scores or len(self.scores[class_id]) == 0:
            return 0.0

        # Sort by confidence score (descending)
        scores_labels = sorted(self.scores[class_id], key=lambda x: x[0], reverse=True)

        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls = []

        total_positives = self.tp[class_id] + self.fn[class_id]

        if total_positives == 0:
            return 0.0

        for score, is_tp in scores_labels:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1

            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            recall = tp_cumsum / total_positives

            precisions.append(precision)
            recalls.append(recall)

        # Compute AP using 11-point interpolation
        ap = 0.0
        for threshold in np.arange(0, 1.1, 0.1):
            precisions_above = [p for p, r in zip(precisions, recalls) if r >= threshold]
            if len(precisions_above) > 0:
                ap += max(precisions_above)

        return ap / 11.0

    def compute_map(self) -> float:
        """Compute mean Average Precision across all classes"""
        aps = []
        for class_id in range(self.num_classes):
            ap = self.compute_ap(class_id)
            aps.append(ap)
        return np.mean(aps) if aps else 0.0

    def get_metrics_summary(self) -> Dict[str, float]:
        """Get summary of all metrics"""
        # Per-class metrics
        per_class_metrics = {}
        for class_id in range(self.num_classes):
            precision, recall = self.compute_precision_recall(class_id)
            f1 = self.compute_f1_score(class_id)
            ap = self.compute_ap(class_id)

            per_class_metrics[f'class_{class_id}'] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'ap': ap,
                'tp': self.tp[class_id],
                'fp': self.fp[class_id],
                'fn': self.fn[class_id]
            }

        # Overall metrics
        all_tp = sum(self.tp.values())
        all_fp = sum(self.fp.values())
        all_fn = sum(self.fn.values())

        overall_precision = all_tp / (all_tp + all_fp + 1e-6)
        overall_recall = all_tp / (all_tp + all_fn + 1e-6)
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall + 1e-6)

        summary = {
            'mAP': self.compute_map(),
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'total_tp': all_tp,
            'total_fp': all_fp,
            'total_fn': all_fn,
            'per_class': per_class_metrics
        }

        return summary


class SpeedMetrics:
    """Measure inference speed metrics"""

    def __init__(self):
        self.inference_times = []

    def reset(self):
        """Reset metrics"""
        self.inference_times = []

    def update(self, inference_time: float):
        """
        Add inference time measurement

        Args:
            inference_time: Time in seconds
        """
        self.inference_times.append(inference_time)

    def get_fps(self) -> float:
        """Get average FPS"""
        if not self.inference_times:
            return 0.0
        avg_time = np.mean(self.inference_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def get_avg_inference_time(self) -> float:
        """Get average inference time in milliseconds"""
        if not self.inference_times:
            return 0.0
        return np.mean(self.inference_times) * 1000  # Convert to ms

    def get_summary(self) -> Dict[str, float]:
        """Get speed metrics summary"""
        if not self.inference_times:
            return {
                'avg_fps': 0.0,
                'avg_inference_time_ms': 0.0,
                'min_inference_time_ms': 0.0,
                'max_inference_time_ms': 0.0
            }

        times_ms = np.array(self.inference_times) * 1000

        return {
            'avg_fps': self.get_fps(),
            'avg_inference_time_ms': np.mean(times_ms),
            'min_inference_time_ms': np.min(times_ms),
            'max_inference_time_ms': np.max(times_ms),
            'std_inference_time_ms': np.std(times_ms)
        }


def calculate_confusion_matrix(
    pred_classes: np.ndarray,
    gt_classes: np.ndarray,
    num_classes: int
) -> np.ndarray:
    """
    Calculate confusion matrix

    Args:
        pred_classes: Predicted classes
        gt_classes: Ground truth classes
        num_classes: Number of classes

    Returns:
        Confusion matrix [num_classes, num_classes]
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)

    for pred, gt in zip(pred_classes, gt_classes):
        cm[gt, pred] += 1

    return cm


if __name__ == "__main__":
    # Example usage
    metrics = DetectionMetrics(num_classes=3, iou_threshold=0.5)

    # Dummy predictions and ground truths
    pred_boxes = np.array([[100, 100, 200, 200], [300, 300, 400, 400]])
    pred_classes = np.array([0, 1])
    pred_scores = np.array([0.9, 0.85])

    gt_boxes = np.array([[105, 105, 205, 205], [310, 310, 410, 410]])
    gt_classes = np.array([0, 1])

    # Update metrics
    metrics.update(pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes)

    # Get summary
    summary = metrics.get_metrics_summary()
    print("Metrics Summary:")
    print(f"mAP: {summary['mAP']:.3f}")
    print(f"Precision: {summary['precision']:.3f}")
    print(f"Recall: {summary['recall']:.3f}")
    print(f"F1 Score: {summary['f1_score']:.3f}")
