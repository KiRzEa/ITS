"""
Visualization utilities for traffic sign detection
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import pandas as pd


class BoundingBoxVisualizer:
    """Visualize bounding boxes on images"""

    def __init__(self, class_names: List[str], colors: Optional[List[Tuple[int, int, int]]] = None):
        """
        Initialize visualizer

        Args:
            class_names: List of class names
            colors: List of BGR colors for each class (optional)
        """
        self.class_names = class_names
        self.num_classes = len(class_names)

        if colors is None:
            # Generate random colors for each class
            np.random.seed(42)
            self.colors = [
                tuple(map(int, np.random.randint(0, 255, 3)))
                for _ in range(self.num_classes)
            ]
        else:
            self.colors = colors

    def draw_box(
        self,
        image: np.ndarray,
        box: Tuple[int, int, int, int],
        class_id: int,
        confidence: float = 1.0,
        label: Optional[str] = None,
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw single bounding box on image

        Args:
            image: Input image
            box: Bounding box (x1, y1, x2, y2)
            class_id: Class ID
            confidence: Confidence score
            label: Custom label (optional)
            thickness: Box thickness

        Returns:
            Image with drawn box
        """
        x1, y1, x2, y2 = map(int, box)
        color = self.colors[class_id % len(self.colors)]

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Prepare label
        if label is None:
            label = f"{self.class_names[class_id]}: {confidence:.2f}"

        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_w, label_h = label_size

        # Ensure label is within image bounds
        y1_label = max(y1 - label_h - 10, label_h + 5)

        cv2.rectangle(
            image,
            (x1, y1_label - label_h - 5),
            (x1 + label_w + 5, y1_label + 5),
            color,
            -1
        )

        # Draw label text
        cv2.putText(
            image,
            label,
            (x1 + 2, y1_label),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        return image

    def draw_boxes(
        self,
        image: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
        class_ids: List[int],
        confidences: Optional[List[float]] = None,
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw multiple bounding boxes on image

        Args:
            image: Input image
            boxes: List of bounding boxes
            class_ids: List of class IDs
            confidences: List of confidence scores (optional)
            thickness: Box thickness

        Returns:
            Image with drawn boxes
        """
        result = image.copy()

        if confidences is None:
            confidences = [1.0] * len(boxes)

        for box, class_id, conf in zip(boxes, class_ids, confidences):
            result = self.draw_box(result, box, class_id, conf, thickness=thickness)

        return result

    def draw_yolo_boxes(
        self,
        image: np.ndarray,
        yolo_boxes: List[Tuple[float, float, float, float]],
        class_ids: List[int],
        confidences: Optional[List[float]] = None,
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw bounding boxes from YOLO format (normalized cx, cy, w, h)

        Args:
            image: Input image
            yolo_boxes: List of YOLO format boxes (cx, cy, w, h)
            class_ids: List of class IDs
            confidences: List of confidence scores (optional)
            thickness: Box thickness

        Returns:
            Image with drawn boxes
        """
        h, w = image.shape[:2]

        # Convert YOLO to corner format
        corner_boxes = []
        for cx, cy, bw, bh in yolo_boxes:
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            corner_boxes.append((x1, y1, x2, y2))

        return self.draw_boxes(image, corner_boxes, class_ids, confidences, thickness)


class ResultVisualizer:
    """Visualize detection results and metrics"""

    @staticmethod
    def plot_confusion_matrix(
        cm: np.ndarray,
        class_names: List[str],
        title: str = "Confusion Matrix",
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None
    ):
        """Plot confusion matrix"""
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_metrics_comparison(
        results: Dict[str, Dict[str, float]],
        metrics: List[str] = ['mAP', 'Precision', 'Recall', 'F1'],
        title: str = "Model Comparison",
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ):
        """
        Plot comparison of different models' metrics

        Args:
            results: Dictionary of {model_name: {metric_name: value}}
            metrics: List of metrics to plot
            title: Plot title
            figsize: Figure size
            save_path: Path to save plot (optional)
        """
        df = pd.DataFrame(results).T
        df = df[metrics]

        fig, ax = plt.subplots(figsize=figsize)
        df.plot(kind='bar', ax=ax, rot=45)

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_pr_curve(
        precisions: np.ndarray,
        recalls: np.ndarray,
        ap: float,
        class_name: str = "",
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ):
        """Plot Precision-Recall curve"""
        plt.figure(figsize=figsize)
        plt.plot(recalls, precisions, linewidth=2, label=f'AP = {ap:.3f}')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {class_name}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_training_history(
        history: Dict[str, List[float]],
        figsize: Tuple[int, int] = (15, 5),
        save_path: Optional[str] = None
    ):
        """
        Plot training history (loss, metrics over epochs)

        Args:
            history: Dictionary with training history
            figsize: Figure size
            save_path: Path to save plot (optional)
        """
        metrics = list(history.keys())
        n_metrics = len(metrics)

        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            values = history[metric]
            epochs = range(1, len(values) + 1)

            ax.plot(epochs, values, linewidth=2, marker='o', markersize=4)
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel(metric, fontsize=10)
            ax.set_title(f'{metric} vs Epoch', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_class_distribution(
        class_counts: Dict[str, int],
        title: str = "Class Distribution",
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ):
        """Plot class distribution"""
        classes = list(class_counts.keys())
        counts = list(class_counts.values())

        plt.figure(figsize=figsize)
        plt.bar(classes, counts, color='steelblue', edgecolor='black', alpha=0.7)
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)

        # Add count labels on bars
        for i, (cls, count) in enumerate(zip(classes, counts)):
            plt.text(i, count, str(count), ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def visualize_detections_grid(
        images: List[np.ndarray],
        titles: List[str],
        grid_size: Optional[Tuple[int, int]] = None,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None
    ):
        """
        Visualize multiple detection results in a grid

        Args:
            images: List of images with drawn detections
            titles: List of titles for each image
            grid_size: Grid size (rows, cols). Auto-calculated if None
            figsize: Figure size
            save_path: Path to save plot (optional)
        """
        n_images = len(images)

        if grid_size is None:
            cols = min(3, n_images)
            rows = (n_images + cols - 1) // cols
        else:
            rows, cols = grid_size

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if n_images > 1 else [axes]

        for idx, (img, title) in enumerate(zip(images, titles)):
            if idx < len(axes):
                # Convert BGR to RGB for matplotlib
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[idx].imshow(img_rgb)
                axes[idx].set_title(title, fontsize=10, fontweight='bold')
                axes[idx].axis('off')

        # Hide unused subplots
        for idx in range(n_images, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # Example usage
    class_names = ['Stop', 'Yield', 'Speed Limit 30', 'Speed Limit 50']

    # Create visualizer
    visualizer = BoundingBoxVisualizer(class_names)

    # Create dummy image
    image = np.ones((480, 640, 3), dtype=np.uint8) * 200

    # Draw some boxes
    boxes = [(100, 100, 200, 200), (300, 150, 400, 250)]
    class_ids = [0, 2]
    confidences = [0.95, 0.87]

    result = visualizer.draw_boxes(image, boxes, class_ids, confidences)

    # Display
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Detection Example')
    plt.show()
