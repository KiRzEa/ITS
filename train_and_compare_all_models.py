"""
Comprehensive Training and Comparison Script
Train all object detection approaches with various backbones and compare performance

Models trained:
- RetinaNet: ResNet50, ResNet34, MobileNetV3
- SSD: VGG16, MobileNetV3
- Faster R-CNN: ResNet50, ResNet34, MobileNetV3
- YOLO: v11n, v11s

All models trained for 20 epochs on the same dataset
"""

import sys
from pathlib import Path

import torch
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.modern.retinanet.trainer import RetinaNetTrainer, TrafficSignDataset as RetinaNetDataset
from src.modern.ssd.trainer import SSDTrainer, TrafficSignDataset as SSDDataset
from src.modern.faster_rcnn.trainer import FasterRCNNTrainer, TrafficSignDataset as FasterRCNNDataset
from src.modern.yolo.trainer import YOLOTrainer


class ModelBenchmark:
    """Benchmark and compare multiple object detection models"""

    def __init__(
        self,
        num_classes: int = 5,
        data_root: str = 'data/raw/yolov8',
        coco_annotations: str = 'data/processed/coco_format',
        experiments_dir: str = 'experiments/model_comparison',
        epochs: int = 20
    ):
        """
        Initialize benchmark

        Args:
            num_classes: Number of object classes
            data_root: Root directory for YOLO format data
            coco_annotations: Directory with COCO format annotations
            experiments_dir: Directory to save experiments
            epochs: Number of training epochs
        """
        self.num_classes = num_classes
        self.data_root = Path(data_root)
        self.coco_annotations = Path(coco_annotations)
        self.experiments_dir = Path(experiments_dir)
        self.epochs = epochs

        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.results = {}

        print(f"\n{'='*80}")
        print("MODEL BENCHMARK INITIALIZED")
        print(f"{'='*80}")
        print(f"Number of classes: {num_classes}")
        print(f"Training epochs: {epochs}")
        print(f"Data root: {data_root}")
        print(f"Results will be saved to: {experiments_dir}")
        print(f"{'='*80}\n")

    def train_retinanet_models(self):
        """Train RetinaNet with different backbones"""
        print(f"\n{'='*80}")
        print("TRAINING RETINANET MODELS")
        print(f"{'='*80}\n")

        backbones = [
            ('resnet50', 8, 0.001),      # (backbone, batch_size, lr)
            ('resnet34', 12, 0.001),
            ('mobilenet_v3', 16, 0.001)
        ]

        for backbone, batch_size, lr in backbones:
            model_name = f"RetinaNet-{backbone}"
            print(f"\n{'='*60}")
            print(f"Training {model_name}")
            print(f"{'='*60}\n")

            try:
                # Create datasets
                train_dataset = RetinaNetDataset(
                    images_dir=str(self.data_root / 'train' / 'images'),
                    annotations_file=str(self.coco_annotations / 'train_coco.json'),
                    filter_empty=True
                )

                val_dataset = RetinaNetDataset(
                    images_dir=str(self.data_root / 'valid' / 'images'),
                    annotations_file=str(self.coco_annotations / 'valid_coco.json'),
                    filter_empty=True
                )

                # Initialize trainer (num_classes + 1 for background)
                trainer = RetinaNetTrainer(
                    num_classes=self.num_classes + 1,
                    backbone=backbone,
                    pretrained=True,
                    device='auto'
                )

                # Train
                start_time = time.time()
                save_dir = self.experiments_dir / f'retinanet_{backbone}'

                history = trainer.train(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    epochs=self.epochs,
                    batch_size=batch_size,
                    learning_rate=lr,
                    save_dir=str(save_dir),
                    checkpoint_freq=10
                )

                training_time = time.time() - start_time

                # Validate
                print(f"\nValidating {model_name}...")
                metrics = trainer.validate(
                    val_dataset=val_dataset,
                    batch_size=batch_size,
                    conf_threshold=0.25,
                    save_results=True,
                    save_dir=str(save_dir / 'validation')
                )

                # Store results
                self.results[model_name] = {
                    'model_type': 'RetinaNet',
                    'backbone': backbone,
                    'batch_size': batch_size,
                    'learning_rate': lr,
                    'training_time': training_time,
                    'training_loss': history['train_loss'][-1],
                    'val_loss': history['val_loss'][-1],
                    'mAP@0.5': metrics['mAP@0.5'],
                    'mAP@0.5:0.95': metrics['mAP@0.5:0.95'],
                    'mAP@0.75': metrics['mAP@0.75'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score'],
                    'history': history,
                    'per_class': metrics['per_class']
                }

                print(f"✓ {model_name} completed successfully!")

            except Exception as e:
                print(f"✗ {model_name} failed: {str(e)}")
                self.results[model_name] = {'error': str(e)}

    def train_ssd_models(self):
        """Train SSD with different backbones"""
        print(f"\n{'='*80}")
        print("TRAINING SSD MODELS")
        print(f"{'='*80}\n")

        backbones = [
            ('vgg16', 12, 0.001),
            ('mobilenet_v3', 16, 0.001)
        ]

        for backbone, batch_size, lr in backbones:
            model_name = f"SSD-{backbone}"
            print(f"\n{'='*60}")
            print(f"Training {model_name}")
            print(f"{'='*60}\n")

            try:
                # Create datasets
                train_dataset = SSDDataset(
                    images_dir=str(self.data_root / 'train' / 'images'),
                    annotations_file=str(self.coco_annotations / 'train_coco.json'),
                    filter_empty=True
                )

                val_dataset = SSDDataset(
                    images_dir=str(self.data_root / 'valid' / 'images'),
                    annotations_file=str(self.coco_annotations / 'valid_coco.json'),
                    filter_empty=True
                )

                # Initialize trainer (num_classes + 1 for background)
                trainer = SSDTrainer(
                    num_classes=self.num_classes + 1,
                    backbone=backbone,
                    pretrained=True,
                    device='auto'
                )

                # Train
                start_time = time.time()
                save_dir = self.experiments_dir / f'ssd_{backbone}'

                history = trainer.train(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    epochs=self.epochs,
                    batch_size=batch_size,
                    learning_rate=lr,
                    save_dir=str(save_dir),
                    checkpoint_freq=10
                )

                training_time = time.time() - start_time

                # Validate
                print(f"\nValidating {model_name}...")
                metrics = trainer.validate(
                    val_dataset=val_dataset,
                    batch_size=batch_size,
                    conf_threshold=0.25,
                    save_results=True,
                    save_dir=str(save_dir / 'validation')
                )

                # Store results
                self.results[model_name] = {
                    'model_type': 'SSD',
                    'backbone': backbone,
                    'batch_size': batch_size,
                    'learning_rate': lr,
                    'training_time': training_time,
                    'training_loss': history['train_loss'][-1],
                    'val_loss': history['val_loss'][-1],
                    'mAP@0.5': metrics['mAP@0.5'],
                    'mAP@0.5:0.95': metrics['mAP@0.5:0.95'],
                    'mAP@0.75': metrics['mAP@0.75'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score'],
                    'history': history,
                    'per_class': metrics['per_class']
                }

                print(f"✓ {model_name} completed successfully!")

            except Exception as e:
                print(f"✗ {model_name} failed: {str(e)}")
                self.results[model_name] = {'error': str(e)}

    def train_faster_rcnn_models(self):
        """Train Faster R-CNN with different backbones"""
        print(f"\n{'='*80}")
        print("TRAINING FASTER R-CNN MODELS")
        print(f"{'='*80}\n")

        backbones = [
            ('resnet50', 8, 0.005),
            ('resnet34', 12, 0.005),
            ('mobilenet_v3_large', 16, 0.005)
        ]

        for backbone, batch_size, lr in backbones:
            model_name = f"FasterRCNN-{backbone}"
            print(f"\n{'='*60}")
            print(f"Training {model_name}")
            print(f"{'='*60}\n")

            try:
                # Create datasets
                train_dataset = FasterRCNNDataset(
                    images_dir=str(self.data_root / 'train' / 'images'),
                    annotations_file=str(self.coco_annotations / 'train_coco.json'),
                    filter_empty=True
                )

                val_dataset = FasterRCNNDataset(
                    images_dir=str(self.data_root / 'valid' / 'images'),
                    annotations_file=str(self.coco_annotations / 'valid_coco.json'),
                    filter_empty=True
                )

                # Initialize trainer (num_classes + 1 for background)
                trainer = FasterRCNNTrainer(
                    num_classes=self.num_classes + 1,
                    backbone=backbone,
                    pretrained=True,
                    device='auto'
                )

                # Train
                start_time = time.time()
                save_dir = self.experiments_dir / f'faster_rcnn_{backbone}'

                history = trainer.train(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    epochs=self.epochs,
                    batch_size=batch_size,
                    learning_rate=lr,
                    save_dir=str(save_dir),
                    checkpoint_freq=10
                )

                training_time = time.time() - start_time

                # Validate
                print(f"\nValidating {model_name}...")
                metrics = trainer.validate(
                    val_dataset=val_dataset,
                    batch_size=batch_size,
                    conf_threshold=0.25,
                    save_results=True,
                    save_dir=str(save_dir / 'validation')
                )

                # Store results
                self.results[model_name] = {
                    'model_type': 'Faster R-CNN',
                    'backbone': backbone,
                    'batch_size': batch_size,
                    'learning_rate': lr,
                    'training_time': training_time,
                    'training_loss': history['train_loss'][-1],
                    'val_loss': history['val_loss'][-1],
                    'mAP@0.5': metrics['mAP@0.5'],
                    'mAP@0.5:0.95': metrics['mAP@0.5:0.95'],
                    'mAP@0.75': metrics['mAP@0.75'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score'],
                    'history': history,
                    'per_class': metrics['per_class']
                }

                print(f"✓ {model_name} completed successfully!")

            except Exception as e:
                print(f"✗ {model_name} failed: {str(e)}")
                self.results[model_name] = {'error': str(e)}

    def train_yolo_models(self):
        """Train YOLO models"""
        print(f"\n{'='*80}")
        print("TRAINING YOLO MODELS")
        print(f"{'='*80}\n")

        models = [
            ('v11', 'n', 16),  # (version, size, batch_size)
            ('v11', 's', 8)
        ]

        data_yaml = str(self.data_root / 'data.yaml')

        for version, size, batch_size in models:
            model_name = f"YOLO{version}{size}"
            print(f"\n{'='*60}")
            print(f"Training {model_name}")
            print(f"{'='*60}\n")

            try:
                # Initialize trainer
                trainer = YOLOTrainer(
                    model_version=version,
                    model_size=size,
                    img_size=640,
                    device='auto'
                )

                # Train
                start_time = time.time()
                save_dir = self.experiments_dir / f'yolo_{version}{size}'

                results = trainer.train(
                    data_yaml=data_yaml,
                    epochs=self.epochs,
                    batch_size=batch_size,
                    patience=self.epochs,  # No early stopping
                    save_dir=str(save_dir),
                    name='training'
                )

                training_time = time.time() - start_time

                # Validate
                print(f"\nValidating {model_name}...")
                val_results = trainer.validate(
                    data_yaml=data_yaml,
                    split='val'
                )

                # Store results
                self.results[model_name] = {
                    'model_type': 'YOLO',
                    'version': version,
                    'size': size,
                    'batch_size': batch_size,
                    'training_time': training_time,
                    'mAP@0.5': float(val_results.box.map50),
                    'mAP@0.5:0.95': float(val_results.box.map),
                    'mAP@0.75': float(val_results.box.map75),
                    'precision': float(val_results.box.mp),
                    'recall': float(val_results.box.mr),
                    'f1_score': 2 * (float(val_results.box.mp) * float(val_results.box.mr)) /
                                (float(val_results.box.mp) + float(val_results.box.mr) + 1e-10)
                }

                print(f"✓ {model_name} completed successfully!")

            except Exception as e:
                print(f"✗ {model_name} failed: {str(e)}")
                self.results[model_name] = {'error': str(e)}

    def save_results(self):
        """Save all results to JSON"""
        results_file = self.experiments_dir / 'all_results.json'

        # Convert numpy types for JSON serialization
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

        serializable_results = convert_numpy(self.results)

        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\n✓ Results saved to: {results_file}")

    def create_comparison_plots(self):
        """Create comprehensive comparison visualizations"""
        print(f"\n{'='*80}")
        print("CREATING COMPARISON PLOTS")
        print(f"{'='*80}\n")

        # Filter out failed models
        valid_results = {k: v for k, v in self.results.items() if 'error' not in v}

        if not valid_results:
            print("⚠️  No valid results to plot")
            return

        # Prepare data for plotting
        models = list(valid_results.keys())

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))

        # 1. mAP Comparison
        ax1 = plt.subplot(2, 3, 1)
        map_50 = [valid_results[m]['mAP@0.5'] for m in models]
        map_50_95 = [valid_results[m]['mAP@0.5:0.95'] for m in models]

        x = np.arange(len(models))
        width = 0.35

        bars1 = ax1.bar(x - width/2, map_50, width, label='mAP@0.5', alpha=0.8)
        bars2 = ax1.bar(x + width/2, map_50_95, width, label='mAP@0.5:0.95', alpha=0.8)

        ax1.set_ylabel('mAP')
        ax1.set_title('Mean Average Precision Comparison', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0, 1])

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)

        # 2. Precision vs Recall
        ax2 = plt.subplot(2, 3, 2)
        precisions = [valid_results[m]['precision'] for m in models]
        recalls = [valid_results[m]['recall'] for m in models]

        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        for i, (model, prec, rec, color) in enumerate(zip(models, precisions, recalls, colors)):
            ax2.scatter(rec, prec, s=200, alpha=0.6, color=color, label=model)

        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision vs Recall', fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])

        # 3. F1 Score Comparison
        ax3 = plt.subplot(2, 3, 3)
        f1_scores = [valid_results[m]['f1_score'] for m in models]

        bars = ax3.barh(models, f1_scores, alpha=0.8, color=colors)
        ax3.set_xlabel('F1 Score')
        ax3.set_title('F1 Score Comparison', fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        ax3.set_xlim([0, 1])

        for i, (bar, f1) in enumerate(zip(bars, f1_scores)):
            ax3.text(f1, i, f' {f1:.3f}', va='center', fontsize=9)

        # 4. Training Time Comparison
        ax4 = plt.subplot(2, 3, 4)
        training_times = [valid_results[m]['training_time'] / 60 for m in models]  # Convert to minutes

        bars = ax4.bar(models, training_times, alpha=0.8, color=colors)
        ax4.set_ylabel('Training Time (minutes)')
        ax4.set_title('Training Time Comparison', fontweight='bold')
        ax4.set_xticklabels(models, rotation=45, ha='right')
        ax4.grid(axis='y', alpha=0.3)

        for bar, time in zip(bars, training_times):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.1f}m', ha='center', va='bottom', fontsize=9)

        # 5. Model Type Grouped Performance
        ax5 = plt.subplot(2, 3, 5)

        # Group by model type
        model_types = {}
        for model, data in valid_results.items():
            model_type = data.get('model_type', 'Unknown')
            if model_type not in model_types:
                model_types[model_type] = []
            model_types[model_type].append(data['mAP@0.5'])

        type_names = list(model_types.keys())
        type_means = [np.mean(model_types[t]) for t in type_names]
        type_stds = [np.std(model_types[t]) if len(model_types[t]) > 1 else 0 for t in type_names]

        bars = ax5.bar(type_names, type_means, yerr=type_stds, alpha=0.8,
                      capsize=10, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax5.set_ylabel('mAP@0.5')
        ax5.set_title('Average Performance by Model Type', fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)
        ax5.set_ylim([0, 1])

        for bar, mean in zip(bars, type_means):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 6. Summary Table
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')

        # Create summary table
        table_data = []
        for model in models:
            data = valid_results[model]
            table_data.append([
                model,
                f"{data['mAP@0.5']:.3f}",
                f"{data['precision']:.3f}",
                f"{data['recall']:.3f}",
                f"{data['f1_score']:.3f}"
            ])

        table = ax6.table(
            cellText=table_data,
            colLabels=['Model', 'mAP@0.5', 'Precision', 'Recall', 'F1'],
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header row
        for i in range(5):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            for j in range(5):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')

        ax6.set_title('Performance Summary', fontweight='bold', pad=20)

        plt.tight_layout()

        # Save figure
        plot_file = self.experiments_dir / 'model_comparison.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved to: {plot_file}")

        plt.show()

        # Create training curves comparison
        self._plot_training_curves(valid_results)

    def _plot_training_curves(self, valid_results):
        """Plot training curves for models with history"""
        models_with_history = {k: v for k, v in valid_results.items()
                               if 'history' in v and 'train_loss' in v['history']}

        if not models_with_history:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        colors = plt.cm.tab10(np.linspace(0, 1, len(models_with_history)))

        for (model, data), color in zip(models_with_history.items(), colors):
            history = data['history']
            epochs = range(1, len(history['train_loss']) + 1)

            # Training loss
            ax1.plot(epochs, history['train_loss'], label=model,
                    color=color, linewidth=2, alpha=0.7)

            # Validation loss
            ax2.plot(epochs, history['val_loss'], label=model,
                    color=color, linewidth=2, alpha=0.7)

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Curves', fontweight='bold', fontsize=14)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Validation Loss Curves', fontweight='bold', fontsize=14)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        curves_file = self.experiments_dir / 'training_curves.png'
        plt.savefig(curves_file, dpi=300, bbox_inches='tight')
        print(f"✓ Training curves saved to: {curves_file}")

        plt.show()

    def generate_report(self):
        """Generate a text report summarizing results"""
        report_file = self.experiments_dir / 'comparison_report.txt'

        valid_results = {k: v for k, v in self.results.items() if 'error' not in v}

        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("OBJECT DETECTION MODEL COMPARISON REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

            f.write(f"Total models trained: {len(self.results)}\n")
            f.write(f"Successful: {len(valid_results)}\n")
            f.write(f"Failed: {len(self.results) - len(valid_results)}\n")
            f.write(f"Training epochs: {self.epochs}\n\n")

            f.write("="*80 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("="*80 + "\n\n")

            for model, data in sorted(valid_results.items(),
                                     key=lambda x: x[1]['mAP@0.5'],
                                     reverse=True):
                f.write(f"\n{model}\n")
                f.write("-" * len(model) + "\n")
                f.write(f"  Model Type:      {data.get('model_type', 'N/A')}\n")
                f.write(f"  Backbone:        {data.get('backbone', data.get('size', 'N/A'))}\n")
                f.write(f"  Batch Size:      {data.get('batch_size', 'N/A')}\n")
                f.write(f"  Training Time:   {data['training_time']/60:.2f} minutes\n")
                f.write(f"\n  Performance Metrics:\n")
                f.write(f"    mAP@0.5:       {data['mAP@0.5']:.4f}\n")
                f.write(f"    mAP@0.5:0.95:  {data['mAP@0.5:0.95']:.4f}\n")
                f.write(f"    mAP@0.75:      {data['mAP@0.75']:.4f}\n")
                f.write(f"    Precision:     {data['precision']:.4f}\n")
                f.write(f"    Recall:        {data['recall']:.4f}\n")
                f.write(f"    F1 Score:      {data['f1_score']:.4f}\n")

            # Add rankings
            f.write("\n\n" + "="*80 + "\n")
            f.write("RANKINGS\n")
            f.write("="*80 + "\n\n")

            metrics_to_rank = ['mAP@0.5', 'mAP@0.5:0.95', 'precision', 'recall', 'f1_score']

            for metric in metrics_to_rank:
                f.write(f"\nTop 3 by {metric}:\n")
                ranked = sorted(valid_results.items(),
                              key=lambda x: x[1][metric],
                              reverse=True)[:3]
                for i, (model, data) in enumerate(ranked, 1):
                    f.write(f"  {i}. {model}: {data[metric]:.4f}\n")

            # Failed models
            if len(self.results) > len(valid_results):
                f.write("\n\n" + "="*80 + "\n")
                f.write("FAILED MODELS\n")
                f.write("="*80 + "\n\n")

                for model, data in self.results.items():
                    if 'error' in data:
                        f.write(f"\n{model}:\n")
                        f.write(f"  Error: {data['error']}\n")

        print(f"✓ Report saved to: {report_file}")

    def run_full_benchmark(self):
        """Run complete benchmark pipeline"""
        print(f"\n{'='*80}")
        print("STARTING FULL BENCHMARK")
        print(f"{'='*80}\n")

        start_time = time.time()

        # Train all models
        self.train_retinanet_models()
        self.train_ssd_models()
        self.train_faster_rcnn_models()
        self.train_yolo_models()

        # Save results
        self.save_results()

        # Create visualizations
        self.create_comparison_plots()

        # Generate report
        self.generate_report()

        total_time = time.time() - start_time

        print(f"\n{'='*80}")
        print("BENCHMARK COMPLETED!")
        print(f"{'='*80}")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Results saved to: {self.experiments_dir}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train and compare all object detection models')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--num-classes', type=int, default=5, help='Number of object classes')
    parser.add_argument('--data-root', type=str, default='data/raw/yolov8',
                       help='Root directory for data')
    parser.add_argument('--coco-annotations', type=str, default='data/processed/coco_format',
                       help='Directory with COCO format annotations')
    parser.add_argument('--experiments-dir', type=str, default='experiments/model_comparison',
                       help='Directory to save results')
    parser.add_argument('--models', type=str, nargs='+',
                       choices=['retinanet', 'ssd', 'faster_rcnn', 'yolo', 'all'],
                       default=['all'],
                       help='Which models to train')

    args = parser.parse_args()

    # Initialize benchmark
    benchmark = ModelBenchmark(
        num_classes=args.num_classes,
        data_root=args.data_root,
        coco_annotations=args.coco_annotations,
        experiments_dir=args.experiments_dir,
        epochs=args.epochs
    )

    # Train selected models
    if 'all' in args.models:
        benchmark.run_full_benchmark()
    else:
        if 'retinanet' in args.models:
            benchmark.train_retinanet_models()
        if 'ssd' in args.models:
            benchmark.train_ssd_models()
        if 'faster_rcnn' in args.models:
            benchmark.train_faster_rcnn_models()
        if 'yolo' in args.models:
            benchmark.train_yolo_models()

        # Save and visualize
        benchmark.save_results()
        benchmark.create_comparison_plots()
        benchmark.generate_report()
