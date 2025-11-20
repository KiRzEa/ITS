"""
YOLOv11 Traffic Sign Detection Trainer
Modern deep learning approach using Ultralytics YOLO
"""

from ultralytics import YOLO
import torch
from pathlib import Path
from typing import Optional, Dict, List
import yaml
import time


class YOLOTrainer:
    """YOLO model trainer for traffic sign detection"""

    def __init__(
        self,
        model_size: str = 'n',  # n, s, m, l, x
        img_size: int = 640,
        device: str = 'auto'
    ):
        """
        Initialize YOLO trainer

        Args:
            model_size: Model size (n=nano, s=small, m=medium, l=large, x=xlarge)
            img_size: Input image size
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.model_size = model_size
        self.img_size = img_size

        # Determine device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Initialize model
        model_name = f'yolo11{model_size}.pt'
        self.model = YOLO(model_name)

        print(f"Initialized YOLOv11{model_size.upper()} on {self.device}")
        print(f"Image size: {img_size}")

    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        batch_size: int = 16,
        patience: int = 50,
        save_dir: str = 'experiments/yolo',
        name: str = 'traffic_sign_detection',
        resume: bool = False,
        **kwargs
    ) -> Dict:
        """
        Train YOLO model

        Args:
            data_yaml: Path to data.yaml file
            epochs: Number of training epochs
            batch_size: Batch size
            patience: Early stopping patience
            save_dir: Directory to save results
            name: Experiment name
            resume: Resume from last checkpoint
            **kwargs: Additional training arguments

        Returns:
            Training results dictionary
        """
        print(f"\n{'='*50}")
        print("Starting YOLO Training")
        print(f"{'='*50}")
        print(f"Model: YOLOv11{self.model_size.upper()}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Device: {self.device}")
        print(f"Data config: {data_yaml}")
        print(f"{'='*50}\n")

        # Training arguments
        train_args = {
            'data': data_yaml,
            'epochs': epochs,
            'imgsz': self.img_size,
            'batch': batch_size,
            'device': self.device,
            'patience': patience,
            'save': True,
            'save_period': 10,
            'project': save_dir,
            'name': name,
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'auto',
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': True,
            'close_mosaic': 10,
            'resume': resume,
            'amp': True,  # Automatic Mixed Precision
            'fraction': 1.0,
            'profile': False,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'plots': True,
        }

        # Update with custom arguments
        train_args.update(kwargs)

        # Start training
        start_time = time.time()
        results = self.model.train(**train_args)
        training_time = time.time() - start_time

        print(f"\n{'='*50}")
        print("Training Completed!")
        print(f"Total time: {training_time/60:.2f} minutes")
        print(f"{'='*50}\n")

        return results

    def validate(
        self,
        data_yaml: Optional[str] = None,
        split: str = 'val',
        **kwargs
    ) -> Dict:
        """
        Validate model

        Args:
            data_yaml: Path to data.yaml (optional if already trained)
            split: Dataset split ('val' or 'test')
            **kwargs: Additional validation arguments

        Returns:
            Validation results dictionary
        """
        print(f"\n{'='*50}")
        print(f"Running Validation on {split} set")
        print(f"{'='*50}\n")

        val_args = {
            'data': data_yaml,
            'split': split,
            'imgsz': self.img_size,
            'device': self.device,
            'batch': 16,
            'verbose': True,
            'save_json': True,
            'save_hybrid': True,
            'plots': True,
        }

        val_args.update(kwargs)

        results = self.model.val(**val_args)

        # Print metrics
        print(f"\n{'='*50}")
        print("Validation Results")
        print(f"{'='*50}")
        print(f"mAP50: {results.box.map50:.4f}")
        print(f"mAP50-95: {results.box.map:.4f}")
        print(f"Precision: {results.box.mp:.4f}")
        print(f"Recall: {results.box.mr:.4f}")
        print(f"{'='*50}\n")

        return results

    def predict(
        self,
        source: str,
        conf: float = 0.25,
        iou: float = 0.7,
        save: bool = True,
        save_txt: bool = False,
        save_conf: bool = False,
        **kwargs
    ):
        """
        Run prediction

        Args:
            source: Image source (path, url, or directory)
            conf: Confidence threshold
            iou: IoU threshold for NMS
            save: Save results
            save_txt: Save results as txt
            save_conf: Save confidences in txt
            **kwargs: Additional prediction arguments

        Returns:
            Prediction results
        """
        predict_args = {
            'source': source,
            'imgsz': self.img_size,
            'conf': conf,
            'iou': iou,
            'device': self.device,
            'save': save,
            'save_txt': save_txt,
            'save_conf': save_conf,
            'verbose': True,
            'visualize': False,
            'augment': False,
            'agnostic_nms': False,
            'retina_masks': False,
        }

        predict_args.update(kwargs)

        results = self.model.predict(**predict_args)

        return results

    def export(
        self,
        format: str = 'onnx',
        **kwargs
    ) -> str:
        """
        Export model to different formats

        Args:
            format: Export format ('onnx', 'torchscript', 'tflite', etc.)
            **kwargs: Additional export arguments

        Returns:
            Path to exported model
        """
        print(f"Exporting model to {format.upper()} format...")

        export_args = {
            'format': format,
            'imgsz': self.img_size,
            'keras': False,
            'optimize': False,
            'half': False,
            'int8': False,
            'dynamic': False,
            'simplify': False,
            'opset': None,
            'workspace': 4,
            'nms': False,
        }

        export_args.update(kwargs)

        export_path = self.model.export(**export_args)

        print(f"Model exported to: {export_path}")

        return export_path

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        self.model = YOLO(checkpoint_path)
        print(f"Loaded checkpoint from: {checkpoint_path}")

    def get_model_info(self):
        """Print model information"""
        self.model.info(verbose=True)


class YOLOEnsemble:
    """Ensemble multiple YOLO models for improved performance"""

    def __init__(self, model_paths: List[str], device: str = 'auto'):
        """
        Initialize ensemble

        Args:
            model_paths: List of paths to trained models
            device: Device to use
        """
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.models = [YOLO(path) for path in model_paths]
        print(f"Loaded {len(self.models)} models for ensemble")

    def predict(
        self,
        source: str,
        conf: float = 0.25,
        iou: float = 0.7,
        **kwargs
    ):
        """
        Predict using ensemble

        Args:
            source: Image source
            conf: Confidence threshold
            iou: IoU threshold for NMS
            **kwargs: Additional arguments

        Returns:
            Ensemble prediction results
        """
        all_results = []

        for model in self.models:
            results = model.predict(
                source=source,
                conf=conf,
                iou=iou,
                device=self.device,
                verbose=False,
                **kwargs
            )
            all_results.append(results)

        # Combine predictions (simple averaging)
        # You can implement more sophisticated ensemble methods here

        return all_results


def create_data_yaml(
    train_path: str,
    val_path: str,
    test_path: str,
    class_names: List[str],
    output_path: str
):
    """
    Create data.yaml file for YOLO training

    Args:
        train_path: Path to training images
        val_path: Path to validation images
        test_path: Path to test images
        class_names: List of class names
        output_path: Output path for data.yaml
    """
    data = {
        'path': str(Path(train_path).parent.parent),
        'train': str(Path(train_path).relative_to(Path(train_path).parent.parent)),
        'val': str(Path(val_path).relative_to(Path(val_path).parent.parent)),
        'test': str(Path(test_path).relative_to(Path(test_path).parent.parent)),
        'nc': len(class_names),
        'names': class_names
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"Created data.yaml at: {output_path}")


if __name__ == "__main__":
    # Example usage
    print("YOLOv11 Trainer Example")

    # Create trainer
    trainer = YOLOTrainer(model_size='n', img_size=640)

    # Print model info
    print("\nModel Information:")
    trainer.get_model_info()

    print("\nTrainer initialized successfully!")
    print("To train: trainer.train(data_yaml='path/to/data.yaml', epochs=100)")
    print("To validate: trainer.validate()")
    print("To predict: trainer.predict(source='path/to/image.jpg')")
