"""
RetinaNet Traffic Sign Detection Trainer
Single-stage object detection with Focal Loss using PyTorch and torchvision
"""

import torch
import torchvision
from torchvision.models.detection import (
    retinanet_resnet50_fpn_v2,
    RetinaNet_ResNet50_FPN_V2_Weights
)
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, mobilenet_backbone
from torchvision.models.detection import RetinaNet
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from pathlib import Path
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm
import cv2
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.detection_validator import validate_model, save_metrics


class TrafficSignDataset(Dataset):
    """Dataset for traffic sign detection in COCO format"""

    def __init__(
        self,
        images_dir: str,
        annotations_file: str,
        transforms=None,
        filter_empty: bool = True
    ):
        """
        Initialize dataset

        Args:
            images_dir: Directory containing images
            annotations_file: Path to COCO format annotations JSON
            transforms: Image transforms
            filter_empty: If True, filter out images with no annotations
        """
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        self.filter_empty = filter_empty

        # Load annotations
        with open(annotations_file, 'r') as f:
            self.coco_data = json.load(f)

        self.annotations = self.coco_data['annotations']
        self.categories = self.coco_data['categories']

        # Create image_id to annotations mapping
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        # Filter out images with no annotations if requested
        if self.filter_empty:
            self.images = [
                img for img in self.coco_data['images']
                if img['id'] in self.img_to_anns and len(self.img_to_anns[img['id']]) > 0
            ]
            print(f"Filtered dataset: {len(self.images)}/{len(self.coco_data['images'])} images with annotations")
        else:
            self.images = self.coco_data['images']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get image info
        img_info = self.images[idx]
        img_id = img_info['id']
        img_path = self.images_dir / img_info['file_name']

        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get annotations for this image
        anns = self.img_to_anns.get(img_id, [])

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            # COCO format: [x, y, width, height]
            x, y, w, h = ann['bbox']

            # Skip invalid boxes
            if w <= 0 or h <= 0:
                continue

            # Convert to [x1, y1, x2, y2]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            areas.append(ann.get('area', w * h))
            iscrowd.append(ann.get('iscrowd', 0))

        # Handle case where no valid boxes exist
        if len(boxes) == 0:
            boxes = [[0, 0, 1, 1]]
            labels = [0]
            areas = [1.0]
            iscrowd = [1]

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([img_id])

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': areas,
            'iscrowd': iscrowd
        }

        # Apply transforms
        if self.transforms:
            image = self.transforms(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, target


class RetinaNetTrainer:
    """RetinaNet trainer for traffic sign detection"""

    def __init__(
        self,
        num_classes: int,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        device: str = 'auto'
    ):
        """
        Initialize RetinaNet trainer

        Args:
            num_classes: Number of classes (including background)
            backbone: Backbone architecture. Options:
                - 'resnet50': ResNet50 FPN (default, ~160M params)
                - 'resnet18': ResNet18 FPN (lightweight, ~21M params)
                - 'resnet34': ResNet34 FPN (lightweight, ~33M params)
                - 'mobilenet_v3': MobileNetV3 FPN (very lightweight, ~5M params)
            pretrained: Use pretrained weights
            device: Device to use
        """
        self.num_classes = num_classes
        self.backbone = backbone

        # Determine device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load model based on backbone choice
        self.model = self._create_model(backbone, pretrained)

        # Replace classification head for custom number of classes
        self._replace_head(num_classes)

        self.model.to(self.device)

        print(f"Initialized RetinaNet with {backbone} backbone")
        print(f"Number of classes: {num_classes}")
        print(f"Device: {self.device}")

    def _create_model(self, backbone: str, pretrained: bool):
        """
        Create RetinaNet model with specified backbone

        Args:
            backbone: Backbone name
            pretrained: Use pretrained weights

        Returns:
            RetinaNet model
        """
        if backbone == 'resnet50':
            if pretrained:
                weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
                model = retinanet_resnet50_fpn_v2(weights=weights)
            else:
                model = retinanet_resnet50_fpn_v2(weights=None)

        elif backbone == 'resnet18':
            # Create ResNet18 FPN backbone
            backbone_model = resnet_fpn_backbone('resnet18', pretrained=pretrained)
            model = RetinaNet(backbone_model, num_classes=91)  # Will replace head later

        elif backbone == 'resnet34':
            # Create ResNet34 FPN backbone
            backbone_model = resnet_fpn_backbone('resnet34', pretrained=pretrained)
            model = RetinaNet(backbone_model, num_classes=91)

        elif backbone == 'mobilenet_v3':
            # Create MobileNetV3 backbone
            from torchvision.models.detection.anchor_utils import AnchorGenerator

            backbone_model = mobilenet_backbone(
                'mobilenet_v3_large',
                pretrained=pretrained,
                fpn=True,
                returned_layers=[3, 4],
                extra_blocks=None
            )

            # Custom anchor generator for MobileNet
            anchor_generator = AnchorGenerator(
                sizes=tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512]),
                aspect_ratios=((0.5, 1.0, 2.0),) * 5
            )

            model = RetinaNet(
                backbone_model,
                num_classes=91,
                anchor_generator=anchor_generator
            )

        else:
            raise ValueError(
                f"Unknown backbone: {backbone}. "
                f"Supported: resnet50, resnet18, resnet34, mobilenet_v3"
            )

        return model

    def _replace_head(self, num_classes: int):
        """Replace RetinaNet classification head for custom number of classes"""
        # Get number of anchors
        num_anchors = self.model.head.classification_head.num_anchors

        # Get in_channels from the first layer of conv
        # Handle both sequential and Conv2dNormActivation cases
        first_conv = self.model.head.classification_head.conv
        if hasattr(first_conv, '__getitem__'):
            in_channels = first_conv[0].in_channels
        elif hasattr(first_conv, '0'):
            in_channels = getattr(first_conv, '0').in_channels
        else:
            # For Conv2dNormActivation, access the first conv layer
            in_channels = list(first_conv.children())[0].in_channels

        # Create new classification head
        self.model.head.classification_head = RetinaNetClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes
        )

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        epochs: int = 50,
        batch_size: int = 4,
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
        save_dir: str = 'experiments/retinanet',
        checkpoint_freq: int = 5,
        **kwargs
    ) -> Dict:
        """
        Train RetinaNet model

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            momentum: SGD momentum
            weight_decay: Weight decay
            save_dir: Directory to save checkpoints
            checkpoint_freq: Save checkpoint every N epochs
            **kwargs: Additional arguments

        Returns:
            Training history
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=self.collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=self.collate_fn
        )

        # Optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim.SGD(
            params,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1
        )

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

        print(f"\n{'='*50}")
        print("Starting RetinaNet Training")
        print(f"{'='*50}")
        print(f"Backbone: {self.backbone}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Device: {self.device}")
        print(f"{'='*50}\n")

        best_val_loss = float('inf')
        start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()

            # Train
            train_loss = self._train_epoch(train_loader, optimizer)

            # Validate
            val_loss = self._validate_epoch(val_loader)

            # Update learning rate
            lr_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['learning_rate'].append(current_lr)

            # Print progress
            epoch_time = time.time() - epoch_start
            print(f"Epoch [{epoch+1}/{epochs}] - "
                  f"Train Loss: {train_loss:.4f} - "
                  f"Val Loss: {val_loss:.4f} - "
                  f"LR: {current_lr:.6f} - "
                  f"Time: {epoch_time:.2f}s")

            # Save checkpoint
            if (epoch + 1) % checkpoint_freq == 0:
                checkpoint_path = save_dir / f"checkpoint_epoch_{epoch+1}.pth"
                self.save_checkpoint(checkpoint_path, epoch, optimizer, history)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = save_dir / "best_model.pth"
                self.save_checkpoint(best_model_path, epoch, optimizer, history)
                print(f"  → Saved best model (val_loss: {val_loss:.4f})")

        training_time = time.time() - start_time

        print(f"\n{'='*50}")
        print("Training Completed!")
        print(f"Total time: {training_time/60:.2f} minutes")
        print(f"Best val loss: {best_val_loss:.4f}")
        print(f"{'='*50}\n")

        # Save final model
        final_model_path = save_dir / "final_model.pth"
        self.save_checkpoint(final_model_path, epochs, optimizer, history)

        # Save training history
        history_path = save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        return history

    def _train_epoch(self, data_loader, optimizer):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(data_loader, desc="Training")

        for images, targets in progress_bar:
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

            # Update progress bar
            progress_bar.set_postfix({'loss': losses.item()})

        return epoch_loss / len(data_loader)

    def _validate_epoch(self, data_loader):
        """Validate for one epoch"""
        self.model.train()  # Keep in train mode to get losses
        epoch_loss = 0.0

        with torch.no_grad():
            for images, targets in tqdm(data_loader, desc="Validation"):
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # Forward pass
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                epoch_loss += losses.item()

        return epoch_loss / len(data_loader)

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for data loader"""
        return tuple(zip(*batch))

    def predict(
        self,
        images: List[torch.Tensor],
        confidence_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Run prediction

        Args:
            images: List of image tensors
            confidence_threshold: Confidence threshold

        Returns:
            List of predictions
        """
        self.model.eval()

        with torch.no_grad():
            images = [img.to(self.device) for img in images]
            predictions = self.model(images)

        # Filter by confidence
        filtered_predictions = []
        for pred in predictions:
            scores = pred['scores'].cpu().numpy()
            keep = scores >= confidence_threshold

            filtered_pred = {
                'boxes': pred['boxes'][keep].cpu().numpy(),
                'labels': pred['labels'][keep].cpu().numpy(),
                'scores': scores[keep]
            }
            filtered_predictions.append(filtered_pred)

        return filtered_predictions

    def validate(
        self,
        val_dataset: Dataset,
        batch_size: int = 4,
        conf_threshold: float = 0.001,
        save_results: bool = True,
        save_dir: str = None
    ) -> Dict:
        """
        Validate model and compute detection metrics

        Args:
            val_dataset: Validation dataset
            batch_size: Batch size for validation
            conf_threshold: Confidence threshold for predictions
            save_results: Save validation results to JSON
            save_dir: Directory to save results (uses model save_dir if None)

        Returns:
            Dictionary with validation metrics:
                - mAP@0.5:0.95: COCO-style mean Average Precision
                - mAP@0.5: Average Precision at IoU=0.5
                - mAP@0.75: Average Precision at IoU=0.75
                - precision: Mean precision across classes
                - recall: Mean recall across classes
                - f1_score: F1 score
                - per_class: Per-class metrics
        """
        print(f"\n{'='*60}")
        print("Running Validation")
        print(f"{'='*60}")
        print(f"Dataset size: {len(val_dataset)}")
        print(f"Batch size: {batch_size}")
        print(f"Confidence threshold: {conf_threshold}")
        print(f"{'='*60}\n")

        # Create data loader
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=self.collate_fn
        )

        # Run validation
        metrics = validate_model(
            model=self.model,
            dataloader=val_loader,
            device=self.device,
            num_classes=self.num_classes,
            conf_threshold=conf_threshold,
            verbose=True
        )

        # Save results if requested
        if save_results:
            if save_dir is None:
                save_dir = Path('experiments') / 'retinanet' / 'validation'
            else:
                save_dir = Path(save_dir)

            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / 'validation_metrics.json'
            save_metrics(metrics, str(save_path))

        return metrics

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        optimizer: optim.Optimizer,
        history: Dict
    ):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'num_classes': self.num_classes,
            'backbone': self.backbone
        }

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

        return checkpoint


if __name__ == "__main__":
    # Example usage
    print("RetinaNet Trainer Example\n")
    print("="*60)
    print("Available Backbones:")
    print("="*60)
    print("  resnet50      - ~160M params (default, high accuracy)")
    print("  resnet34      - ~33M params (lightweight)")
    print("  resnet18      - ~21M params (lightweight)")
    print("  mobilenet_v3  - ~5M params (very lightweight)")
    print("="*60)

    print("\n[Example 1] Default ResNet50:")
    trainer_resnet50 = RetinaNetTrainer(
        num_classes=4,
        backbone='resnet50',
        pretrained=True
    )

    print("\n[Example 2] Lightweight ResNet34:")
    trainer_resnet34 = RetinaNetTrainer(
        num_classes=4,
        backbone='resnet34',
        pretrained=True
    )

    print("\n[Example 3] Very Lightweight MobileNetV3:")
    trainer_mobilenet = RetinaNetTrainer(
        num_classes=4,
        backbone='mobilenet_v3',
        pretrained=True
    )

    print("\n" + "="*60)
    print("Usage Examples:")
    print("="*60)
    print("trainer.train(train_dataset, val_dataset, epochs=50)")
    print("trainer.predict(images, confidence_threshold=0.5)")
    print("trainer.save_checkpoint(path, epoch, optimizer, history)")
    print("="*60)
