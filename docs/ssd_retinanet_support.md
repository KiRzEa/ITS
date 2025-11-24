# SSD and RetinaNet Support

## Overview

Added comprehensive support for **SSD** and **RetinaNet** architectures, providing you with 4 different detection architectures to experiment with:

1. **YOLO** (v5, v8, v11) - One-stage, real-time
2. **SSD** - One-stage, classic
3. **RetinaNet** - One-stage, Focal Loss
4. **Faster R-CNN** - Two-stage, highest accuracy

## Why Add SSD and RetinaNet?

### SSD (Single Shot Detector)
- **Extremely lightweight**: SSDLite MobileNetV3 has only ~3.4M parameters
- **Fast inference**: One of the fastest detectors available
- **Mobile-friendly**: Perfect for edge devices and embedded systems
- **Simple architecture**: Easy to understand and debug
- **Best for**: Resource-constrained environments, mobile deployment

### RetinaNet
- **Focal Loss**: Handles class imbalance better than standard loss functions
- **Better than SSD**: More accurate while maintaining single-stage speed
- **Balanced**: Good compromise between speed and accuracy
- **Single-stage efficiency**: Faster than two-stage detectors
- **Best for**: Datasets with imbalanced classes, when accuracy matters but speed too

---

## SSD Implementation

### Available Backbones

| Backbone | Model | Parameters | Input Size | Speed | Use Case |
|----------|-------|------------|------------|-------|----------|
| VGG16 | SSD300 | ~26M | 300x300 | Fast | Baseline, experiments |
| MobileNetV3 | SSDLite320 | ~3.4M | 320x320 | Very Fast | Mobile, edge devices |

### Usage Examples

```python
from src.modern.ssd.trainer import SSDTrainer, TrafficSignDataset

# Example 1: Extremely lightweight for mobile (RECOMMENDED for limited resources)
trainer = SSDTrainer(
    num_classes=4,  # 3 classes + background
    backbone='mobilenet_v3',
    pretrained=True,
    device='auto'
)

# Example 2: Classic SSD with VGG16
trainer = SSDTrainer(
    num_classes=4,
    backbone='vgg16',
    pretrained=True,
    device='auto'
)

# Create datasets
train_dataset = TrafficSignDataset(
    images_dir='data/train/images',
    annotations_file='data/train/annotations.json'
)

val_dataset = TrafficSignDataset(
    images_dir='data/val/images',
    annotations_file='data/val/annotations.json'
)

# Train
history = trainer.train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=50,
    batch_size=8,  # SSD can handle larger batches than Faster R-CNN
    learning_rate=0.001
)

# Predict
predictions = trainer.predict(images, confidence_threshold=0.5)
```

### SSD Advantages

1. **Fastest among all models**: SSDLite MobileNetV3 is the fastest detector
2. **Lightweight**: Only 3.4M parameters for MobileNetV3 variant
3. **Real-time capable**: Can run at 30+ FPS even on CPU
4. **Mobile deployment**: Small model size perfect for mobile apps
5. **Easy to train**: Converges quickly, less prone to overfitting

### SSD Limitations

1. **Lower accuracy**: Less accurate than RetinaNet or Faster R-CNN
2. **Small object detection**: May struggle with very small objects
3. **Older architecture**: Not as modern as YOLO or RetinaNet

---

## RetinaNet Implementation

### Available Backbones

| Backbone | Parameters | Speed | Accuracy | Use Case |
|----------|------------|-------|----------|----------|
| ResNet50 | ~160M | Moderate | High | High accuracy single-stage |
| ResNet34 | ~33M | Fast | Good | Balanced performance |
| ResNet18 | ~21M | Fast | Good | Lightweight |
| MobileNetV3 | ~5M | Very Fast | Moderate | Very lightweight |

### Usage Examples

```python
from src.modern.retinanet.trainer import RetinaNetTrainer, TrafficSignDataset

# Example 1: Very lightweight with MobileNetV3
trainer = RetinaNetTrainer(
    num_classes=4,
    backbone='mobilenet_v3',
    pretrained=True,
    device='auto'
)

# Example 2: Balanced with ResNet34 (RECOMMENDED for moderate resources)
trainer = RetinaNetTrainer(
    num_classes=4,
    backbone='resnet34',
    pretrained=True,
    device='auto'
)

# Example 3: High accuracy with ResNet50
trainer = RetinaNetTrainer(
    num_classes=4,
    backbone='resnet50',
    pretrained=True,
    device='auto'
)

# Create datasets
train_dataset = TrafficSignDataset(
    images_dir='data/train/images',
    annotations_file='data/train/annotations.json'
)

val_dataset = TrafficSignDataset(
    images_dir='data/val/images',
    annotations_file='data/val/annotations.json'
)

# Train
history = trainer.train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=50,
    batch_size=4,
    learning_rate=0.001
)

# Predict
predictions = trainer.predict(images, confidence_threshold=0.5)
```

### RetinaNet Advantages

1. **Focal Loss**: Automatically handles class imbalance
2. **Better than SSD**: More accurate while still being single-stage
3. **Flexible backbones**: Can use lightweight or heavy backbones
4. **Good for imbalanced data**: Focuses on hard examples
5. **Single-stage speed**: Faster than Faster R-CNN

### RetinaNet Limitations

1. **Slower than YOLO/SSD**: Not as fast as other single-stage detectors
2. **More memory**: Requires more GPU memory than SSD
3. **Training time**: Takes longer to train than SSD

---

## Architecture Comparison

### Speed Ranking (Fastest to Slowest)
1. **SSDLite MobileNetV3** (~3.4M params) - Fastest
2. **YOLOv5n** (~1.9M params) - Very Fast
3. **YOLOv8n/v11n** (~2-3M params) - Very Fast
4. **SSD VGG16** (~26M params) - Fast
5. **RetinaNet MobileNetV3** (~5M params) - Fast
6. **YOLOv8s/v11s** (~9-11M params) - Fast
7. **RetinaNet ResNet18/34** (~21-33M params) - Moderate
8. **YOLOv8m/v11m** (~20-25M params) - Moderate
9. **RetinaNet ResNet50** (~160M params) - Moderate
10. **Faster R-CNN MobileNetV3** (~19M params) - Slow
11. **Faster R-CNN ResNet34/50** (~33-160M params) - Slow

### Accuracy Ranking (Highest to Lowest)
1. **Faster R-CNN ResNet50** - Highest
2. **RetinaNet ResNet50** - High
3. **YOLOv11l/x** - High
4. **YOLOv11m** - Good
5. **Faster R-CNN ResNet34** - Good
6. **RetinaNet ResNet34** - Good
7. **YOLOv8s/v11s** - Good
8. **SSD VGG16** - Moderate
9. **YOLOv5n/v8n** - Moderate
10. **RetinaNet MobileNetV3** - Moderate
11. **SSDLite MobileNetV3** - Lower

### Resource Requirements

| Model | Min GPU VRAM | Batch Size | Training Time (50 epochs) |
|-------|--------------|------------|---------------------------|
| SSDLite MobileNetV3 | 2GB | 16 | 3-5 hours |
| YOLOv5n/v8n | 2GB | 16 | 2-4 hours |
| SSD VGG16 | 4GB | 8 | 5-7 hours |
| RetinaNet MobileNetV3 | 4GB | 4 | 6-8 hours |
| YOLOv8s/v11s | 4GB | 16 | 4-6 hours |
| RetinaNet ResNet34 | 6GB | 4 | 8-12 hours |
| Faster R-CNN MobileNetV3 | 4GB | 4 | 6-10 hours |
| RetinaNet ResNet50 | 8GB | 4 | 12-18 hours |
| Faster R-CNN ResNet50 | 8GB | 4 | 15-24 hours |

---

## When to Use Each Architecture

### Use **SSD** when:
- ✅ Extremely limited resources (CPU only, low-end GPU)
- ✅ Need absolute fastest inference speed
- ✅ Deploying to mobile or embedded devices
- ✅ Model size is critical (<5MB)
- ✅ Want simplest architecture
- ❌ Don't need highest accuracy

### Use **RetinaNet** when:
- ✅ Dataset has class imbalance (many background, few objects)
- ✅ Want better accuracy than SSD without two-stage overhead
- ✅ Balancing speed and accuracy
- ✅ Need Focal Loss benefits
- ✅ Have moderate GPU resources (4-8GB)
- ❌ Don't need absolute fastest speed

### Use **YOLO** when:
- ✅ Need real-time performance
- ✅ Want state-of-the-art results
- ✅ Need good documentation
- ✅ Want easy deployment (ONNX, TensorRT)
- ✅ General-purpose detection
- ✅ Latest architectures

### Use **Faster R-CNN** when:
- ✅ Accuracy is top priority
- ✅ Have good GPU resources (6-8GB+)
- ✅ Can afford slower inference
- ✅ Need highest quality detections
- ❌ Speed is not critical

---

## Decision Tree

```
What's your priority?

1. Absolute fastest speed + smallest model?
   → SSDLite MobileNetV3

2. Mobile/edge deployment?
   → SSDLite MobileNetV3 or YOLOv5n

3. Dataset has class imbalance?
   → RetinaNet (ResNet34 or MobileNetV3)

4. Need real-time with good accuracy?
   → YOLOv8s or YOLOv11s

5. Maximum accuracy, speed not critical?
   → Faster R-CNN ResNet50

6. Balanced speed and accuracy?
   → RetinaNet ResNet34 or YOLOv8s

7. Limited resources (<4GB GPU)?
   → SSDLite MobileNetV3, YOLOv8n, or RetinaNet MobileNetV3

8. State-of-the-art results?
   → YOLOv11m or YOLOv11l
```

---

## Practical Recommendations

### For Beginners
Start with **YOLOv8n** - easiest to use, good accuracy, fast training

### For Limited Resources
1. **SSDLite MobileNetV3** - Absolute fastest
2. **YOLOv5n** - Very fast, slightly better accuracy
3. **RetinaNet MobileNetV3** - If you have class imbalance

### For Production Deployment
- **Mobile**: SSDLite MobileNetV3
- **Server**: YOLOv8s or RetinaNet ResNet34
- **High-accuracy**: Faster R-CNN ResNet50

### For Experiments
Try all four architectures with lightweight backbones:
1. YOLOv8n (2-4 hours training)
2. SSDLite MobileNetV3 (3-5 hours training)
3. RetinaNet MobileNetV3 (6-8 hours training)
4. Faster R-CNN MobileNetV3 (6-10 hours training)

---

## Example Experiment Workflow

### Phase 1: Quick Baseline (1 day)
```python
# Train lightweight models for quick comparison
models = [
    ('yolo', 'v8', 'n'),
    ('ssd', 'mobilenet_v3'),
    ('retinanet', 'mobilenet_v3'),
]

# Train each for 30-50 epochs
# Compare mAP, speed, and memory usage
```

### Phase 2: Balanced Models (2-3 days)
```python
# Train medium-sized models
models = [
    ('yolo', 'v8', 's'),
    ('ssd', 'vgg16'),
    ('retinanet', 'resnet34'),
    ('faster_rcnn', 'mobilenet_v3'),
]

# Train each for 50-100 epochs
# Detailed evaluation and analysis
```

### Phase 3: High Accuracy (1 week)
```python
# Train best-performing architectures with larger backbones
models = [
    ('yolo', 'v11', 'm'),
    ('retinanet', 'resnet50'),
    ('faster_rcnn', 'resnet50'),
]

# Train for 100-150 epochs
# Hyperparameter tuning
# Ensemble methods
```

---

## Files Created

1. **[src/modern/ssd/trainer.py](../src/modern/ssd/trainer.py)** - SSD trainer implementation
2. **[src/modern/ssd/__init__.py](../src/modern/ssd/__init__.py)** - SSD module init
3. **[src/modern/retinanet/trainer.py](../src/modern/retinanet/trainer.py)** - RetinaNet trainer
4. **[src/modern/retinanet/__init__.py](../src/modern/retinanet/__init__.py)** - RetinaNet module init

## Files Updated

1. **[docs/model_selection_guide.md](model_selection_guide.md)** - Added SSD and RetinaNet sections
2. **[scripts/compare_models.py](../scripts/compare_models.py)** - Added SSD and RetinaNet benchmarking

---

## Summary

You now have **4 complete detection architectures** with **multiple backbone options**:

| Architecture | Backbones | Total Models |
|--------------|-----------|--------------|
| YOLO | 3 versions × 5 sizes | 15 models |
| SSD | 2 backbones | 2 models |
| RetinaNet | 4 backbones | 4 models |
| Faster R-CNN | 5 backbones | 5 models |
| **Total** | | **26 models** |

This gives you comprehensive options for experimenting with different speed/accuracy trade-offs!
