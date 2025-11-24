# Model Selection Guide

## Overview

This guide helps you choose the right model architecture and size based on your computational resources and requirements.

## Available Architectures

### One-Stage Detectors (Faster, Lower Accuracy)
- **YOLO (v5, v8, v11)** - Real-time detection, best overall speed/accuracy trade-off
- **SSD** - Classic one-stage detector, good for embedded systems
- **RetinaNet** - Uses Focal Loss, better handling of class imbalance

### Two-Stage Detectors (Slower, Higher Accuracy)
- **Faster R-CNN** - Classical two-stage detector, highest accuracy

## Quick Recommendations

### For Limited Resources (CPU or Low-end GPU)
1. **YOLOv5n** - Smallest and fastest YOLO variant
2. **SSDLite MobileNetV3** - Extremely lightweight (~3.4M params)
3. **YOLOv8n** - Slightly better accuracy than v5n
4. **RetinaNet MobileNetV3** - Lightweight with Focal Loss (~5M params)

### For Balanced Performance
1. **YOLOv8s** or **YOLOv11s** - Good balance of speed and accuracy
2. **SSD VGG16** - Classic single-stage detector (~26M params)
3. **RetinaNet ResNet34** - Better than SSD, Focal Loss benefits

### For Maximum Accuracy (with good GPU)
1. **YOLOv11m** or **YOLOv11l** - State-of-the-art YOLO
2. **Faster R-CNN with ResNet50** - Highest accuracy two-stage detector
3. **RetinaNet ResNet50** - Single-stage with near two-stage accuracy

---

## YOLO Models Comparison

### YOLOv5 (Ultralytics)
- **Release**: 2020
- **Pros**: Most mature, extensive documentation, fastest training
- **Cons**: Slightly lower accuracy than newer versions
- **Best for**: Quick experiments, CPU training, resource-constrained environments

| Model | Params | Speed (GPU) | Speed (CPU) | Best Use Case |
|-------|--------|-------------|-------------|---------------|
| YOLOv5n | ~1.9M | Very Fast | Fast | Edge devices, mobile |
| YOLOv5s | ~7.2M | Fast | Moderate | Balanced applications |
| YOLOv5m | ~21.2M | Moderate | Slow | Better accuracy needed |
| YOLOv5l | ~46.5M | Slow | Very Slow | High accuracy required |
| YOLOv5x | ~86.7M | Very Slow | Extremely Slow | Maximum accuracy |

### YOLOv8 (Ultralytics)
- **Release**: 2023
- **Pros**: Better accuracy than v5, improved architecture
- **Cons**: Slightly slower than v5
- **Best for**: Modern projects with moderate resources

| Model | Params | Speed (GPU) | Speed (CPU) | Best Use Case |
|-------|--------|-------------|-------------|---------------|
| YOLOv8n | ~3.2M | Very Fast | Fast | Mobile, edge devices |
| YOLOv8s | ~11.2M | Fast | Moderate | General purpose |
| YOLOv8m | ~25.9M | Moderate | Slow | Better accuracy |
| YOLOv8l | ~43.7M | Slow | Very Slow | High accuracy |
| YOLOv8x | ~68.2M | Very Slow | Extremely Slow | Maximum accuracy |

### YOLOv11 (Ultralytics)
- **Release**: 2024
- **Pros**: Latest architecture, best accuracy
- **Cons**: Newest, may have fewer community resources
- **Best for**: State-of-the-art results, research

| Model | Params | Speed (GPU) | Speed (CPU) | Best Use Case |
|-------|--------|-------------|-------------|---------------|
| YOLOv11n | ~2.6M | Very Fast | Fast | Latest lightweight model |
| YOLOv11s | ~9.4M | Fast | Moderate | Best balanced option |
| YOLOv11m | ~20.1M | Moderate | Slow | Modern accuracy needs |
| YOLOv11l | ~25.3M | Slow | Very Slow | High accuracy |
| YOLOv11x | ~56.9M | Very Slow | Extremely Slow | Maximum performance |

---

## Faster R-CNN Backbones Comparison

### MobileNetV3 Large
- **Parameters**: ~19M
- **Speed**: Fast
- **Accuracy**: Good
- **Best for**: Limited resources, mobile deployment
- **Variants**:
  - `mobilenet_v3_large`: Standard 640px input
  - `mobilenet_v3_large_320`: Faster with 320px input

### ResNet18
- **Parameters**: ~21M
- **Speed**: Fast
- **Accuracy**: Good
- **Best for**: Lightweight training, quick experiments

### ResNet34
- **Parameters**: ~33M
- **Speed**: Moderate
- **Accuracy**: Better
- **Best for**: Balanced speed and accuracy

### ResNet50 (Default)
- **Parameters**: ~160M
- **Speed**: Slower
- **Accuracy**: High
- **Best for**: Maximum accuracy, sufficient GPU memory

---

## SSD Models Comparison

### SSD300 VGG16
- **Release**: 2016
- **Parameters**: ~26M
- **Speed**: Fast
- **Accuracy**: Good
- **Pros**: Classic architecture, well-tested, good speed
- **Cons**: Lower accuracy than modern detectors
- **Best for**: Quick prototypes, baseline comparisons

### SSDLite320 MobileNetV3
- **Release**: 2019 (MobileNetV3) + SSDLite
- **Parameters**: ~3.4M
- **Speed**: Very Fast
- **Accuracy**: Moderate
- **Pros**: Extremely lightweight, fastest among all models, mobile-friendly
- **Cons**: Lower accuracy, less suitable for small objects
- **Best for**: Edge devices, mobile deployment, severely limited resources

| Model | Params | Input Size | Speed | Best Use Case |
|-------|--------|------------|-------|---------------|
| SSD300 VGG16 | ~26M | 300x300 | Fast | Baseline, quick experiments |
| SSDLite320 MobileNetV3 | ~3.4M | 320x320 | Very Fast | Mobile, edge devices |

---

## RetinaNet Models Comparison

RetinaNet uses **Focal Loss** to handle class imbalance, making it particularly good when you have many more background samples than object samples.

### Key Advantage
- **Focal Loss**: Reduces weight of easy examples, focuses on hard examples
- **Better than SSD**: Generally more accurate than SSD with similar speed
- **Single-stage**: Faster than Faster R-CNN, nearly as accurate

| Backbone | Params | Speed | Accuracy | Best Use Case |
|----------|--------|-------|----------|---------------|
| ResNet50 | ~160M | Moderate | High | High accuracy single-stage |
| ResNet34 | ~33M | Fast | Good | Balanced performance |
| ResNet18 | ~21M | Fast | Good | Lightweight |
| MobileNetV3 | ~5M | Very Fast | Moderate | Very lightweight |

---

## Architecture Comparison

| Architecture | Type | Speed | Accuracy | Parameters | Best For |
|--------------|------|-------|----------|------------|----------|
| YOLO | One-stage | Very Fast | High | 1.9M - 86M | Real-time, general use |
| SSD | One-stage | Very Fast | Moderate | 3.4M - 26M | Mobile, embedded |
| RetinaNet | One-stage | Fast | High | 5M - 160M | Imbalanced datasets |
| Faster R-CNN | Two-stage | Slow | Very High | 19M - 160M | Maximum accuracy |

### When to Choose Each Architecture

**Choose YOLO when:**
- You need real-time performance
- Want latest state-of-the-art results
- Need good documentation and community support
- Want easy deployment

**Choose SSD when:**
- Extremely limited resources (mobile, edge devices)
- Need fastest possible inference
- Working with embedded systems
- Want simplest architecture

**Choose RetinaNet when:**
- Dataset has class imbalance issues
- Want better accuracy than SSD without two-stage overhead
- Need Focal Loss benefits
- Balancing speed and accuracy

**Choose Faster R-CNN when:**
- Accuracy is top priority
- Have sufficient computational resources
- Can afford slower inference
- Need highest quality detections

---

## Training Resource Requirements

### Minimum Requirements by Model Size

| Model Type | GPU VRAM | RAM | Training Time (100 epochs) |
|------------|----------|-----|---------------------------|
| YOLOv5n/v8n | 2GB | 8GB | 2-4 hours |
| YOLOv8s/v11s | 4GB | 8GB | 4-6 hours |
| YOLOv8m/v11m | 6GB | 16GB | 8-12 hours |
| YOLOv11l/x | 8GB+ | 16GB+ | 12-24 hours |
| Faster R-CNN MobileNet | 4GB | 8GB | 6-10 hours |
| Faster R-CNN ResNet18/34 | 6GB | 16GB | 10-15 hours |
| Faster R-CNN ResNet50 | 8GB+ | 16GB+ | 15-24 hours |

### CPU Training
- **Possible but slow**: Use nano/small models only
- **Recommended batch size**: 2-4
- **Expected time**: 5-10x longer than GPU

---

## Code Examples

### YOLO Training Examples

```python
from src.modern.yolo.trainer import YOLOTrainer

# Example 1: Lightweight for limited resources
trainer = YOLOTrainer(
    model_version='v5',  # Use v5 for fastest training
    model_size='n',      # Nano = smallest
    img_size=640,
    device='auto'
)

# Example 2: Balanced performance
trainer = YOLOTrainer(
    model_version='v11',
    model_size='s',      # Small = good balance
    img_size=640,
    device='auto'
)

# Example 3: High accuracy with good GPU
trainer = YOLOTrainer(
    model_version='v11',
    model_size='m',      # Medium = better accuracy
    img_size=640,
    device='auto'
)

# Train
results = trainer.train(
    data_yaml='data/data.yaml',
    epochs=100,
    batch_size=16,  # Adjust based on GPU memory
    patience=50
)
```

### Faster R-CNN Training Examples

```python
from src.modern.faster_rcnn.trainer import FasterRCNNTrainer, TrafficSignDataset

# Example 1: Lightweight MobileNet
trainer = FasterRCNNTrainer(
    num_classes=4,  # 3 classes + background
    backbone='mobilenet_v3_large',
    pretrained=True,
    device='auto'
)

# Example 2: Faster inference with 320px
trainer = FasterRCNNTrainer(
    num_classes=4,
    backbone='mobilenet_v3_large_320',
    pretrained=True,
    device='auto'
)

# Example 3: Better accuracy with ResNet34
trainer = FasterRCNNTrainer(
    num_classes=4,
    backbone='resnet34',
    pretrained=True,
    device='auto'
)

# Example 4: Maximum accuracy with ResNet50
trainer = FasterRCNNTrainer(
    num_classes=4,
    backbone='resnet50',
    pretrained=True,
    device='auto'
)

# Create dataset
train_dataset = TrafficSignDataset(
    images_dir='data/train/images',
    annotations_file='data/train/annotations.json'
)

# Train
history = trainer.train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=50,
    batch_size=4,  # Adjust based on GPU memory
    learning_rate=0.005
)
```

### SSD Training Examples

```python
from src.modern.ssd.trainer import SSDTrainer, TrafficSignDataset

# Example 1: Extremely lightweight SSDLite (best for mobile/edge)
trainer = SSDTrainer(
    num_classes=4,
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

# Create dataset
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
    batch_size=8,  # SSD can handle larger batches
    learning_rate=0.001
)
```

### RetinaNet Training Examples

```python
from src.modern.retinanet.trainer import RetinaNetTrainer, TrafficSignDataset

# Example 1: Very lightweight MobileNetV3
trainer = RetinaNetTrainer(
    num_classes=4,
    backbone='mobilenet_v3',
    pretrained=True,
    device='auto'
)

# Example 2: Balanced ResNet34
trainer = RetinaNetTrainer(
    num_classes=4,
    backbone='resnet34',
    pretrained=True,
    device='auto'
)

# Example 3: High accuracy ResNet50
trainer = RetinaNetTrainer(
    num_classes=4,
    backbone='resnet50',
    pretrained=True,
    device='auto'
)

# Create dataset
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
```

---

## Batch Size Guidelines

Adjust batch size based on your GPU memory:

### YOLO Models
- **2GB VRAM**: batch_size=2-4 (nano models only)
- **4GB VRAM**: batch_size=8-16 (nano/small models)
- **6GB VRAM**: batch_size=16-32 (up to medium models)
- **8GB+ VRAM**: batch_size=32-64 (any model)

### Faster R-CNN
- **4GB VRAM**: batch_size=1-2 (MobileNet only)
- **6GB VRAM**: batch_size=2-4 (MobileNet/ResNet18)
- **8GB VRAM**: batch_size=4-8 (up to ResNet34)
- **12GB+ VRAM**: batch_size=8-16 (ResNet50)

**Note**: Lower batch sizes may lead to less stable training. Consider using gradient accumulation if you need larger effective batch sizes.

---

## Performance Tips

### For Limited Resources
1. Use nano or small models
2. Reduce image size (e.g., 416 or 320 instead of 640)
3. Lower batch size
4. Use mixed precision training (AMP)
5. Reduce number of epochs with early stopping

### For Better Accuracy
1. Use larger models (medium/large)
2. Increase image size (e.g., 1280)
3. Train longer (more epochs)
4. Use ensemble of multiple models
5. Fine-tune with lower learning rate

### For Faster Training
1. Use smaller models
2. Use YOLOv5 (fastest to train)
3. Reduce image size
4. Use larger batch size (if GPU allows)
5. Enable mixed precision (AMP)

---

## Model Selection Decision Tree

```
Do you have a GPU?
├─ No → Use YOLOv5n or YOLOv8n with small image size
└─ Yes
    └─ GPU VRAM?
        ├─ < 4GB → YOLOv5n, YOLOv8n, or Faster R-CNN MobileNet
        ├─ 4-6GB → YOLOv8s, YOLOv11s, or Faster R-CNN ResNet18/34
        └─ > 6GB
            └─ Need maximum accuracy?
                ├─ Yes → YOLOv11l/x or Faster R-CNN ResNet50
                └─ No → YOLOv11m or Faster R-CNN ResNet34
```

---

## Expected Performance

### On GTSDB/GTSRB Traffic Signs (approximate)

| Model | mAP50 | mAP50-95 | FPS (GPU) | FPS (CPU) |
|-------|-------|----------|-----------|-----------|
| YOLOv5n | ~0.85 | ~0.55 | 150+ | 20-30 |
| YOLOv8s | ~0.90 | ~0.65 | 100+ | 10-15 |
| YOLOv11m | ~0.93 | ~0.72 | 60-80 | 5-10 |
| Faster R-CNN MobileNet | ~0.88 | ~0.68 | 30-40 | 3-5 |
| Faster R-CNN ResNet50 | ~0.92 | ~0.75 | 15-25 | 1-2 |

*Note: Actual performance depends on dataset quality and hyperparameters*

---

## Summary

**Best starter choice**: YOLOv8s or YOLOv11s with 640px image size
**Most lightweight**: YOLOv5n with 416px image size
**Best accuracy**: YOLOv11l or Faster R-CNN ResNet50
**Best for deployment**: YOLOv8n or Faster R-CNN MobileNetV3 320
