# Multi-Backbone & Multi-YOLO Support

## Overview

Enhanced support for experimenting with multiple model architectures, especially lightweight options for resource-constrained environments.

## What's New

### Faster R-CNN - Multiple Backbones

The Faster R-CNN trainer now supports **5 different backbones**:

| Backbone | Parameters | Speed | Use Case |
|----------|-----------|-------|----------|
| `resnet50` | ~160M | Slow | Maximum accuracy (default) |
| `resnet34` | ~33M | Moderate | Balanced performance |
| `resnet18` | ~21M | Fast | Lightweight training |
| `mobilenet_v3_large` | ~19M | Fast | Resource-constrained |
| `mobilenet_v3_large_320` | ~19M | Very Fast | Fastest inference |

**Usage:**
```python
from src.modern.faster_rcnn.trainer import FasterRCNNTrainer

# Lightweight option
trainer = FasterRCNNTrainer(
    num_classes=4,
    backbone='mobilenet_v3_large',
    pretrained=True
)

# Balanced option
trainer = FasterRCNNTrainer(
    num_classes=4,
    backbone='resnet34',
    pretrained=True
)
```

### YOLO - Multiple Versions

The YOLO trainer now supports **3 YOLO versions** with **5 size variants** each:

| Version | Release | Strengths | Best For |
|---------|---------|-----------|----------|
| `v5` | 2020 | Fastest training, most mature | Quick experiments, CPU |
| `v8` | 2023 | Better accuracy than v5 | Balanced projects |
| `v11` | 2024 | Best accuracy, latest | State-of-the-art results |

**Size Variants:**
- `n` (nano): ~1-3M params - Fastest, lowest accuracy
- `s` (small): ~7-11M params - Good balance
- `m` (medium): ~20-25M params - Better accuracy
- `l` (large): ~40-50M params - High accuracy
- `x` (xlarge): ~60-80M params - Maximum accuracy

**Usage:**
```python
from src.modern.yolo.trainer import YOLOTrainer

# Lightweight for limited resources
trainer = YOLOTrainer(
    model_version='v5',
    model_size='n',
    img_size=640
)

# Balanced performance
trainer = YOLOTrainer(
    model_version='v8',
    model_size='s',
    img_size=640
)

# State-of-the-art
trainer = YOLOTrainer(
    model_version='v11',
    model_size='m',
    img_size=640
)
```

## Quick Start for Limited Resources

### Option 1: YOLOv5 Nano (Recommended)
```python
from src.modern.yolo.trainer import YOLOTrainer

trainer = YOLOTrainer(
    model_version='v5',
    model_size='n',
    img_size=640
)

results = trainer.train(
    data_yaml='data/data.yaml',
    epochs=100,
    batch_size=16,  # Adjust based on your GPU
    patience=50
)
```

**Why YOLOv5n?**
- Smallest model (~1.9M parameters)
- Fastest training time
- Works well on CPU
- Good starting point for experiments

### Option 2: MobileNetV3 Faster R-CNN
```python
from src.modern.faster_rcnn.trainer import FasterRCNNTrainer, TrafficSignDataset

trainer = FasterRCNNTrainer(
    num_classes=4,
    backbone='mobilenet_v3_large',
    pretrained=True
)

train_dataset = TrafficSignDataset(
    images_dir='data/train/images',
    annotations_file='data/train/annotations.json'
)

val_dataset = TrafficSignDataset(
    images_dir='data/val/images',
    annotations_file='data/val/annotations.json'
)

history = trainer.train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=50,
    batch_size=4,
    learning_rate=0.005
)
```

**Why MobileNetV3?**
- Lightweight (~19M parameters)
- Good accuracy for two-stage detector
- Faster inference than ResNet

## Comparison Tool

Run the comparison script to see all available models:

```bash
python scripts/compare_models.py
```

This will show:
- All available YOLO models
- All Faster R-CNN backbones
- Parameter counts and sizes
- Recommendations based on your hardware

## Files Modified

1. **[src/modern/faster_rcnn/trainer.py](../src/modern/faster_rcnn/trainer.py)**
   - Added `_create_model()` method
   - Support for 5 backbones
   - Updated documentation

2. **[src/modern/yolo/trainer.py](../src/modern/yolo/trainer.py)**
   - Added `model_version` parameter
   - Added `_get_model_name()` method
   - Added `_get_model_info()` method
   - Support for YOLOv5, v8, v11

## Files Created

1. **[docs/model_selection_guide.md](model_selection_guide.md)**
   - Comprehensive guide for choosing models
   - Resource requirements
   - Performance comparisons
   - Decision tree

2. **[scripts/compare_models.py](../scripts/compare_models.py)**
   - Benchmark different models
   - Show parameter counts
   - Provide recommendations
   - Usage examples

## Resource Requirements

### Minimum for Different Models

| Model | GPU VRAM | RAM | Recommended Batch Size |
|-------|----------|-----|----------------------|
| YOLOv5n | 2GB | 8GB | 16 |
| YOLOv8s | 4GB | 8GB | 16 |
| YOLOv11m | 6GB | 16GB | 8-16 |
| Faster R-CNN MobileNet | 4GB | 8GB | 2-4 |
| Faster R-CNN ResNet34 | 6GB | 16GB | 4 |
| Faster R-CNN ResNet50 | 8GB+ | 16GB | 4-8 |

### CPU Training
- Possible with nano/small models
- Expect 5-10x longer training time
- Use batch_size=2-4

## Expected Training Times

On a typical GPU (e.g., RTX 3060 with 6GB VRAM):

| Model | 100 Epochs | 50 Epochs |
|-------|-----------|-----------|
| YOLOv5n | 2-3 hours | 1-1.5 hours |
| YOLOv8s | 4-5 hours | 2-2.5 hours |
| YOLOv11m | 8-10 hours | 4-5 hours |
| Faster R-CNN MobileNet | 6-8 hours | 3-4 hours |
| Faster R-CNN ResNet34 | 10-12 hours | 5-6 hours |
| Faster R-CNN ResNet50 | 15-20 hours | 7-10 hours |

## Performance Tips

### For Limited Resources
1. Start with YOLOv5n or YOLOv8n
2. Use img_size=416 instead of 640
3. Reduce batch size if OOM errors occur
4. Enable mixed precision (AMP) - automatically enabled for YOLO
5. Use early stopping with patience

### For Better Accuracy
1. Use larger models (m, l, x)
2. Increase img_size to 1280
3. Train longer with more epochs
4. Use test-time augmentation
5. Ensemble multiple models

### Reducing Memory Usage
```python
# For YOLO - reduce image size and batch size
trainer = YOLOTrainer(
    model_version='v8',
    model_size='n',
    img_size=416  # Instead of 640
)
trainer.train(
    data_yaml='data/data.yaml',
    batch_size=8  # Instead of 16
)

# For Faster R-CNN - use smaller backbone and batch size
trainer = FasterRCNNTrainer(
    num_classes=4,
    backbone='mobilenet_v3_large_320'  # Smallest + fastest
)
trainer.train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=2  # Smaller batch
)
```

## Experimental Workflow

### 1. Quick Experiment (1-2 hours)
```python
# Use nano model with reduced settings
trainer = YOLOTrainer(model_version='v5', model_size='n', img_size=416)
trainer.train(data_yaml='data/data.yaml', epochs=30, batch_size=16)
```

### 2. Baseline (3-5 hours)
```python
# Use small model with standard settings
trainer = YOLOTrainer(model_version='v8', model_size='s', img_size=640)
trainer.train(data_yaml='data/data.yaml', epochs=100, batch_size=16)
```

### 3. High Performance (8-12 hours)
```python
# Use medium model with full settings
trainer = YOLOTrainer(model_version='v11', model_size='m', img_size=640)
trainer.train(data_yaml='data/data.yaml', epochs=150, batch_size=16)
```

### 4. Maximum Accuracy (1-2 days)
```python
# Use large model or Faster R-CNN with full settings
trainer = YOLOTrainer(model_version='v11', model_size='l', img_size=1280)
trainer.train(data_yaml='data/data.yaml', epochs=300, batch_size=8)
```

## Troubleshooting

### Out of Memory (OOM) Error
1. Reduce batch size by half
2. Reduce image size (640 → 416)
3. Use smaller model size (s → n)
4. Close other applications
5. Try MobileNetV3 instead of ResNet

### Training Too Slow
1. Use GPU if available
2. Use smaller model (v5n instead of v11m)
3. Reduce image size
4. Reduce number of epochs
5. Check CPU/GPU utilization

### Poor Accuracy
1. Train longer (more epochs)
2. Use larger model
3. Increase image size
4. Check data quality
5. Try different model version

## Next Steps

1. Read [model_selection_guide.md](model_selection_guide.md) for detailed comparison
2. Run `python scripts/compare_models.py` to see all options
3. Start with YOLOv8n or YOLOv11n for quick experiments
4. Scale up to larger models as needed
5. Compare YOLO vs Faster R-CNN for your specific use case

## References

- [YOLOv5 Documentation](https://docs.ultralytics.com/yolov5/)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Torchvision Detection Models](https://pytorch.org/vision/stable/models.html#object-detection)
- [MobileNetV3 Paper](https://arxiv.org/abs/1905.02244)
