# Model Training Errors - Debug Analysis

## Overview
Analyzed the `all_results.json` file and identified multiple training failures and one important clarification about YOLO metrics.

---

## Issue 1: Error "5" - RetinaNet and Faster R-CNN Models

### Affected Models:
- RetinaNet-resnet50
- RetinaNet-resnet34
- SSD-vgg16
- FasterRCNN-resnet50
- FasterRCNN-resnet34
- FasterRCNN-mobilenet_v3_large

### Root Cause:
The error "5" is likely coming from PyTorch's torchvision detection models when they encounter label values that equal or exceed `num_classes`.

**The Problem:**
- Dataset has 5 classes with category_ids: 1, 2, 3, 4, 5 (from COCO format)
- Training script passes `num_classes = 6` (5 classes + 1 background)
- But torchvision models expect labels in range [0, num_classes) which is [0, 6)
- Category ID 5 is valid (5 < 6), but the actual error might be an assertion failure

**Most Likely Cause:**
The COCO format annotations might not exist yet or have issues. Check if these files exist:
- `data/processed/coco_format/train_coco.json`
- `data/processed/coco_format/valid_coco.json`

### Solution:
1. Ensure COCO format annotations are created properly
2. Verify category IDs in COCO annotations are in range [1, 5] (background is 0, automatically handled)
3. Run the dataset conversion script to generate proper COCO annotations

---

## Issue 2: RetinaNet-mobilenet_v3 Anchor Configuration Error

### Error Message:
```
Anchors should be Tuple[Tuple[int]] because each feature map could potentially have different sizes and aspect ratios. There needs to be a match between the number of feature maps passed and the number of sizes / aspect ratios specified.
```

### Root Cause:
File: `src/modern/retinanet/trainer.py`, lines 228-231

The original code was:
```python
anchor_generator = AnchorGenerator(
    sizes=tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512]),
    aspect_ratios=((0.5, 1.0, 2.0),) * 5
)
```

This creates 5 anchor size groups but the MobileNetV3 backbone with `returned_layers=[3, 4]` only produces 2 feature maps.

### Solution: ✅ FIXED
Updated to:
```python
anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128), (64, 128, 256)),  # 2 feature maps from returned_layers
    aspect_ratios=((0.5, 1.0, 2.0), (0.5, 1.0, 2.0))
)
```

---

## Issue 3: SSD-mobilenet_v3 in_channels Detection Error

### Error Message:
```
Cannot determine in_channels list for SSD head replacement
```

### Root Cause:
File: `src/modern/ssd/trainer.py`, lines 228-240

The original logic couldn't extract `in_channels` from the MobileNetV3 SSD head's nested structure.

### Solution: ✅ FIXED
Enhanced the `_replace_head` method with:
1. Better nested structure traversal
2. Fallback logic to iterate through `modules()` and find Conv2d layers
3. More robust error handling

---

## Issue 4: YOLO Metrics Source

### Question:
Are the YOLO metrics from the test set or validation set?

### Answer: ⚠️ VALIDATION SET
Looking at `train_and_compare_all_models.py` lines 390-395:

```python
# Validate
print(f"\nValidating {model_name}...")
val_results = trainer.validate(
    data_yaml=data_yaml,
    split='val'  # ← Using validation set, NOT test set
)
```

**The metrics reported in `all_results.json` for YOLO models are from the VALIDATION set.**

### Impact:
- YOLOv11n: mAP@0.5 = 0.9054 (validation)
- YOLOv11s: mAP@0.5 = 0.9106 (validation)

These are validation metrics, not final test set metrics. For fair comparison with other models, YOLO should also be evaluated on the test set.

### Recommendation:
The script has an `evaluate_on_test_set()` method (lines 722-919) that should evaluate all models on the test set. Make sure to:
1. Run the full benchmark with test evaluation
2. Compare models using test set metrics, not validation metrics

---

## Summary of Fixes Applied

✅ **Fixed**: RetinaNet-mobilenet_v3 anchor configuration
✅ **Fixed**: SSD-mobilenet_v3 in_channels detection
⚠️ **Requires Data**: Error "5" models need COCO format annotations to be created
ℹ️ **Clarified**: YOLO metrics are from validation set, not test set

---

## Next Steps

1. **Create COCO Format Annotations**
   ```bash
   # Run your COCO conversion script
   python scripts/convert_to_coco.py  # or similar
   ```

2. **Verify Annotations**
   Check that these files exist and are valid:
   - `data/processed/coco_format/train_coco.json`
   - `data/processed/coco_format/valid_coco.json`
   - `data/processed/coco_format/test_coco.json`

3. **Re-run Training**
   ```bash
   python train_and_compare_all_models.py --epochs 20
   ```

4. **Evaluate on Test Set**
   Make sure the script completes the test set evaluation at the end

---

## Expected Results After Fixes

All models should train successfully:
- ✅ RetinaNet (ResNet50, ResNet34, MobileNetV3)
- ✅ SSD (VGG16, MobileNetV3)
- ✅ Faster R-CNN (ResNet50, ResNet34, MobileNetV3)
- ✅ YOLO (v11n, v11s)

Final comparison should use test set metrics for all models.
