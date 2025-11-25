# Model Training Errors - Debug Analysis

## Overview
Analyzed the `all_results.json` file and identified multiple training failures and one important clarification about YOLO metrics.

---

## Issue 1: Error "5" - RetinaNet and Faster R-CNN Models ✅ FIXED

### Affected Models:
- RetinaNet-resnet50
- RetinaNet-resnet34
- SSD-vgg16
- FasterRCNN-resnet50
- FasterRCNN-resnet34
- FasterRCNN-mobilenet_v3_large

### Root Cause:
**KeyError: 5** in the validation code when processing ground truth labels.

**The Problem:**
1. COCO format annotations use 1-indexed category IDs: 1, 2, 3, 4, 5
2. The validator creates dictionaries with keys `range(num_classes-1)` = [0, 1, 2, 3, 4]
3. Predictions are correctly remapped: 1->0, 2->1, etc. (line 373 in detection_validator.py)
4. **BUT** ground truth labels were NOT remapped and stayed as 1, 2, 3, 4, 5
5. When trying to access `all_gt_boxes[5]`, it causes KeyError because key 5 doesn't exist

### Solution: ✅ FIXED
File: `src/utils/detection_validator.py`, lines 379-384

Changed from:
```python
# Ground truth - already 0-indexed (no background class)
gt_boxes.append(target['boxes'].cpu().numpy())
gt_labels.append(target['labels'].cpu().numpy())
```

To:
```python
# Ground truth - remap labels to 0-indexed (COCO format uses 1-indexed categories)
gt_boxes.append(target['boxes'].cpu().numpy())
gt_labels_np = target['labels'].cpu().numpy()
# Remap labels: 1->0, 2->1, 3->2, etc. (subtract 1)
gt_labels_np = gt_labels_np - 1
gt_labels.append(gt_labels_np)
```

Now ground truth labels are properly remapped to match the validator's expected range [0, 4].

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

✅ **Fixed**: Error "5" - Ground truth label remapping in validator
✅ **Fixed**: RetinaNet-mobilenet_v3 anchor configuration
✅ **Fixed**: SSD-mobilenet_v3 in_channels detection
ℹ️ **Clarified**: YOLO metrics are from validation set, not test set

---

## Next Steps

**Re-run Training** (all fixes have been applied to the code):
```bash
python train_and_compare_all_models.py --epochs 20
```

**Note**: The COCO format annotations will be created automatically on Kaggle when you run the training script, or you can create them manually using:
```bash
python scripts/prepare_coco_format.py
```

---

## Expected Results After Fixes

All models should train successfully:
- ✅ RetinaNet (ResNet50, ResNet34, MobileNetV3)
- ✅ SSD (VGG16, MobileNetV3)
- ✅ Faster R-CNN (ResNet50, ResNet34, MobileNetV3)
- ✅ YOLO (v11n, v11s)

Final comparison should use test set metrics for all models.
