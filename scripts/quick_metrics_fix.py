"""
Quick Metrics Fix - Recompute mAP excluding background class
Uses the per-class AP values from the training output
"""

import numpy as np

print("\n" + "="*80)
print("CORRECTED METRICS COMPUTATION (Excluding Background Class)")
print("="*80)

# Data from your training output
models = {
    "RetinaNet-resnet50": {
        "per_class_ap": {
            0: 0.0000,  # Background - should be excluded
            1: 0.8800,  # bien_bao_hieu_lenh
            2: 0.6185,  # bien_bao_nguy_hiem_va_canh_bao
            3: 1.0000,  # bien_chi_dan
            4: 0.7506,  # bien_phu
            5: 0.3604,  # bien_bao_cam
        },
        "original_metrics": {
            "mAP@0.5": 0.6016,
            "mAP@0.75": 0.5575,
            "mAP@0.5:0.95": 0.4801,
            "Precision": 0.3688,
            "Recall": 0.7086,
            "F1": 0.4851
        }
    },
    "RetinaNet-resnet34": {
        "per_class_ap": {
            0: 0.0000,  # Background
            1: 0.8402,
            2: 0.5599,
            3: 0.9091,
            4: 0.6482,
            5: 0.2495,
        },
        "original_metrics": {
            "mAP@0.5": 0.5345,
            "mAP@0.75": 0.4398,
            "mAP@0.5:0.95": 0.3822,
            "Precision": 0.2994,
            "Recall": 0.6397,
            "F1": 0.4079
        }
    },
    "SSD-vgg16": {
        "per_class_ap": {
            0: 0.0000,  # Background
            1: 0.7822,
            2: 0.5455,
            3: 0.9061,
            4: 0.6403,
            5: 0.4080,
        },
        "original_metrics": {
            "mAP@0.5": 0.5470,
            "mAP@0.75": 0.4510,
            "mAP@0.5:0.95": 0.3861,
            "Precision": 0.5546,
            "Recall": 0.5900,
            "F1": 0.5717
        }
    }
}


def compute_corrected_metrics(model_name, model_data):
    """Compute corrected metrics excluding background class"""

    print(f"\n{'='*80}")
    print(f"{model_name}")
    print(f"{'='*80}")

    # Extract per-class AP values
    per_class_ap = model_data["per_class_ap"]
    original_metrics = model_data["original_metrics"]

    # Show original computation (WRONG - includes background)
    print(f"\n❌ ORIGINAL (Incorrect - with background):")
    print(f"   Per-class AP@0.5:")
    all_aps = []
    for class_id in range(6):
        ap = per_class_ap[class_id]
        all_aps.append(ap)
        marker = " ← Background (wrong!)" if class_id == 0 else ""
        print(f"     Class {class_id}: {ap:.4f}{marker}")

    original_mAP = np.mean(all_aps)
    print(f"\n   Formula: ({' + '.join([f'{ap:.4f}' for ap in all_aps])}) / 6")
    print(f"   mAP@0.5 = {original_mAP:.4f}")
    print(f"   (Matches reported: {original_metrics['mAP@0.5']:.4f} ✓)")

    # Compute corrected metrics (RIGHT - excludes background)
    print(f"\n✅ CORRECTED (Excluding background):")
    print(f"   Per-class AP@0.5:")
    corrected_aps = []
    class_names = {
        1: "bien_bao_hieu_lenh",
        2: "bien_bao_nguy_hiem_va_canh_bao",
        3: "bien_chi_dan",
        4: "bien_phu",
        5: "bien_bao_cam"
    }

    for class_id in range(1, 6):  # Skip class 0 (background)
        ap = per_class_ap[class_id]
        corrected_aps.append(ap)
        print(f"     Class {class_id-1} ({class_names[class_id][:20]:<20}): {ap:.4f}")

    corrected_mAP = np.mean(corrected_aps)
    print(f"\n   Formula: ({' + '.join([f'{ap:.4f}' for ap in corrected_aps])}) / 5")
    print(f"   mAP@0.5 = {corrected_mAP:.4f}")

    # Show improvement
    improvement_abs = corrected_mAP - original_mAP
    improvement_pct = (improvement_abs / original_mAP) * 100

    print(f"\n📈 Improvement:")
    print(f"   Absolute: +{improvement_abs:.4f}")
    print(f"   Relative: +{improvement_pct:.1f}%")
    print(f"   From {original_mAP:.4f} → {corrected_mAP:.4f}")

    # Estimate corrected mAP@0.5:0.95 and mAP@0.75
    # The background class penalty affects all IoU thresholds similarly
    # So we can apply the same correction ratio

    original_mAP_75 = model_data["original_metrics"].get("mAP@0.75", None)
    original_mAP_coco = original_metrics["mAP@0.5:0.95"]

    # Correction factor (how much background class dragged down metrics)
    correction_factor = corrected_mAP / original_mAP

    # Apply same correction to other metrics
    corrected_mAP_coco = original_mAP_coco * correction_factor
    corrected_mAP_75 = original_mAP_75 * correction_factor if original_mAP_75 else None

    print(f"\n📊 Corrected metrics (applying correction factor: {correction_factor:.4f}):")
    print(f"   mAP@0.5:     {corrected_mAP:.4f} (exact - computed from per-class APs)")
    print(f"   mAP@0.5:0.95: {corrected_mAP_coco:.4f} (estimated: {original_mAP_coco:.4f} × {correction_factor:.4f})")
    if corrected_mAP_75:
        print(f"   mAP@0.75:    {corrected_mAP_75:.4f} (estimated: {original_mAP_75:.4f} × {correction_factor:.4f})")
    print(f"   Precision:   {original_metrics['Precision']:.4f} (unchanged)")
    print(f"   Recall:      {original_metrics['Recall']:.4f} (unchanged)")
    print(f"   F1 Score:    {original_metrics['F1']:.4f} (unchanged)")

    return {
        "model": model_name,
        "original_mAP@0.5": original_mAP,
        "corrected_mAP@0.5": corrected_mAP,
        "corrected_mAP@0.75": corrected_mAP_75,
        "corrected_mAP@0.5:0.95": corrected_mAP_coco,
        "correction_factor": correction_factor,
        "improvement_abs": improvement_abs,
        "improvement_pct": improvement_pct,
        "precision": original_metrics['Precision'],
        "recall": original_metrics['Recall'],
        "f1": original_metrics['F1']
    }


# Compute corrected metrics for all models
results = []
for model_name, model_data in models.items():
    result = compute_corrected_metrics(model_name, model_data)
    results.append(result)


# Print summary table
print("\n\n" + "="*80)
print("SUMMARY: CORRECTED METRICS (Excluding Background)")
print("="*80)
print(f"\n{'Model':<25} {'mAP@0.5':<12} {'mAP@0.75':<12} {'mAP@0.5:0.95':<14} {'Improvement':<12} {'P':<8} {'R':<8}")
print(f"{'':25} {'(corr.)':<12} {'(est.)':<12} {'(est.)':<14} {'':12} {'':8} {'':8}")
print("-"*100)

for r in results:
    print(f"{r['model']:<25} {r['corrected_mAP@0.5']:<12.4f} {r['corrected_mAP@0.75']:<12.4f} "
          f"{r['corrected_mAP@0.5:0.95']:<14.4f} +{r['improvement_pct']:>5.1f}% "
          f"{r['precision']:<8.4f} {r['recall']:<8.4f}")

print("="*100)

# Best model
best_model = max(results, key=lambda x: x['corrected_mAP@0.5'])
print(f"\n🏆 Best Model: {best_model['model']}")
print(f"   Corrected mAP@0.5: {best_model['corrected_mAP@0.5']:.4f}")
print(f"   Corrected mAP@0.5:0.95: {best_model['corrected_mAP@0.5:0.95']:.4f}")

print("\n💡 Key Insight:")
print("   By excluding the background class (which always has AP=0.0),")
print("   all models show 15-20% improvement in mAP@0.5!")
print("\n" + "="*80 + "\n")
