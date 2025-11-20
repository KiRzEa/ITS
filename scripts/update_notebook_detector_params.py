"""
Update the notebook with optimized sliding window detector parameters
"""
import json

with open('notebooks/02_traditional_methods.ipynb', 'r') as f:
    nb = json.load(f)

# Find the cell with sliding window initialization (cell 15)
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'SlidingWindowDetector' in ''.join(cell['source']):
        # Update with optimized parameters
        cell['source'] = [
            '# Initialize sliding window detector with OPTIMIZED parameters\n',
            'print("Initializing sliding window detector...")\n',
            'print("⚠️  Using optimized parameters to reduce false positives:\\n")\n',
            'print("  - Higher confidence threshold (0.8 vs 0.5)")\n',
            'print("  - Larger step size (48 vs 32) = fewer windows")\n',
            'print("  - Narrower scale range (0.5-1.5 vs 0.3-2.0) = fewer scales")\n',
            'print("  - Higher NMS IoU (0.6) = more aggressive merging\\n")\n',
            '\n',
            'sliding_detector = SlidingWindowDetector(\n',
            '    classifier=hog_svm,\n',
            '    window_size=(64, 64),\n',
            '    step_size=48,          # Larger step = fewer windows (was 32)\n',
            '    scale_factor=1.25,     # Fewer scales between min/max\n',
            '    min_scale=0.5,         # Don\'t search for very small signs (was 0.3)\n',
            '    max_scale=1.5          # Don\'t search for very large signs (was 2.0)\n',
            ')\n',
            '\n',
            '# Initialize detection metrics\n',
            'detection_metrics = DetectionMetrics(num_classes=num_classes, iou_threshold=0.5)\n',
            '\n',
            '# Evaluate on validation set with HIGHER confidence threshold\n',
            'CONFIDENCE_THRESHOLD = 0.80  # Much higher to reduce false positives (was 0.5)\n',
            '\n',
            'print(f"\\nEvaluating on {max_val_images} validation images...")\n',
            'print(f"Using confidence threshold: {CONFIDENCE_THRESHOLD}")\n',
            'print("This may take a while...\\n")\n',
            '\n',
            'n_detected_images = 0\n',
            '\n',
            'for img_path in tqdm(val_images[:max_val_images]):\n',
            '    img = cv2.imread(str(img_path))\n',
            '    if img is None:\n',
            '        continue\n',
            '\n',
            '    h, w = img.shape[:2]\n',
            '\n',
            '    # Load ground truth\n',
            '    label_path = img_path.parent.parent / \'labels\' / f"{img_path.stem}.txt"\n',
            '    gt_boxes, gt_classes = load_yolo_annotations(label_path, w, h)\n',
            '\n',
            '    # Detect with HIGHER confidence threshold\n',
            '    pred_boxes, pred_classes, pred_scores = sliding_detector.detect(\n',
            '        img, \n',
            '        confidence_threshold=CONFIDENCE_THRESHOLD\n',
            '    )\n',
            '\n',
            '    # Convert to numpy arrays\n',
            '    if pred_boxes:\n',
            '        pred_boxes = np.array(pred_boxes)\n',
            '        pred_classes = np.array(pred_classes)\n',
            '        pred_scores = np.array(pred_scores)\n',
            '        n_detected_images += 1\n',
            '    else:\n',
            '        pred_boxes = np.array([]).reshape(0, 4)\n',
            '        pred_classes = np.array([])\n',
            '        pred_scores = np.array([])\n',
            '\n',
            '    # Update metrics\n',
            '    detection_metrics.update(pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes)\n',
            '\n',
            '# Get metrics\n',
            'hog_svm_metrics = detection_metrics.get_metrics_summary()\n',
            '\n',
            'print("\\n" + "="*70)\n',
            'print("HOG+SVM OBJECT DETECTION METRICS (OPTIMIZED)")\n',
            'print("="*70)\n',
            'print(f"mAP@0.5: {hog_svm_metrics[\'mAP\']:.4f}")\n',
            'print(f"Precision: {hog_svm_metrics[\'precision\']:.4f}")\n',
            'print(f"Recall: {hog_svm_metrics[\'recall\']:.4f}")\n',
            'print(f"F1 Score: {hog_svm_metrics[\'f1_score\']:.4f}")\n',
            'print(f"\\nTP: {hog_svm_metrics[\'total_tp\']}, FP: {hog_svm_metrics[\'total_fp\']}, FN: {hog_svm_metrics[\'total_fn\']}")\n',
            'print(f"Images with detections: {n_detected_images}/{max_val_images}")\n',
            '\n',
            'print("\\nPer-class AP:")\n',
            'for class_id in range(num_classes):\n',
            '    ap = hog_svm_metrics[\'per_class\'][f\'class_{class_id}\'][\'ap\']\n',
            '    print(f"  {class_names[class_id]}: {ap:.4f}")\n',
            'print("="*70)\n',
            '\n',
            'print("\\n💡 Note: Traditional methods have inherent limitations:")\n',
            'print("   - Sliding window generates many candidates")\n',
            'print("   - HOG features are less discriminative than deep features")\n',
            'print("   - Performance is typically 10-30x lower than modern methods")\n',
            'print("   - Best used as a baseline or for CPU-only scenarios")\n'
        ]
        break

# Also update visualization cell to use higher threshold
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'Visualize detections' in ''.join(cell['source']):
        cell['source'] = [
            '# Visualize detections with HIGHER confidence threshold\n',
            'n_viz = 6\n',
            'sample_imgs = np.random.choice(val_images[:max_val_images], n_viz, replace=False)\n',
            '\n',
            'fig, axes = plt.subplots(2, 3, figsize=(18, 12))\n',
            'axes = axes.flatten()\n',
            '\n',
            'for idx, img_path in enumerate(sample_imgs):\n',
            '    img = cv2.imread(str(img_path))\n',
            '    if img is None:\n',
            '        continue\n',
            '\n',
            '    # Detect with same high threshold\n',
            '    pred_boxes, pred_classes, pred_scores = sliding_detector.detect(\n',
            '        img, \n',
            '        confidence_threshold=0.80  # High threshold to reduce false positives\n',
            '    )\n',
            '\n',
            '    # Draw\n',
            '    result = img.copy()\n',
            '    for box, cls, score in zip(pred_boxes, pred_classes, pred_scores):\n',
            '        x1, y1, x2, y2 = box\n',
            '        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)\n',
            '        label = f"{class_names[cls]}: {score:.2f}"\n',
            '        cv2.putText(result, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)\n',
            '\n',
            '    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)\n',
            '    axes[idx].imshow(result)\n',
            '    axes[idx].set_title(f"Detected: {len(pred_boxes)} signs (conf>0.8)", fontsize=10)\n',
            '    axes[idx].axis(\'off\')\n',
            '\n',
            'plt.tight_layout()\n',
            'plt.show()\n',
            '\n',
            'print("\\n💡 With higher confidence threshold (0.8), we see fewer false positives")\n',
            'print("   but may miss some true positives (lower recall).")\n'
        ]
        break

# Save
with open('notebooks/02_traditional_methods.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("✅ Updated notebook with optimized detector parameters!")
print("\nKey changes:")
print("  - Confidence threshold: 0.5 → 0.8")
print("  - Step size: 32 → 48 (fewer windows)")
print("  - Scale range: [0.3, 2.0] → [0.5, 1.5] (fewer scales)")
print("  - NMS IoU: 0.3 → 0.5 (improved in code)")
print("\nThis should drastically reduce false positives!")
