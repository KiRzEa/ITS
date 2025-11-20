# Experimental Design: Traffic Sign Detection

## Research Question

**How do traditional computer vision methods compare to modern deep learning approaches for traffic sign detection in terms of accuracy, speed, and practical deployment?**

## Objectives

1. Implement representative traditional computer vision methods (HOG+SVM, Color+Shape)
2. Implement state-of-the-art deep learning methods (YOLOv11, Faster R-CNN)
3. Evaluate and compare methods across multiple metrics
4. Provide practical insights for real-world deployment

## Experimental Setup

### Dataset

**Source**: Roboflow - Traffic Sign Detection Dataset
- **Workspace**: GiaoThong
- **Project**: phat_hien_bien_bao-zsswb
- **Version**: 1

**Characteristics**:
- Real-world traffic sign images
- Multiple classes (stop, yield, speed limits, etc.)
- Various lighting and weather conditions
- Different scales and viewpoints

**Splits**:
- Training: ~70%
- Validation: ~20%
- Test: ~10%

### Methods

#### Traditional Approaches

##### 1. HOG + SVM (Histogram of Oriented Gradients + Support Vector Machine)

**Concept**: Extract gradient-based features and classify with SVM

**Configuration**:
```python
{
    'img_size': (64, 64),
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'svm_kernel': 'rbf',
    'svm_c': 10.0
}
```

**Process**:
1. Extract ROIs from images using ground truth
2. Resize to fixed size (64x64)
3. Extract HOG features
4. Train SVM classifier
5. Use sliding window for detection

**Advantages**:
- Fast inference on CPU
- Interpretable features
- Low memory footprint
- No need for large datasets

**Disadvantages**:
- Manual feature engineering
- Sensitive to scale and rotation
- Lower accuracy than deep learning
- Sliding window is computationally expensive

##### 2. Color + Shape Detection

**Concept**: Use HSV color ranges and geometric shape matching

**Configuration**:
```python
{
    'color_ranges': {
        'red': [(0, 100, 100), (10, 255, 255)] + [(160, 100, 100), (180, 255, 255)],
        'blue': [(100, 100, 100), (130, 255, 255)],
        'yellow': [(20, 100, 100), (30, 255, 255)]
    },
    'min_area': 500,
    'confidence_threshold': 0.6
}
```

**Process**:
1. Convert image to HSV color space
2. Apply color-based segmentation
3. Find contours
4. Classify based on shape (circle, triangle, octagon, etc.)
5. Rule-based classification (e.g., red octagon = stop sign)

**Advantages**:
- Very fast (real-time on CPU)
- No training required
- Simple and interpretable
- Works well in controlled conditions

**Disadvantages**:
- Sensitive to lighting conditions
- Limited to standard colors/shapes
- High false positive rate in complex scenes
- Not robust to occlusion

#### Modern Deep Learning Approaches

##### 1. YOLOv11 (You Only Look Once v11)

**Concept**: Single-stage end-to-end object detector

**Variants Tested**:
- **YOLOv11n** (Nano): 3.2M params, fastest
- **YOLOv11s** (Small): 11.2M params, balanced
- **YOLOv11m** (Medium): 25.9M params, more accurate

**Configuration**:
```python
{
    'img_size': 640,
    'epochs': 100,
    'batch_size': 16,
    'learning_rate': 0.01,
    'optimizer': 'auto',
    'patience': 20,
    'augmentation': True,
    'cos_lr': True,
    'warmup_epochs': 3
}
```

**Architecture**:
- Backbone: CSPDarknet (modified)
- Neck: PANet with C2f blocks
- Head: Decoupled detection head
- Loss: DFL + BCE + CIoU

**Advantages**:
- Excellent speed-accuracy trade-off
- End-to-end training
- Handles multi-scale objects
- Pre-trained weights available
- Easy to deploy (ONNX, TensorRT)

**Disadvantages**:
- Requires GPU for training
- Black box (less interpretable)
- Needs substantial training data
- Memory intensive

##### 2. Faster R-CNN (Region-based Convolutional Neural Network)

**Concept**: Two-stage detector with region proposals

**Configuration**:
```python
{
    'backbone': 'ResNet50-FPN',
    'pretrained': True,
    'epochs': 50,
    'batch_size': 4,
    'learning_rate': 0.005,
    'momentum': 0.9,
    'weight_decay': 0.0005
}
```

**Architecture**:
- Backbone: ResNet50 with FPN
- RPN: Region Proposal Network
- ROI Head: Fast R-CNN detector
- Loss: Classification + Regression

**Advantages**:
- High accuracy (especially for small objects)
- Well-established architecture
- Good localization
- Pre-trained weights available

**Disadvantages**:
- Slower than YOLO
- More complex architecture
- Higher memory usage
- Longer training time

### Evaluation Metrics

#### Detection Metrics

1. **mAP@0.5** (Mean Average Precision at IoU=0.5)
   - Primary metric for detection accuracy
   - Standard PASCAL VOC metric

2. **mAP@0.5:0.95** (COCO metric)
   - Average mAP across IoU thresholds 0.5 to 0.95
   - More strict than mAP@0.5

3. **Precision**
   - TP / (TP + FP)
   - Measures false positive rate

4. **Recall**
   - TP / (TP + FN)
   - Measures detection completeness

5. **F1-Score**
   - Harmonic mean of precision and recall
   - Balance between precision and recall

#### Speed Metrics

1. **FPS** (Frames Per Second)
   - Inference speed on GPU/CPU
   - Real-time capability indicator

2. **Inference Time**
   - Average time per image (ms)
   - Includes pre/post-processing

#### Model Metrics

1. **Model Size** (MB)
   - Storage requirements
   - Important for edge deployment

2. **Parameters** (millions)
   - Model complexity
   - Memory footprint

3. **FLOPs** (Giga floating-point operations)
   - Computational complexity

### Experimental Procedure

#### Phase 1: Data Preparation
1. Download dataset from Roboflow
2. Analyze class distribution
3. Visualize sample images
4. Calculate statistics
5. Split into train/val/test

#### Phase 2: Traditional Methods
1. **HOG+SVM**:
   - Extract training patches
   - Train SVM classifier
   - Evaluate on validation set
   - Test with sliding window

2. **Color+Shape**:
   - Define color ranges
   - Test on sample images
   - Evaluate performance
   - Analyze failure cases

#### Phase 3: Modern Methods
1. **YOLOv11**:
   - Train multiple sizes (n, s, m)
   - Monitor training curves
   - Validate on test set
   - Export to ONNX
   - Measure inference speed

2. **Faster R-CNN**:
   - Prepare COCO format data
   - Train with transfer learning
   - Evaluate on test set
   - Compare with YOLO

#### Phase 4: Comparison & Analysis
1. Compile all metrics
2. Create comparison tables
3. Generate visualization plots
4. Analyze trade-offs
5. Provide recommendations

## Expected Results

### Hypothesis

**H1**: Modern deep learning methods will achieve significantly higher accuracy (mAP) than traditional methods.

**H2**: YOLOv11 will offer the best balance between accuracy and speed among deep learning methods.

**H3**: Traditional methods will be faster on CPU and easier to deploy but with lower accuracy.

**H4**: Color+Shape detection will have the fastest inference but lowest accuracy.

### Predicted Performance

| Method | mAP@0.5 | FPS (GPU) | FPS (CPU) | Size |
|--------|---------|-----------|-----------|------|
| Color+Shape | 0.30-0.40 | - | 100+ | - |
| HOG+SVM | 0.40-0.55 | - | 20-30 | <1MB |
| YOLOv11n | 0.80-0.90 | 100-150 | 10-20 | 6MB |
| YOLOv11s | 0.85-0.92 | 70-100 | 5-10 | 22MB |
| Faster R-CNN | 0.82-0.90 | 20-30 | 2-5 | 160MB |

## Variables

### Independent Variables
- **Detection Method**: HOG+SVM, Color+Shape, YOLOv11, Faster R-CNN
- **Model Configuration**: Hyperparameters, architecture variants
- **Training Data**: Amount, augmentation strategies

### Dependent Variables
- **Accuracy**: mAP, Precision, Recall, F1
- **Speed**: FPS, inference time
- **Resources**: Model size, memory usage, training time

### Controlled Variables
- Dataset (same for all methods)
- Test set (same evaluation data)
- Hardware (consistent benchmarking)
- Image resolution (640x640 for deep learning)
- Confidence threshold (0.25 for fair comparison)

## Analysis Plan

### Quantitative Analysis

1. **Performance Comparison**
   - Statistical significance testing (t-test, ANOVA)
   - Confidence intervals for metrics
   - Per-class performance analysis

2. **Speed-Accuracy Trade-off**
   - Scatter plots (FPS vs mAP)
   - Pareto frontier analysis
   - Efficiency ratio (accuracy/time)

3. **Ablation Studies**
   - Model size impact (YOLO n/s/m)
   - Augmentation effectiveness
   - Backbone comparison

### Qualitative Analysis

1. **Visual Inspection**
   - Successful detection examples
   - Failure case analysis
   - Confusion patterns

2. **Practical Considerations**
   - Deployment complexity
   - Hardware requirements
   - Training requirements
   - Maintenance needs

## Reproducibility

### Seeds and Randomization
```python
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
```

### Environment
- Python 3.12
- PyTorch 2.0+
- CUDA 11.8+ (for GPU)
- Documented in `requirements.txt` and `environment.yml`

### Code Organization
- Modular design
- Clear documentation
- Jupyter notebooks for experiments
- Configuration files for reproducibility

## Ethical Considerations

1. **Dataset**: Publicly available, properly licensed
2. **Use Case**: Safety-critical application (traffic signs)
3. **Bias**: Check for geographic/environmental biases
4. **Deployment**: Validate thoroughly before real-world use

## Limitations

1. **Dataset Size**: Limited to Roboflow dataset
2. **Geographic Scope**: May not generalize to all countries
3. **Weather Conditions**: Dataset may not cover all scenarios
4. **Computational Resources**: GPU training time constraints
5. **Time Constraints**: Cannot test all possible configurations

## Future Work

1. **Additional Methods**:
   - Transformer-based detectors (DETR, DINO)
   - Lightweight models (MobileNet-SSD, EfficientDet)
   - Ensemble methods

2. **Advanced Techniques**:
   - Knowledge distillation
   - Model quantization
   - Neural architecture search

3. **Real-World Testing**:
   - Video datasets
   - Real-time camera feeds
   - Edge device deployment

4. **Domain Adaptation**:
   - Different countries/regions
   - Different weather conditions
   - Night-time detection

## Timeline

| Phase | Duration | Tasks |
|-------|----------|-------|
| Setup | 1 day | Environment, data download |
| Data Analysis | 1 day | EDA, visualization |
| Traditional Methods | 2 days | HOG+SVM, Color+Shape |
| Modern Methods | 3-5 days | YOLOv11, Faster R-CNN training |
| Evaluation | 1 day | Metrics, comparison |
| Documentation | 1 day | Report, presentation |

**Total**: 1-2 weeks (depending on GPU availability)

## References

1. Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection.
2. Redmon, J., et al. (2016). You only look once: Unified, real-time object detection.
3. Ren, S., et al. (2015). Faster r-cnn: Towards real-time object detection with region proposal networks.
4. Ultralytics YOLOv11: https://github.com/ultralytics/ultralytics
5. Roboflow: https://roboflow.com/

---

**Last Updated**: 2025-01-18
**Status**: Ready for execution
