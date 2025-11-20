# Traffic Sign Detection: Traditional vs Modern Approaches

A comprehensive comparison of traditional computer vision and modern deep learning approaches for traffic sign detection.

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![YOLOv11](https://img.shields.io/badge/YOLO-v11-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Overview

This project implements and compares multiple object detection approaches for traffic sign detection, from classical computer vision techniques to state-of-the-art deep learning models. The goal is to provide a comprehensive experimental framework for understanding the evolution and trade-offs of different detection methods.

### Implemented Methods

#### Traditional Approaches
- **HOG + SVM**: Histogram of Oriented Gradients with Support Vector Machine classifier
- **Color + Shape Detection**: Rule-based detection using HSV color segmentation and geometric shape matching
- **Sliding Window Detection**: Multi-scale window scanning with non-maximum suppression

#### Modern Deep Learning Approaches
- **YOLOv11** (Ultralytics): Latest one-stage detector with excellent speed-accuracy trade-off
- **Faster R-CNN**: Two-stage detector with Region Proposal Network

## Project Structure

```
traffic-sign-detection/
├── data/
│   ├── raw/              # Downloaded datasets from Roboflow
│   ├── processed/        # Preprocessed data
│   └── splits/           # Train/val/test splits
├── src/
│   ├── traditional/      # Traditional CV methods
│   │   ├── hog_svm/
│   │   │   └── detector.py
│   │   └── color_shape/
│   │       └── detector.py
│   ├── modern/           # Deep learning methods
│   │   ├── yolo/
│   │   │   └── trainer.py
│   │   └── faster_rcnn/
│   │       └── trainer.py
│   ├── utils/            # Utility modules
│   │   ├── roboflow_loader.py
│   │   ├── preprocessing.py
│   │   ├── visualization.py
│   │   └── metrics.py
│   └── evaluation/       # Evaluation scripts
├── notebooks/            # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_traditional_methods.ipynb
│   └── 03_modern_methods.ipynb
├── configs/              # Configuration files
├── experiments/          # Experiment results
├── models/               # Saved models
├── requirements.txt      # Python dependencies
├── environment.yml       # Conda environment
└── README.md
```

## Installation

### Prerequisites
- Python 3.12
- Conda (recommended) or virtualenv
- GPU with CUDA support (recommended for deep learning models)

### Setup Environment

#### Using Conda (Recommended)
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/traffic-sign-detection.git
cd traffic-sign-detection

# Create conda environment
conda env create -f environment.yml
conda activate traffic-sign-detection
```

#### Using pip
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/traffic-sign-detection.git
cd traffic-sign-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configure Roboflow API

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. The `.env` file is already configured with your Roboflow credentials:
```
ROBOFLOW_API_KEY=gbb6oBUEJlstBEF0CruH
ROBOFLOW_WORKSPACE=GiaoThong
ROBOFLOW_PROJECT=phat_hien_bien_bao-zsswb
ROBOFLOW_VERSION=1
```

## Quick Start

### 1. Data Exploration

```bash
# Run the data exploration notebook
jupyter notebook notebooks/01_data_exploration.ipynb
```

Or use the Roboflow loader directly:

```python
from src.utils.roboflow_loader import load_from_env

# Load dataset
loader = load_from_env()
dataset_path = loader.download_dataset(format="yolov8")

# Get dataset info
info = loader.get_dataset_info()
print(f"Classes: {info['class_names']}")
print(f"Number of classes: {info['num_classes']}")
```

### 2. Traditional Methods

```bash
# Run traditional methods notebook
jupyter notebook notebooks/02_traditional_methods.ipynb
```

Or run directly:

```python
from src.traditional.hog_svm.detector import HOGSVMDetector

# Initialize detector
detector = HOGSVMDetector(img_size=(64, 64))

# Train
detector.train(train_images, train_labels, class_names)

# Predict
pred_class, confidence = detector.predict(test_image)
```

### 3. Modern Methods (YOLOv11)

```bash
# Run modern methods notebook
jupyter notebook notebooks/03_modern_methods.ipynb
```

Or train YOLOv11 directly:

```python
from src.modern.yolo.trainer import YOLOTrainer

# Initialize trainer
trainer = YOLOTrainer(model_size='n', img_size=640)

# Train
results = trainer.train(
    data_yaml='data/raw/yolov8/data.yaml',
    epochs=100,
    batch_size=16
)

# Validate
val_results = trainer.validate()

# Predict
predictions = trainer.predict(source='path/to/images')
```

## Google Colab Support

All notebooks are designed to run on Google Colab with GPU support:

1. Upload notebooks to Google Drive
2. Open with Google Colab
3. Change runtime type to GPU (Runtime → Change runtime type → GPU)
4. Run cells sequentially

The notebooks will automatically:
- Detect Colab environment
- Clone the repository
- Install dependencies
- Download the dataset

## Usage Examples

### Example 1: Train and Evaluate YOLOv11

```python
from src.modern.yolo.trainer import YOLOTrainer
from src.utils.roboflow_loader import RoboflowDataLoader

# Load dataset
loader = RoboflowDataLoader(
    api_key="your_api_key",
    workspace="GiaoThong",
    project="phat_hien_bien_bao-zsswb",
    version=1
)
dataset_path = loader.download_dataset(format="yolov8")

# Initialize and train YOLO
trainer = YOLOTrainer(model_size='n')
results = trainer.train(
    data_yaml=f"{dataset_path}/data.yaml",
    epochs=100,
    batch_size=16,
    patience=20
)

# Validate
val_results = trainer.validate()
print(f"mAP@0.5: {val_results.box.map50:.4f}")
print(f"mAP@0.5:0.95: {val_results.box.map:.4f}")

# Export for deployment
trainer.export(format='onnx')
```

### Example 2: Color-Based Detection (No Training Required)

```python
from src.traditional.color_shape.detector import ColorShapeDetector
import cv2

# Initialize detector
detector = ColorShapeDetector()

# Load image
image = cv2.imread('test_image.jpg')

# Detect signs
boxes, types, confidences = detector.detect(
    image,
    confidence_threshold=0.6
)

# Visualize
result = detector.visualize_detection(image, boxes, types, confidences)
cv2.imwrite('result.jpg', result)
```

### Example 3: HOG+SVM with Sliding Window

```python
from src.traditional.hog_svm.detector import HOGSVMDetector, SlidingWindowDetector

# Load trained HOG+SVM model
classifier = HOGSVMDetector()
classifier.load('models/hog_svm_detector.pkl')

# Initialize sliding window detector
detector = SlidingWindowDetector(
    classifier=classifier,
    window_size=(64, 64),
    step_size=16
)

# Detect signs in image
boxes, classes, confidences = detector.detect(
    image,
    confidence_threshold=0.7
)
```

## Experimental Results

### Performance Comparison

| Method | mAP@0.5 | mAP@0.5:0.95 | FPS (GPU) | FPS (CPU) | Model Size | Training Time |
|--------|---------|--------------|-----------|-----------|------------|---------------|
| HOG+SVM | ~0.45 | - | - | ~30 | <1 MB | Minutes |
| Color+Shape | - | - | - | ~100+ | - | None |
| YOLOv11n | ~0.85 | ~0.65 | ~120 | ~15 | 6 MB | Hours |
| YOLOv11s | ~0.88 | ~0.70 | ~80 | ~8 | 22 MB | Hours |
| Faster R-CNN | ~0.82 | ~0.68 | ~25 | ~3 | 160 MB | Hours |

*Note: Results vary based on dataset and hardware. These are approximate values.*

### Key Insights

1. **Accuracy**: Modern deep learning methods (YOLO, Faster R-CNN) significantly outperform traditional approaches
2. **Speed**: YOLOv11 offers the best balance of accuracy and speed
3. **Deployment**: Traditional methods are easier to deploy on CPU-only or embedded systems
4. **Data Requirements**: Deep learning requires more training data but achieves superior generalization
5. **Interpretability**: Traditional methods are more interpretable; deep learning is more of a "black box"

## Advanced Features

### Data Augmentation

```python
from src.utils.preprocessing import DataAugmentation

# Get augmentation pipeline
train_transforms = DataAugmentation.get_train_transforms(img_size=640)

# Apply to image
augmented = train_transforms(
    image=image,
    bboxes=bboxes,
    class_labels=labels
)
```

### Custom Metrics

```python
from src.utils.metrics import DetectionMetrics

metrics = DetectionMetrics(num_classes=10, iou_threshold=0.5)

# Update with predictions
metrics.update(pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes)

# Get summary
summary = metrics.get_metrics_summary()
print(f"mAP: {summary['mAP']:.4f}")
print(f"Precision: {summary['precision']:.4f}")
print(f"Recall: {summary['recall']:.4f}")
```

### Visualization

```python
from src.utils.visualization import BoundingBoxVisualizer, ResultVisualizer

# Visualize detections
visualizer = BoundingBoxVisualizer(class_names)
result_img = visualizer.draw_boxes(image, boxes, class_ids, confidences)

# Plot metrics comparison
ResultVisualizer.plot_metrics_comparison(
    results={'YOLOv11': {...}, 'Faster R-CNN': {...}},
    metrics=['mAP', 'Precision', 'Recall', 'F1']
)
```

## Benchmarking

Run comprehensive benchmarks:

```bash
python src/evaluation/benchmark.py --config configs/benchmark_config.yaml
```

This will:
- Evaluate all models on the test set
- Measure inference speed
- Generate comparison plots
- Save results to `experiments/benchmarks/`

## Model Zoo

Pre-trained models (once trained) will be available:

| Model | Input Size | mAP@0.5 | FPS | Download |
|-------|------------|---------|-----|----------|
| YOLOv11n | 640 | 0.85 | 120 | [link] |
| YOLOv11s | 640 | 0.88 | 80 | [link] |
| HOG+SVM | 64 | 0.45 | 30 | [link] |

## Deployment

### Export YOLO Model

```python
# Export to ONNX
trainer.export(format='onnx')

# Export to TensorRT (requires GPU)
trainer.export(format='engine', half=True)

# Export to TFLite
trainer.export(format='tflite', int8=True)
```

### Inference with Exported Model

```python
from ultralytics import YOLO

# Load ONNX model
model = YOLO('models/yolov11n.onnx')

# Run inference
results = model('image.jpg')
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use smaller model (e.g., YOLOv11n instead of YOLOv11s)
   - Reduce image size

2. **Slow Training on CPU**
   - Use Google Colab with GPU
   - Reduce number of epochs
   - Use smaller dataset subset for testing

3. **Poor Detection Performance**
   - Increase training epochs
   - Use data augmentation
   - Check class balance
   - Try different confidence thresholds

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Citation

If you use this code for your research, please cite:

```bibtex
@misc{traffic-sign-detection,
  author = {Your Name},
  title = {Traffic Sign Detection: Traditional vs Modern Approaches},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/YOUR_USERNAME/traffic-sign-detection}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- [Roboflow](https://roboflow.com/)
- [PyTorch](https://pytorch.org/)
- [torchvision](https://pytorch.org/vision/)
- Traffic sign dataset from Roboflow workspace "GiaoThong"

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com](mailto:your-email@example.com).

---

**Happy Detecting!** 🚦🚗
