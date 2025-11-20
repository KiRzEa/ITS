# Setup Guide - Traffic Sign Detection

This guide will help you set up the project environment and run your first experiments.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Verify Installation](#verify-installation)
4. [Running Experiments](#running-experiments)
5. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS, or Linux
- **Python**: 3.12
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space

### Recommended for Deep Learning
- **GPU**: NVIDIA GPU with 6GB+ VRAM (GTX 1660 or better)
- **CUDA**: 11.8 or later
- **cuDNN**: Compatible version with CUDA
- **RAM**: 16GB+

### For Google Colab Users
- Free GPU access available (T4 GPU with 15GB VRAM)
- No local installation required for notebooks

## Installation

### Option 1: Local Installation (Conda - Recommended)

#### Step 1: Install Conda
If you don't have Conda installed:
- Download Miniconda: https://docs.conda.io/en/latest/miniconda.html
- Or Anaconda: https://www.anaconda.com/download

#### Step 2: Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/traffic-sign-detection.git
cd traffic-sign-detection
```

#### Step 3: Create Environment
```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate traffic-sign-detection
```

#### Step 4: Verify Installation
```bash
# Check Python version
python --version  # Should show Python 3.12.x

# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Option 2: Local Installation (pip + venv)

#### Step 1: Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/traffic-sign-detection.git
cd traffic-sign-detection
```

#### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Option 3: Google Colab (No Installation)

1. Go to [Google Colab](https://colab.research.google.com/)
2. Upload notebooks from `notebooks/` folder
3. Change runtime type to GPU:
   - Runtime → Change runtime type → Hardware accelerator: GPU → Save
4. Run the first cell to clone repo and install dependencies

## Verify Installation

### Quick Test Script

Create a file `test_installation.py`:

```python
#!/usr/bin/env python
"""Test script to verify installation"""

print("Testing installation...")

# Test imports
try:
    import numpy as np
    print("✓ NumPy")
except ImportError as e:
    print(f"✗ NumPy: {e}")

try:
    import cv2
    print("✓ OpenCV")
except ImportError as e:
    print(f"✗ OpenCV: {e}")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA {torch.version.cuda}")
        print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  ⚠ CUDA not available (CPU only)")
except ImportError as e:
    print(f"✗ PyTorch: {e}")

try:
    import torchvision
    print(f"✓ Torchvision {torchvision.__version__}")
except ImportError as e:
    print(f"✗ Torchvision: {e}")

try:
    from ultralytics import YOLO
    print("✓ Ultralytics (YOLO)")
except ImportError as e:
    print(f"✗ Ultralytics: {e}")

try:
    from roboflow import Roboflow
    print("✓ Roboflow")
except ImportError as e:
    print(f"✗ Roboflow: {e}")

try:
    import sklearn
    print(f"✓ Scikit-learn {sklearn.__version__}")
except ImportError as e:
    print(f"✗ Scikit-learn: {e}")

print("\nInstallation test complete!")
```

Run it:
```bash
python test_installation.py
```

Expected output:
```
Testing installation...
✓ NumPy
✓ OpenCV
✓ PyTorch 2.x.x
  ✓ CUDA 11.8
  ✓ GPU: NVIDIA GeForce RTX ...
✓ Torchvision 0.x.x
✓ Ultralytics (YOLO)
✓ Roboflow
✓ Scikit-learn 1.x.x

Installation test complete!
```

## Running Experiments

### Quick Start (Automated)

Run the quick start script for a 10-epoch demo:

```bash
python quick_start.py
```

This will:
1. Download the dataset from Roboflow
2. Train YOLOv11n for 10 epochs (demo)
3. Validate the model
4. Export to ONNX format

Expected time: ~10-30 minutes (depending on GPU)

### Step-by-Step Experiments

#### 1. Data Exploration

```bash
# Start Jupyter
jupyter notebook

# Open and run:
notebooks/01_data_exploration.ipynb
```

This notebook will:
- Download dataset from Roboflow
- Analyze class distribution
- Visualize sample images
- Calculate dataset statistics
- Provide recommendations

Expected time: ~5-10 minutes

#### 2. Traditional Methods

```bash
# Run notebook:
notebooks/02_traditional_methods.ipynb
```

This notebook will:
- Train HOG+SVM detector
- Test color+shape detection
- Evaluate performance
- Compare with modern methods

Expected time: ~15-30 minutes (CPU only)

#### 3. Modern Methods (Requires GPU)

```bash
# Run notebook:
notebooks/03_modern_methods.ipynb
```

This notebook will:
- Train YOLOv11 (multiple sizes)
- Optionally train Faster R-CNN
- Validate on test set
- Export models
- Compare all methods

Expected time:
- With GPU: 1-3 hours (full training)
- With Colab: 1-2 hours
- CPU only: Not recommended (very slow)

## Project Workflow

```
1. Data Exploration (01_data_exploration.ipynb)
   ↓
2. Traditional Methods (02_traditional_methods.ipynb)
   ↓
3. Modern Methods (03_modern_methods.ipynb)
   ↓
4. Analysis & Comparison
```

## Directory Structure After Running

```
traffic-sign-detection/
├── data/
│   └── raw/
│       ├── yolov8/          # Dataset in YOLO format
│       └── coco/            # Dataset in COCO format (optional)
├── experiments/
│   ├── yolo/
│   │   ├── yolov11n_traffic_signs/
│   │   │   ├── weights/
│   │   │   │   ├── best.pt
│   │   │   │   └── last.pt
│   │   │   └── results.png
│   │   └── predictions/
│   └── faster_rcnn/
│       ├── best_model.pth
│       └── training_history.json
├── models/
│   ├── yolov11n.onnx        # Exported ONNX model
│   └── hog_svm_detector.pkl # Traditional model
└── plots/                    # Generated visualizations
```

## Troubleshooting

### Issue 1: CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# Reduce batch size
trainer.train(batch_size=8)  # Instead of 16

# Use smaller model
trainer = YOLOTrainer(model_size='n')  # Instead of 's' or 'm'

# Reduce image size
trainer = YOLOTrainer(img_size=416)  # Instead of 640
```

### Issue 2: Slow Training on CPU

**Problem**: Training takes too long without GPU

**Solutions**:
1. Use Google Colab (free GPU)
2. Reduce epochs for testing: `epochs=10`
3. Use traditional methods (fast on CPU)
4. Rent cloud GPU (AWS, GCP, etc.)

### Issue 3: Roboflow API Error

**Error**: `roboflow.core.exceptions.APIKeyError`

**Solutions**:
1. Check `.env` file exists and has correct API key
2. Verify internet connection
3. Check Roboflow workspace/project names
4. Create new API key on Roboflow dashboard

### Issue 4: Import Errors

**Error**: `ModuleNotFoundError: No module named 'xxx'`

**Solutions**:
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Or install specific package
pip install package-name

# For conda
conda install package-name
```

### Issue 5: Jupyter Kernel Issues

**Error**: Kernel crashes or doesn't start

**Solutions**:
```bash
# Install/update ipykernel
pip install --upgrade ipykernel

# Register kernel
python -m ipykernel install --user --name traffic-sign-detection

# Then select kernel in Jupyter:
# Kernel → Change kernel → traffic-sign-detection
```

### Issue 6: Permission Errors (Windows)

**Error**: `PermissionError: [WinError 5]`

**Solutions**:
1. Run as Administrator
2. Or change permissions on project folder
3. Or use different directory (e.g., Documents instead of Desktop)

## GPU Setup (Optional but Recommended)

### Check CUDA Installation

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version
```

### Install CUDA Toolkit (if needed)

1. Download from: https://developer.nvidia.com/cuda-downloads
2. Install for your OS
3. Restart computer
4. Verify with `nvidia-smi`

### Install PyTorch with CUDA

```bash
# For CUDA 11.8
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

## Tips for Best Results

### Training Tips

1. **Start small**: Use YOLOv11n for initial experiments
2. **Monitor training**: Watch loss curves and validation metrics
3. **Use early stopping**: Set `patience=20` to avoid overfitting
4. **Data augmentation**: Already enabled by default
5. **Checkpoints**: Models save automatically every 10 epochs

### Dataset Tips

1. **Check class balance**: Use data exploration notebook
2. **Augmentation**: Helps with imbalanced classes
3. **Validation split**: Default 70/20/10 train/val/test is good
4. **Image quality**: Remove blurry/corrupted images if any

### Performance Tips

1. **GPU utilization**: Monitor with `nvidia-smi`
2. **Batch size**: Maximize based on GPU memory
3. **Mixed precision**: Enabled by default (AMP)
4. **DataLoader workers**: Set `num_workers=4` for faster loading

## Next Steps

After successful setup:

1. ✅ Run `quick_start.py` for automated demo
2. ✅ Explore dataset with `01_data_exploration.ipynb`
3. ✅ Try traditional methods with `02_traditional_methods.ipynb`
4. ✅ Train deep learning models with `03_modern_methods.ipynb`
5. ✅ Compare all approaches
6. ✅ Export best model for deployment

## Getting Help

- **Issues**: Open GitHub issue
- **Documentation**: Check README.md
- **Code examples**: See notebooks/
- **Community**: Stack Overflow, PyTorch forums

---

**Happy coding! 🚀**
