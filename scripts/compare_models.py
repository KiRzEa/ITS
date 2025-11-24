"""
Model Comparison Script
Compare different model architectures for traffic sign detection
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.modern.yolo.trainer import YOLOTrainer
from src.modern.faster_rcnn.trainer import FasterRCNNTrainer
from src.modern.ssd.trainer import SSDTrainer
from src.modern.retinanet.trainer import RetinaNetTrainer


def get_model_size(model):
    """Get model size in MB and parameter count"""
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'size_mb': size_mb,
        'total_params': total_params,
        'trainable_params': trainable_params
    }


def benchmark_yolo_models():
    """Benchmark different YOLO models"""
    print("\n" + "="*70)
    print("YOLO Models Comparison")
    print("="*70)
    print(f"{'Model':<20} {'Params':<15} {'Size (MB)':<12} {'Status':<15}")
    print("-"*70)

    versions = ['v5', 'v8', 'v11']
    sizes = ['n', 's', 'm']

    results = []

    for version in versions:
        for size in sizes:
            try:
                trainer = YOLOTrainer(
                    model_version=version,
                    model_size=size,
                    img_size=640,
                    device='cpu'  # Use CPU for fair comparison
                )

                # Get model info
                model_name = f"YOLO{version}-{size}"

                # Ultralytics models don't expose PyTorch model directly for param count
                # So we use approximate values
                param_map = {
                    'v5': {'n': 1.9, 's': 7.2, 'm': 21.2, 'l': 46.5, 'x': 86.7},
                    'v8': {'n': 3.2, 's': 11.2, 'm': 25.9, 'l': 43.7, 'x': 68.2},
                    'v11': {'n': 2.6, 's': 9.4, 'm': 20.1, 'l': 25.3, 'x': 56.9}
                }

                params_m = param_map[version][size]
                size_mb = params_m * 4  # Rough estimate (4 bytes per param)

                print(f"{model_name:<20} {params_m:>6.1f}M{'':<7} {size_mb:>6.1f}{'':<5} {'✓ Available':<15}")

                results.append({
                    'model': model_name,
                    'params_m': params_m,
                    'size_mb': size_mb
                })

            except Exception as e:
                print(f"{model_name:<20} {'N/A':<15} {'N/A':<12} {'✗ Error':<15}")

    print("-"*70)
    return results


def benchmark_faster_rcnn_models():
    """Benchmark different Faster R-CNN backbones"""
    print("\n" + "="*70)
    print("Faster R-CNN Backbones Comparison")
    print("="*70)
    print(f"{'Backbone':<25} {'Params':<15} {'Size (MB)':<12} {'Status':<15}")
    print("-"*70)

    backbones = [
        'resnet18',
        'resnet34',
        'resnet50',
        'mobilenet_v3_large',
        'mobilenet_v3_large_320'
    ]

    results = []

    for backbone in backbones:
        try:
            trainer = FasterRCNNTrainer(
                num_classes=4,
                backbone=backbone,
                pretrained=False,  # Don't download weights for comparison
                device='cpu'
            )

            # Get model size
            info = get_model_size(trainer.model)
            params_m = info['total_params'] / 1e6
            size_mb = info['size_mb']

            print(f"{backbone:<25} {params_m:>6.1f}M{'':<7} {size_mb:>6.1f}{'':<5} {'✓ Available':<15}")

            results.append({
                'backbone': backbone,
                'params_m': params_m,
                'size_mb': size_mb
            })

        except Exception as e:
            print(f"{backbone:<25} {'N/A':<15} {'N/A':<12} {'✗ Error: ' + str(e)[:10]:<15}")

    print("-"*70)
    return results


def benchmark_ssd_models():
    """Benchmark different SSD models"""
    print("\n" + "="*70)
    print("SSD Models Comparison")
    print("="*70)
    print(f"{'Backbone':<25} {'Params':<15} {'Size (MB)':<12} {'Status':<15}")
    print("-"*70)

    backbones = ['vgg16', 'mobilenet_v3']

    results = []

    for backbone in backbones:
        try:
            trainer = SSDTrainer(
                num_classes=4,
                backbone=backbone,
                pretrained=False,
                device='cpu'
            )

            # Get model size
            info = get_model_size(trainer.model)
            params_m = info['total_params'] / 1e6
            size_mb = info['size_mb']

            print(f"{backbone:<25} {params_m:>6.1f}M{'':<7} {size_mb:>6.1f}{'':<5} {'✓ Available':<15}")

            results.append({
                'backbone': backbone,
                'params_m': params_m,
                'size_mb': size_mb
            })

        except Exception as e:
            print(f"{backbone:<25} {'N/A':<15} {'N/A':<12} {'✗ Error: ' + str(e)[:10]:<15}")

    print("-"*70)
    return results


def benchmark_retinanet_models():
    """Benchmark different RetinaNet models"""
    print("\n" + "="*70)
    print("RetinaNet Models Comparison")
    print("="*70)
    print(f"{'Backbone':<25} {'Params':<15} {'Size (MB)':<12} {'Status':<15}")
    print("-"*70)

    backbones = ['resnet18', 'resnet34', 'resnet50', 'mobilenet_v3']

    results = []

    for backbone in backbones:
        try:
            trainer = RetinaNetTrainer(
                num_classes=4,
                backbone=backbone,
                pretrained=False,
                device='cpu'
            )

            # Get model size
            info = get_model_size(trainer.model)
            params_m = info['total_params'] / 1e6
            size_mb = info['size_mb']

            print(f"{backbone:<25} {params_m:>6.1f}M{'':<7} {size_mb:>6.1f}{'':<5} {'✓ Available':<15}")

            results.append({
                'backbone': backbone,
                'params_m': params_m,
                'size_mb': size_mb
            })

        except Exception as e:
            print(f"{backbone:<25} {'N/A':<15} {'N/A':<12} {'✗ Error: ' + str(e)[:10]:<15}")

    print("-"*70)
    return results


def print_recommendations():
    """Print model recommendations"""
    print("\n" + "="*70)
    print("Recommendations Based on Resources")
    print("="*70)

    recommendations = [
        {
            'scenario': 'Very Limited Resources (CPU only, <4GB RAM)',
            'yolo': 'YOLOv5-n or YOLOv8-n with img_size=416',
            'ssd': 'SSDLite MobileNetV3 (best for CPU)',
            'retinanet': 'Skip or MobileNetV3',
            'frcnn': 'Skip Faster R-CNN'
        },
        {
            'scenario': 'Limited GPU (2-4GB VRAM)',
            'yolo': 'YOLOv8-n or YOLOv11-n with batch_size=8',
            'ssd': 'SSDLite MobileNetV3 with batch_size=16',
            'retinanet': 'MobileNetV3 with batch_size=4',
            'frcnn': 'mobilenet_v3_large with batch_size=2'
        },
        {
            'scenario': 'Moderate GPU (4-6GB VRAM)',
            'yolo': 'YOLOv8-s or YOLOv11-s with batch_size=16',
            'ssd': 'VGG16 with batch_size=8',
            'retinanet': 'ResNet18/34 with batch_size=4',
            'frcnn': 'resnet18 or resnet34 with batch_size=4'
        },
        {
            'scenario': 'Good GPU (6-8GB VRAM)',
            'yolo': 'YOLOv11-m with batch_size=16-32',
            'ssd': 'VGG16 with batch_size=16',
            'retinanet': 'ResNet34/50 with batch_size=4-8',
            'frcnn': 'resnet34 or resnet50 with batch_size=4-8'
        },
        {
            'scenario': 'High-end GPU (>8GB VRAM)',
            'yolo': 'YOLOv11-l or YOLOv11-x with batch_size=32',
            'ssd': 'VGG16 with batch_size=32',
            'retinanet': 'ResNet50 with batch_size=8-16',
            'frcnn': 'resnet50 with batch_size=8-16'
        }
    ]

    for rec in recommendations:
        print(f"\n{rec['scenario']}:")
        print(f"  • YOLO:         {rec['yolo']}")
        print(f"  • SSD:          {rec['ssd']}")
        print(f"  • RetinaNet:    {rec['retinanet']}")
        print(f"  • Faster R-CNN: {rec['frcnn']}")

    print("\n" + "="*70)


def print_usage_examples():
    """Print usage examples"""
    print("\n" + "="*70)
    print("Quick Start Examples")
    print("="*70)

    print("\n# Example 1: Lightweight YOLO (recommended for beginners)")
    print("from src.modern.yolo.trainer import YOLOTrainer")
    print("trainer = YOLOTrainer(model_version='v8', model_size='n', img_size=640)")
    print("trainer.train(data_yaml='data/data.yaml', epochs=100, batch_size=16)")

    print("\n# Example 2: Balanced YOLO")
    print("trainer = YOLOTrainer(model_version='v11', model_size='s', img_size=640)")
    print("trainer.train(data_yaml='data/data.yaml', epochs=100, batch_size=16)")

    print("\n# Example 3: Lightweight Faster R-CNN")
    print("from src.modern.faster_rcnn.trainer import FasterRCNNTrainer, TrafficSignDataset")
    print("trainer = FasterRCNNTrainer(num_classes=4, backbone='mobilenet_v3_large')")
    print("trainer.train(train_dataset, val_dataset, epochs=50, batch_size=4)")

    print("\n# Example 4: High Accuracy Faster R-CNN")
    print("trainer = FasterRCNNTrainer(num_classes=4, backbone='resnet50')")
    print("trainer.train(train_dataset, val_dataset, epochs=50, batch_size=4)")

    print("\n# Example 5: Extremely Lightweight SSDLite (best for mobile)")
    print("from src.modern.ssd.trainer import SSDTrainer, TrafficSignDataset")
    print("trainer = SSDTrainer(num_classes=4, backbone='mobilenet_v3')")
    print("trainer.train(train_dataset, val_dataset, epochs=50, batch_size=8)")

    print("\n# Example 6: Classic SSD with VGG16")
    print("trainer = SSDTrainer(num_classes=4, backbone='vgg16')")
    print("trainer.train(train_dataset, val_dataset, epochs=50, batch_size=8)")

    print("\n# Example 7: RetinaNet with Focal Loss (good for imbalanced data)")
    print("from src.modern.retinanet.trainer import RetinaNetTrainer, TrafficSignDataset")
    print("trainer = RetinaNetTrainer(num_classes=4, backbone='resnet34')")
    print("trainer.train(train_dataset, val_dataset, epochs=50, batch_size=4)")

    print("\n# Example 8: Lightweight RetinaNet with MobileNetV3")
    print("trainer = RetinaNetTrainer(num_classes=4, backbone='mobilenet_v3')")
    print("trainer.train(train_dataset, val_dataset, epochs=50, batch_size=4)")

    print("\n" + "="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Traffic Sign Detection - Model Comparison Tool")
    print("="*70)

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n✓ CUDA Available: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("\n✗ CUDA Not Available - CPU only")

    # Benchmark YOLO models
    yolo_results = benchmark_yolo_models()

    # Benchmark Faster R-CNN models
    frcnn_results = benchmark_faster_rcnn_models()

    # Benchmark SSD models
    ssd_results = benchmark_ssd_models()

    # Benchmark RetinaNet models
    retinanet_results = benchmark_retinanet_models()

    # Print recommendations
    print_recommendations()

    # Print usage examples
    print_usage_examples()

    print("\n" + "="*70)
    print("For detailed guide, see: docs/model_selection_guide.md")
    print("="*70 + "\n")
