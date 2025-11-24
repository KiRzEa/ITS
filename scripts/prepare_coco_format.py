"""
Convert YOLO format annotations to COCO format
Required for RetinaNet, SSD, and Faster R-CNN training
"""

import json
import yaml
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse


def yolo_to_coco_bbox(yolo_bbox, img_width, img_height):
    """
    Convert YOLO format bbox to COCO format

    Args:
        yolo_bbox: [x_center, y_center, width, height] in normalized coordinates
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        COCO format bbox: [x, y, width, height] in absolute coordinates
    """
    x_center, y_center, width, height = yolo_bbox

    # Convert to absolute coordinates
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    width_abs = width * img_width
    height_abs = height * img_height

    # Convert to top-left corner format
    x = x_center_abs - (width_abs / 2)
    y = y_center_abs - (height_abs / 2)

    return [x, y, width_abs, height_abs]


def convert_yolo_to_coco(
    images_dir: Path,
    labels_dir: Path,
    class_names: list,
    output_file: Path
):
    """
    Convert YOLO format dataset to COCO format

    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing YOLO format labels
        class_names: List of class names
        output_file: Output JSON file path
    """
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Create categories
    for i, class_name in enumerate(class_names):
        coco_format["categories"].append({
            "id": i + 1,  # COCO categories start from 1
            "name": class_name,
            "supercategory": "traffic_sign"
        })

    # Process images and annotations
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    annotation_id = 1

    print(f"Processing {len(image_files)} images...")

    for image_id, img_path in enumerate(tqdm(image_files), start=1):
        # Read image to get dimensions
        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"Warning: Could not read {img_path}: {e}")
            continue

        # Add image info
        coco_format["images"].append({
            "id": image_id,
            "file_name": img_path.name,
            "width": img_width,
            "height": img_height
        })

        # Process annotations
        label_path = labels_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            continue

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                yolo_bbox = [float(x) for x in parts[1:5]]

                # Convert bbox
                coco_bbox = yolo_to_coco_bbox(yolo_bbox, img_width, img_height)

                # Calculate area
                area = coco_bbox[2] * coco_bbox[3]

                # Add annotation
                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,  # COCO categories start from 1
                    "bbox": coco_bbox,
                    "area": area,
                    "iscrowd": 0
                })

                annotation_id += 1

    # Save to JSON
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(coco_format, f, indent=2)

    print(f"✓ Saved COCO format annotations to: {output_file}")
    print(f"  Images: {len(coco_format['images'])}")
    print(f"  Annotations: {len(coco_format['annotations'])}")
    print(f"  Categories: {len(coco_format['categories'])}")


def main():
    parser = argparse.ArgumentParser(description='Convert YOLO format to COCO format')
    parser.add_argument('--data-root', type=str, default='data/raw/yolov8',
                       help='Root directory of YOLO format dataset')
    parser.add_argument('--output-dir', type=str, default='data/processed/coco_format',
                       help='Output directory for COCO format annotations')

    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    # Read data.yaml to get class names
    data_yaml = data_root / 'data.yaml'
    if not data_yaml.exists():
        print(f"Error: data.yaml not found at {data_yaml}")
        return

    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)

    class_names = data_config['names']
    print(f"Found {len(class_names)} classes: {class_names}")

    # Convert train set
    print(f"\n{'='*60}")
    print("Converting training set...")
    print(f"{'='*60}")
    convert_yolo_to_coco(
        images_dir=data_root / 'train' / 'images',
        labels_dir=data_root / 'train' / 'labels',
        class_names=class_names,
        output_file=output_dir / 'train_coco.json'
    )

    # Convert validation set
    print(f"\n{'='*60}")
    print("Converting validation set...")
    print(f"{'='*60}")
    convert_yolo_to_coco(
        images_dir=data_root / 'valid' / 'images',
        labels_dir=data_root / 'valid' / 'labels',
        class_names=class_names,
        output_file=output_dir / 'valid_coco.json'
    )

    # Convert test set if exists
    if (data_root / 'test' / 'images').exists():
        print(f"\n{'='*60}")
        print("Converting test set...")
        print(f"{'='*60}")
        convert_yolo_to_coco(
            images_dir=data_root / 'test' / 'images',
            labels_dir=data_root / 'test' / 'labels',
            class_names=class_names,
            output_file=output_dir / 'test_coco.json'
        )

    print(f"\n{'='*60}")
    print("Conversion complete!")
    print(f"{'='*60}")
    print(f"COCO format annotations saved to: {output_dir}")


if __name__ == "__main__":
    main()
