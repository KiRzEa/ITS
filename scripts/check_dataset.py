"""
Dataset Diagnostic Script
Check if dataset exists and show directory structure
"""

import os
from pathlib import Path

def check_directory(path):
    """Check if directory exists and list contents"""
    path = Path(path)
    print(f"\nChecking: {path}")
    print(f"  Absolute path: {path.absolute()}")
    print(f"  Exists: {path.exists()}")

    if path.exists():
        print(f"  Is directory: {path.is_dir()}")
        if path.is_dir():
            contents = list(path.iterdir())
            print(f"  Contents ({len(contents)} items):")
            for item in sorted(contents)[:20]:  # Show first 20 items
                item_type = "DIR" if item.is_dir() else "FILE"
                size = f"{item.stat().st_size:,} bytes" if item.is_file() else ""
                print(f"    [{item_type}] {item.name} {size}")
            if len(contents) > 20:
                print(f"    ... and {len(contents) - 20} more items")
    else:
        print(f"  ⚠️  Path does not exist!")
    print("-" * 80)

# Check current working directory
print("=" * 80)
print("DATASET DIAGNOSTIC")
print("=" * 80)
print(f"\nCurrent working directory: {os.getcwd()}")

# Check expected data locations
locations_to_check = [
    "data",
    "data/raw",
    "data/raw/yolov8",
    "data/raw/yolov8/train",
    "data/raw/yolov8/train/images",
    "data/raw/yolov8/train/labels",
    "data/raw/yolov8/valid",
    "data/raw/yolov8/valid/images",
    "data/raw/yolov8/valid/labels",
    "data/raw/yolov8/test",
    "data/raw/yolov8/test/images",
    "data/raw/yolov8/test/labels",
    "data/raw",
]

for location in locations_to_check:
    check_directory(location)

# Check for any yolov8 directories in data/raw
print("\n" + "=" * 80)
print("SEARCHING FOR YOLOV8 DIRECTORIES")
print("=" * 80)

data_raw = Path("data/raw")
if data_raw.exists():
    for item in data_raw.rglob("*"):
        if item.is_dir():
            print(f"Found directory: {item}")

# Check for data.yaml files
print("\n" + "=" * 80)
print("SEARCHING FOR data.yaml FILES")
print("=" * 80)

for yaml_file in Path(".").rglob("data.yaml"):
    print(f"Found: {yaml_file.absolute()}")
    try:
        with open(yaml_file, 'r') as f:
            print(f"  Content preview (first 10 lines):")
            for i, line in enumerate(f):
                if i >= 10:
                    break
                print(f"    {line.rstrip()}")
    except Exception as e:
        print(f"  Error reading file: {e}")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
