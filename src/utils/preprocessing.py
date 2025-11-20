"""
Preprocessing utilities for traffic sign detection
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from pathlib import Path
import albumentations as A


class ImagePreprocessor:
    """Image preprocessing for traffic sign detection"""

    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        """
        Initialize preprocessor

        Args:
            target_size: Target image size (width, height)
        """
        self.target_size = target_size

    def resize(self, image: np.ndarray, keep_aspect: bool = True) -> np.ndarray:
        """
        Resize image to target size

        Args:
            image: Input image
            keep_aspect: Whether to keep aspect ratio

        Returns:
            Resized image
        """
        if keep_aspect:
            return self._resize_keep_aspect(image)
        else:
            return cv2.resize(image, self.target_size)

    def _resize_keep_aspect(self, image: np.ndarray) -> np.ndarray:
        """Resize image keeping aspect ratio with padding"""
        h, w = image.shape[:2]
        target_w, target_h = self.target_size

        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize
        resized = cv2.resize(image, (new_w, new_h))

        # Create padded image
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)

        # Calculate padding
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2

        # Place resized image in center
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        return padded

    def normalize(self, image: np.ndarray, method: str = "standard") -> np.ndarray:
        """
        Normalize image

        Args:
            image: Input image
            method: Normalization method ('standard', 'minmax', 'imagenet')

        Returns:
            Normalized image
        """
        image = image.astype(np.float32)

        if method == "standard":
            # Standardization (mean=0, std=1)
            mean = np.mean(image, axis=(0, 1), keepdims=True)
            std = np.std(image, axis=(0, 1), keepdims=True)
            return (image - mean) / (std + 1e-7)

        elif method == "minmax":
            # Min-max normalization to [0, 1]
            return image / 255.0

        elif method == "imagenet":
            # ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406]) * 255
            std = np.array([0.229, 0.224, 0.225]) * 255
            return (image - mean) / std

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def enhance_contrast(self, image: np.ndarray, method: str = "clahe") -> np.ndarray:
        """
        Enhance image contrast

        Args:
            image: Input image
            method: Enhancement method ('clahe', 'hist_eq', 'adaptive')

        Returns:
            Enhanced image
        """
        if method == "clahe":
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        elif method == "hist_eq":
            # Histogram equalization
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

        elif method == "adaptive":
            # Adaptive histogram equalization
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            adaptive = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            return cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)

        else:
            raise ValueError(f"Unknown enhancement method: {method}")

    def denoise(self, image: np.ndarray, method: str = "bilateral") -> np.ndarray:
        """
        Denoise image

        Args:
            image: Input image
            method: Denoising method ('bilateral', 'gaussian', 'median', 'nlm')

        Returns:
            Denoised image
        """
        if method == "bilateral":
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == "gaussian":
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif method == "median":
            return cv2.medianBlur(image, 5)
        elif method == "nlm":
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            raise ValueError(f"Unknown denoising method: {method}")


class ColorSpaceConverter:
    """Convert images between color spaces"""

    @staticmethod
    def to_hsv(image: np.ndarray) -> np.ndarray:
        """Convert BGR to HSV"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    @staticmethod
    def to_gray(image: np.ndarray) -> np.ndarray:
        """Convert BGR to grayscale"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def to_lab(image: np.ndarray) -> np.ndarray:
        """Convert BGR to LAB"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    @staticmethod
    def to_ycrcb(image: np.ndarray) -> np.ndarray:
        """Convert BGR to YCrCb"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)


class DataAugmentation:
    """Data augmentation for training"""

    @staticmethod
    def get_train_transforms(img_size: int = 640) -> A.Compose:
        """Get training augmentation pipeline"""
        return A.Compose([
            A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.GaussianBlur(p=1.0),
                A.MotionBlur(p=1.0),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(p=1.0),
                A.GridDistortion(p=1.0),
            ], p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                p=0.3
            ),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))

    @staticmethod
    def get_val_transforms(img_size: int = 640) -> A.Compose:
        """Get validation augmentation pipeline"""
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))


def load_image(image_path: str) -> np.ndarray:
    """
    Load image from path

    Args:
        image_path: Path to image

    Returns:
        Loaded image in BGR format
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return image


def save_image(image: np.ndarray, output_path: str):
    """
    Save image to path

    Args:
        image: Image to save
        output_path: Output path
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


if __name__ == "__main__":
    # Example usage
    preprocessor = ImagePreprocessor(target_size=(640, 640))

    # Test with a dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Resize
    resized = preprocessor.resize(dummy_image)
    print(f"Original shape: {dummy_image.shape}")
    print(f"Resized shape: {resized.shape}")

    # Normalize
    normalized = preprocessor.normalize(resized, method="minmax")
    print(f"Normalized range: [{normalized.min():.2f}, {normalized.max():.2f}]")

    # Enhance contrast
    enhanced = preprocessor.enhance_contrast(dummy_image, method="clahe")
    print(f"Enhanced shape: {enhanced.shape}")
