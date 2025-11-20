"""
HOG + SVM Traffic Sign Detector
Traditional computer vision approach using Histogram of Oriented Gradients and Support Vector Machine
"""

import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from typing import List, Tuple, Optional, Dict
import pickle
from pathlib import Path
import joblib


class HOGSVMDetector:
    """Traffic sign detector using HOG features and SVM classifier"""

    def __init__(
        self,
        orientations: int = 9,
        pixels_per_cell: Tuple[int, int] = (8, 8),
        cells_per_block: Tuple[int, int] = (2, 2),
        block_norm: str = 'L2-Hys',
        img_size: Tuple[int, int] = (64, 64),
        svm_kernel: str = 'rbf',
        svm_c: float = 10.0
    ):
        """
        Initialize HOG+SVM detector

        Args:
            orientations: Number of orientation bins
            pixels_per_cell: Size of a cell
            cells_per_block: Number of cells in each block
            block_norm: Block normalization method
            img_size: Target image size for feature extraction
            svm_kernel: SVM kernel type
            svm_c: SVM regularization parameter
        """
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
        self.img_size = img_size
        self.svm_kernel = svm_kernel
        self.svm_c = svm_c

        # Initialize models
        self.svm = SVC(
            kernel=svm_kernel,
            C=svm_c,
            probability=True,
            random_state=42
        )
        self.scaler = StandardScaler()

        self.is_trained = False
        self.class_names = None

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract HOG features from image

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            HOG feature vector
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Resize to standard size
        resized = cv2.resize(gray, self.img_size)

        # Extract HOG features
        features = hog(
            resized,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm=self.block_norm,
            visualize=False,
            feature_vector=True
        )

        return features

    def extract_features_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extract HOG features from multiple images

        Args:
            images: List of images

        Returns:
            Feature matrix [n_images, n_features]
        """
        features = []
        for img in images:
            feat = self.extract_features(img)
            features.append(feat)

        return np.array(features)

    def train(
        self,
        train_images: List[np.ndarray],
        train_labels: np.ndarray,
        class_names: Optional[List[str]] = None
    ):
        """
        Train the SVM classifier

        Args:
            train_images: List of training images
            train_labels: Training labels
            class_names: List of class names (optional)
        """
        print("Extracting HOG features from training images...")
        features = self.extract_features_batch(train_images)

        print("Training SVM classifier...")
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)

        # Train SVM
        self.svm.fit(features_scaled, train_labels)

        self.is_trained = True
        self.class_names = class_names

        print(f"Training completed. Model accuracy: {self.svm.score(features_scaled, train_labels):.3f}")

    def predict(self, image: np.ndarray) -> Tuple[int, float]:
        """
        Predict class for single image

        Args:
            image: Input image

        Returns:
            (predicted_class, confidence)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        features = self.extract_features(image).reshape(1, -1)
        features_scaled = self.scaler.transform(features)

        pred_class = self.svm.predict(features_scaled)[0]
        pred_proba = self.svm.predict_proba(features_scaled)[0]
        confidence = pred_proba[pred_class]

        return int(pred_class), float(confidence)

    def predict_batch(self, images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict classes for multiple images

        Args:
            images: List of images

        Returns:
            (predicted_classes, confidences)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        features = self.extract_features_batch(images)
        features_scaled = self.scaler.transform(features)

        pred_classes = self.svm.predict(features_scaled)
        pred_probas = self.svm.predict_proba(features_scaled)
        confidences = np.max(pred_probas, axis=1)

        return pred_classes, confidences

    def save(self, save_path: str):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("Model not trained. Nothing to save.")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'svm': self.svm,
            'scaler': self.scaler,
            'config': {
                'orientations': self.orientations,
                'pixels_per_cell': self.pixels_per_cell,
                'cells_per_block': self.cells_per_block,
                'block_norm': self.block_norm,
                'img_size': self.img_size,
                'svm_kernel': self.svm_kernel,
                'svm_c': self.svm_c
            },
            'class_names': self.class_names
        }

        joblib.dump(model_data, save_path)
        print(f"Model saved to {save_path}")

    def load(self, load_path: str):
        """Load trained model"""
        model_data = joblib.load(load_path)

        self.svm = model_data['svm']
        self.scaler = model_data['scaler']
        self.class_names = model_data['class_names']

        config = model_data['config']
        self.orientations = config['orientations']
        self.pixels_per_cell = config['pixels_per_cell']
        self.cells_per_block = config['cells_per_block']
        self.block_norm = config['block_norm']
        self.img_size = config['img_size']
        self.svm_kernel = config['svm_kernel']
        self.svm_c = config['svm_c']

        self.is_trained = True
        print(f"Model loaded from {load_path}")

    def visualize_hog(self, image: np.ndarray) -> np.ndarray:
        """
        Visualize HOG features

        Args:
            image: Input image

        Returns:
            HOG visualization image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Resize
        resized = cv2.resize(gray, self.img_size)

        # Extract HOG with visualization
        _, hog_image = hog(
            resized,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm=self.block_norm,
            visualize=True,
            feature_vector=True
        )

        # Normalize for display
        hog_image = (hog_image * 255).astype(np.uint8)

        return hog_image


class SlidingWindowDetector:
    """Sliding window detection for localization"""

    def __init__(
        self,
        classifier: HOGSVMDetector,
        window_size: Tuple[int, int] = (64, 64),
        step_size: int = 16,
        scale_factor: float = 1.2,
        min_scale: float = 0.5,
        max_scale: float = 2.0
    ):
        """
        Initialize sliding window detector

        Args:
            classifier: Trained HOG+SVM classifier
            window_size: Size of sliding window
            step_size: Step size for sliding window
            scale_factor: Scale factor for image pyramid
            min_scale: Minimum scale
            max_scale: Maximum scale
        """
        self.classifier = classifier
        self.window_size = window_size
        self.step_size = step_size
        self.scale_factor = scale_factor
        self.min_scale = min_scale
        self.max_scale = max_scale

    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.7
    ) -> Tuple[List[Tuple[int, int, int, int]], List[int], List[float]]:
        """
        Detect traffic signs in image using sliding window

        Args:
            image: Input image
            confidence_threshold: Minimum confidence threshold

        Returns:
            (boxes, classes, confidences)
        """
        h, w = image.shape[:2]
        detections = []

        # Create image pyramid
        scale = self.min_scale
        while scale <= self.max_scale:
            # Resize image
            scaled_h = int(h * scale)
            scaled_w = int(w * scale)
            scaled_image = cv2.resize(image, (scaled_w, scaled_h))

            # Sliding window
            for y in range(0, scaled_h - self.window_size[1], self.step_size):
                for x in range(0, scaled_w - self.window_size[0], self.step_size):
                    # Extract window
                    window = scaled_image[y:y + self.window_size[1], x:x + self.window_size[0]]

                    # Classify
                    pred_class, confidence = self.classifier.predict(window)

                    if confidence >= confidence_threshold:
                        # Convert coordinates back to original scale
                        x1 = int(x / scale)
                        y1 = int(y / scale)
                        x2 = int((x + self.window_size[0]) / scale)
                        y2 = int((y + self.window_size[1]) / scale)

                        detections.append({
                            'box': (x1, y1, x2, y2),
                            'class': pred_class,
                            'confidence': confidence
                        })

            scale *= self.scale_factor

        # Apply non-maximum suppression
        if detections:
            boxes, classes, confidences = self._non_max_suppression(detections)
        else:
            boxes, classes, confidences = [], [], []

        return boxes, classes, confidences

    def _non_max_suppression(
        self,
        detections: List[Dict],
        iou_threshold: float = 0.3
    ) -> Tuple[List[Tuple[int, int, int, int]], List[int], List[float]]:
        """Apply non-maximum suppression to detections"""
        if not detections:
            return [], [], []

        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)

            # Remove overlapping detections
            detections = [
                det for det in detections
                if self._calculate_iou(best['box'], det['box']) < iou_threshold
            ]

        boxes = [d['box'] for d in keep]
        classes = [d['class'] for d in keep]
        confidences = [d['confidence'] for d in keep]

        return boxes, classes, confidences

    @staticmethod
    def _calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - intersection

        return intersection / (union + 1e-6)


if __name__ == "__main__":
    # Example usage
    print("HOG+SVM Detector Example")

    # Create detector
    detector = HOGSVMDetector(img_size=(64, 64))

    # Create dummy training data
    n_samples = 100
    n_classes = 3
    train_images = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(n_samples)]
    train_labels = np.random.randint(0, n_classes, n_samples)

    # Train
    detector.train(train_images, train_labels, class_names=['Stop', 'Yield', 'Speed Limit'])

    # Predict
    test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    pred_class, confidence = detector.predict(test_image)
    print(f"Predicted class: {pred_class}, Confidence: {confidence:.3f}")
