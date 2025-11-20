"""
Color + Shape Based Traffic Sign Detector
Traditional computer vision approach using color segmentation and geometric shape detection
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from enum import Enum


class SignColor(Enum):
    """Traffic sign colors"""
    RED = "red"
    BLUE = "blue"
    YELLOW = "yellow"
    GREEN = "green"


class SignShape(Enum):
    """Traffic sign shapes"""
    CIRCLE = "circle"
    TRIANGLE = "triangle"
    RECTANGLE = "rectangle"
    OCTAGON = "octagon"  # Stop sign
    DIAMOND = "diamond"


class ColorShapeDetector:
    """Traffic sign detector using color and shape"""

    def __init__(self):
        """Initialize color-shape detector"""
        # HSV color ranges for different sign colors
        self.color_ranges = {
            SignColor.RED: [
                # Red wraps around in HSV
                (np.array([0, 100, 100]), np.array([10, 255, 255])),
                (np.array([160, 100, 100]), np.array([180, 255, 255]))
            ],
            SignColor.BLUE: [
                (np.array([100, 100, 100]), np.array([130, 255, 255]))
            ],
            SignColor.YELLOW: [
                (np.array([20, 100, 100]), np.array([30, 255, 255]))
            ],
            SignColor.GREEN: [
                (np.array([40, 50, 50]), np.array([80, 255, 255]))
            ]
        }

    def detect_color(
        self,
        image: np.ndarray,
        color: SignColor,
        min_area: int = 500
    ) -> List[np.ndarray]:
        """
        Detect regions of specific color

        Args:
            image: Input image (BGR)
            color: Target color
            min_area: Minimum contour area

        Returns:
            List of contours
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create mask for color
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for lower, upper in self.color_ranges[color]:
            color_mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.bitwise_or(mask, color_mask)

        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter by area
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

        return valid_contours, mask

    def detect_shape(
        self,
        contour: np.ndarray,
        epsilon_factor: float = 0.04
    ) -> Tuple[SignShape, float]:
        """
        Detect shape of contour

        Args:
            contour: Input contour
            epsilon_factor: Approximation accuracy factor

        Returns:
            (detected_shape, confidence)
        """
        # Approximate contour
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Get number of vertices
        vertices = len(approx)

        # Calculate circularity
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return SignShape.CIRCLE, 0.0

        circularity = 4 * np.pi * area / (perimeter ** 2)

        # Determine shape based on vertices and circularity
        if vertices == 3:
            return SignShape.TRIANGLE, 0.9

        elif vertices == 4:
            # Check if rectangle or diamond
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h

            # Check orientation
            rect = cv2.minAreaRect(contour)
            angle = rect[2]

            if 0.85 <= aspect_ratio <= 1.15:  # Square-ish
                if 40 <= abs(angle) <= 50:  # Rotated ~45 degrees
                    return SignShape.DIAMOND, 0.85
                else:
                    return SignShape.RECTANGLE, 0.85
            else:
                return SignShape.RECTANGLE, 0.8

        elif vertices >= 6 and vertices <= 10:
            if circularity > 0.75:
                return SignShape.CIRCLE, circularity
            elif vertices == 8:
                return SignShape.OCTAGON, 0.8
            else:
                return SignShape.CIRCLE, circularity * 0.8

        elif circularity > 0.8:
            return SignShape.CIRCLE, circularity

        else:
            return SignShape.CIRCLE, 0.5  # Default to circle with low confidence

    def classify_sign(
        self,
        color: SignColor,
        shape: SignShape
    ) -> Tuple[str, float]:
        """
        Classify traffic sign based on color and shape

        Args:
            color: Detected color
            shape: Detected shape

        Returns:
            (sign_type, confidence)
        """
        # Simple rule-based classification
        rules = {
            (SignColor.RED, SignShape.OCTAGON): ("Stop", 0.95),
            (SignColor.RED, SignShape.TRIANGLE): ("Yield", 0.9),
            (SignColor.RED, SignShape.CIRCLE): ("Prohibition", 0.85),
            (SignColor.BLUE, SignShape.CIRCLE): ("Mandatory", 0.85),
            (SignColor.BLUE, SignShape.RECTANGLE): ("Information", 0.8),
            (SignColor.YELLOW, SignShape.DIAMOND): ("Warning", 0.9),
            (SignColor.YELLOW, SignShape.TRIANGLE): ("Warning", 0.85),
            (SignColor.GREEN, SignShape.RECTANGLE): ("Guide", 0.8),
        }

        key = (color, shape)
        if key in rules:
            return rules[key]
        else:
            return ("Unknown", 0.5)

    def detect(
        self,
        image: np.ndarray,
        colors: Optional[List[SignColor]] = None,
        min_area: int = 500,
        confidence_threshold: float = 0.7
    ) -> Tuple[List[Tuple[int, int, int, int]], List[str], List[float]]:
        """
        Detect traffic signs in image

        Args:
            image: Input image (BGR)
            colors: List of colors to detect (None = all colors)
            min_area: Minimum contour area
            confidence_threshold: Minimum confidence threshold

        Returns:
            (boxes, sign_types, confidences)
        """
        if colors is None:
            colors = list(SignColor)

        all_detections = []

        for color in colors:
            # Detect color regions
            contours, mask = self.detect_color(image, color, min_area)

            for contour in contours:
                # Detect shape
                shape, shape_conf = self.detect_shape(contour)

                # Classify sign
                sign_type, class_conf = self.classify_sign(color, shape)

                # Overall confidence
                confidence = shape_conf * class_conf

                if confidence >= confidence_threshold:
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)

                    all_detections.append({
                        'box': (x, y, x + w, y + h),
                        'type': sign_type,
                        'confidence': confidence,
                        'color': color.value,
                        'shape': shape.value
                    })

        # Sort by confidence
        all_detections = sorted(all_detections, key=lambda x: x['confidence'], reverse=True)

        # Extract results
        boxes = [d['box'] for d in all_detections]
        types = [d['type'] for d in all_detections]
        confidences = [d['confidence'] for d in all_detections]

        return boxes, types, confidences

    def visualize_detection(
        self,
        image: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
        types: List[str],
        confidences: List[float]
    ) -> np.ndarray:
        """
        Visualize detections on image

        Args:
            image: Input image
            boxes: List of bounding boxes
            types: List of sign types
            confidences: List of confidences

        Returns:
            Image with visualizations
        """
        result = image.copy()

        for box, sign_type, conf in zip(boxes, types, confidences):
            x1, y1, x2, y2 = box

            # Draw box
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{sign_type}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_w, label_h = label_size

            cv2.rectangle(
                result,
                (x1, y1 - label_h - 10),
                (x1 + label_w, y1),
                (0, 255, 0),
                -1
            )

            cv2.putText(
                result,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2
            )

        return result


class CircleDetector:
    """Specialized circular sign detector using Hough Circle Transform"""

    def __init__(
        self,
        dp: float = 1.2,
        min_dist: int = 50,
        param1: int = 50,
        param2: int = 30,
        min_radius: int = 10,
        max_radius: int = 100
    ):
        """
        Initialize circle detector

        Args:
            dp: Inverse ratio of accumulator resolution
            min_dist: Minimum distance between circle centers
            param1: Higher threshold for Canny edge detector
            param2: Accumulator threshold
            min_radius: Minimum circle radius
            max_radius: Maximum circle radius
        """
        self.dp = dp
        self.min_dist = min_dist
        self.param1 = param1
        self.param2 = param2
        self.min_radius = min_radius
        self.max_radius = max_radius

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Detect circles in image

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            List of circles (x, y, radius)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Detect circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            return [(x, y, r) for x, y, r in circles]
        else:
            return []


if __name__ == "__main__":
    # Example usage
    print("Color-Shape Detector Example")

    # Create detector
    detector = ColorShapeDetector()

    # Create dummy image with red circle
    image = np.ones((480, 640, 3), dtype=np.uint8) * 100

    # Draw a red circle
    cv2.circle(image, (320, 240), 50, (0, 0, 255), -1)

    # Detect
    boxes, types, confidences = detector.detect(image, confidence_threshold=0.5)

    print(f"Found {len(boxes)} detections:")
    for box, sign_type, conf in zip(boxes, types, confidences):
        print(f"  {sign_type}: {conf:.3f} at {box}")

    # Visualize
    result = detector.visualize_detection(image, boxes, types, confidences)

    print("Detection completed!")
