"""
Deterministic object detector for Hanoi domain.

This detector uses size and position information to identify discs and pegs.
Since all discs in the default Hanoi rendering are the same color (red),
we use contour detection and size analysis to distinguish between them.
"""

from typing import Dict, List, Tuple
import cv2
import numpy as np

from src.object_detection.base_object_detector import ObjectDetector
from src.typings import ObjectLabel
from src.utils.visualize import NormalizedRGB


class HanoiObjectDetector(ObjectDetector):
    """
    Deterministic object detector for the Hanoi domain.

    Detects pegs (gray vertical rectangles) and discs (red horizontal rectangles)
    and identifies them based on position and size.
    """

    def __init__(self, object_name_to_color: Dict[ObjectLabel, NormalizedRGB] = None):
        """
        Initialize the Hanoi object detector.

        :param object_name_to_color: Optional mapping from object labels to colors.
                                      If None, uses default Hanoi colors.
        """
        if object_name_to_color is None:
            # Default colors for Hanoi domain (based on pddlgym rendering)
            object_name_to_color = {
                ObjectLabel("peg1:peg"): (0.5, 0.5, 0.5),  # Gray
                ObjectLabel("peg2:peg"): (0.5, 0.5, 0.5),  # Gray
                ObjectLabel("peg3:peg"): (0.5, 0.5, 0.5),  # Gray
                ObjectLabel("d1:disc"): (0.8, 0.1, 0.1),   # Red (all discs are red in default rendering)
                ObjectLabel("d2:disc"): (0.8, 0.1, 0.1),   # Red
                ObjectLabel("d3:disc"): (0.8, 0.1, 0.1),   # Red
                ObjectLabel("d4:disc"): (0.8, 0.1, 0.1),   # Red
                ObjectLabel("d5:disc"): (0.8, 0.1, 0.1),   # Red
                ObjectLabel("d6:disc"): (0.8, 0.1, 0.1),   # Red
            }

        super().__init__(object_color_map=object_name_to_color)

        # Threshold values for color matching (more lenient for similar red discs)
        self.color_threshold = 0.15  # Allow more variation in color matching
        self.min_area_threshold = 50  # Minimum area for a detected object

    def detect_objects(self, image_path: str) -> Dict[ObjectLabel, Tuple[int, int]]:
        """
        Detect objects in the Hanoi image.

        Since all discs are the same color, we detect them by:
        1. Finding all red rectangles (discs)
        2. Sorting them by size (largest = d3, medium = d2, smallest = d1)
        3. Finding gray rectangles (pegs)
        4. Sorting pegs by horizontal position (leftmost = peg1, etc.)

        :param image_path: Path to the image file
        :return: Dictionary mapping object labels to their (x, y) positions
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Convert to RGB (OpenCV loads as BGR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize to 0-1 range
        image_normalized = image_rgb.astype(float) / 255.0

        # Detect red regions (discs)
        discs = self._detect_discs(image_normalized)

        # Detect gray regions (pegs)
        pegs = self._detect_pegs(image_normalized)

        # Combine results
        detected_objects = {}
        detected_objects.update(discs)
        detected_objects.update(pegs)

        return detected_objects

    def _detect_discs(self, image: np.ndarray) -> Dict[ObjectLabel, Tuple[int, int]]:
        """
        Detect discs in the image by finding red horizontal rectangles.

        :param image: Normalized RGB image (0-1 range)
        :return: Dictionary mapping disc labels to positions
        """
        # Define red color range (in normalized RGB)
        red_target = np.array([0.8, 0.1, 0.1])

        # Find pixels close to red color
        color_diff = np.linalg.norm(image - red_target, axis=2)
        red_mask = (color_diff < self.color_threshold).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter and sort contours by area (size)
        disc_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area_threshold:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                # Check if it's more horizontal than vertical (disc shape)
                if w > h:
                    disc_contours.append({
                        'contour': contour,
                        'area': area,
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h,
                        'center': (x + w // 2, y + h // 2)
                    })

        # Sort by area (largest to smallest)
        disc_contours.sort(key=lambda d: d['area'], reverse=True)

        # Assign disc IDs based on size
        detected_discs = {}
        for i, disc_info in enumerate(disc_contours):
            disc_num = len(disc_contours) - i  # Largest = highest number (d3), smallest = d1
            disc_label = ObjectLabel(f"d{disc_num}:disc")
            detected_discs[disc_label] = disc_info['center']

        return detected_discs

    def _detect_pegs(self, image: np.ndarray) -> Dict[ObjectLabel, Tuple[int, int]]:
        """
        Detect pegs in the image by finding gray vertical rectangles.

        :param image: Normalized RGB image (0-1 range)
        :return: Dictionary mapping peg labels to positions
        """
        # Define gray color range
        gray_target = np.array([0.5, 0.5, 0.5])

        # Find pixels close to gray color
        color_diff = np.linalg.norm(image - gray_target, axis=2)
        gray_mask = (color_diff < self.color_threshold).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter and sort contours
        peg_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area_threshold:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                # Check if it's more vertical than horizontal (peg shape)
                if h > w:
                    peg_contours.append({
                        'contour': contour,
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h,
                        'center': (x + w // 2, y + h // 2)
                    })

        # Sort by horizontal position (left to right)
        peg_contours.sort(key=lambda p: p['x'])

        # Assign peg IDs based on position
        detected_pegs = {}
        for i, peg_info in enumerate(peg_contours):
            peg_num = i + 1  # peg1 = leftmost
            peg_label = ObjectLabel(f"peg{peg_num}:peg")
            detected_pegs[peg_label] = peg_info['center']

        return detected_pegs
