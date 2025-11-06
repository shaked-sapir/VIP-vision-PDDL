"""
Deterministic trajectory handler for Hanoi domain.

Uses color-based object detection and geometric fluent classification.
"""

from typing import Dict

from src.fluent_classification.hanoi_fluent_classifier import HanoiFluentClassifier
from src.object_detection.hanoi_object_detector import HanoiObjectDetector
from src.trajectory_handlers import ImageTrajectoryHandler
from src.typings import ObjectLabel
from src.utils.visualize import NormalizedRGB


class HanoiImageTrajectoryHandler(ImageTrajectoryHandler):
    """
    Deterministic trajectory handler for the Hanoi domain.
    Uses HanoiObjectDetector (position/size-based) and HanoiFluentClassifier (geometric).
    """

    def init_visual_components(self, init_state_image_path=None) -> None:
        """
        Initialize the object detector and fluent classifier for Hanoi domain.

        For Hanoi, we use:
        - HanoiObjectDetector: Detects pegs (gray) and discs (red) based on size and position
        - HanoiFluentClassifier: Classifies predicates based on geometric relationships

        :param init_state_image_path: Optional path to initial state image (not used for Hanoi)
        """
        # Define default colors for Hanoi objects
        # In the default rendering: pegs are gray (0.5, 0.5, 0.5), discs are red (0.8, 0.1, 0.1)
        object_name_to_color: Dict[ObjectLabel, NormalizedRGB] = {
            # Pegs (gray)
            ObjectLabel("peg1:peg"): (0.5, 0.5, 0.5),
            ObjectLabel("peg2:peg"): (0.5, 0.5, 0.5),
            ObjectLabel("peg3:peg"): (0.5, 0.5, 0.5),
            # Discs (all red in default rendering)
            ObjectLabel("d1:disc"): (0.8, 0.1, 0.1),
            ObjectLabel("d2:disc"): (0.8, 0.1, 0.1),
            ObjectLabel("d3:disc"): (0.8, 0.1, 0.1),
            ObjectLabel("d4:disc"): (0.8, 0.1, 0.1),
            ObjectLabel("d5:disc"): (0.8, 0.1, 0.1),
            ObjectLabel("d6:disc"): (0.8, 0.1, 0.1),
        }

        self.object_detector = HanoiObjectDetector(object_name_to_color)
        self.fluent_classifier = HanoiFluentClassifier(self.object_detector)

        print(f"Hanoi deterministic visual components initialized")
        print(f"Object detector: HanoiObjectDetector (position/size-based)")
        print(f"Fluent classifier: HanoiFluentClassifier (geometric)")
