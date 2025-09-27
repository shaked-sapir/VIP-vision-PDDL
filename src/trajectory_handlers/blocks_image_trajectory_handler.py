from typing import Dict

from src.fluent_classification.blocks_contour_fluent_classifier import BlocksContourFluentClassifier
from src.object_detection import ColorObjectDetector
from src.trajectory_handlers import ImageTrajectoryHandler
from src.types import ObjectLabel
from src.utils.visualize import NormalizedRGB
from pddlgym.rendering.blocks import _block_name_to_color


class BlocksImageTrajectoryHandler(ImageTrajectoryHandler):
    def _init_visual_components(self) -> None:
        """
        In this class, this method should only be called after initializing a specific
        blocks problem, because the object detection module depends on blocks colors which
        are determined only at problem initialization time by the `_block_name_to_color`
        method from gym.
        """

        # extracting colors of objects from trajectory so we can detect the objects in the image
        object_name_to_color: Dict[ObjectLabel, NormalizedRGB] = {
            **{ObjectLabel(str(obj)): color for obj, color in _block_name_to_color.items()},
            ObjectLabel("robot:robot"): (0.4, 0.4, 0.4), # robot color is const
            ObjectLabel("table:table"): (0.5, 0.2, 0.0)  # table color is const
        }

        self.object_detector = ColorObjectDetector(object_name_to_color)
        self.fluent_classifier = BlocksContourFluentClassifier(self.object_detector)

        print(f"Object name to color map: {self.object_detector.object_color_map}")
