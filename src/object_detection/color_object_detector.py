from typing import List, Dict

import cv2
import numpy as np

from src.types import ObjectLabel
from src.utils.visualize import to_int_rgb, find_exact_rgb_color_mask, NormalizedRGB
from src.object_detection.base_object_detector import ObjectDetector
from src.object_detection.bounded_object import BoundedObject


class ColorObjectDetector(ObjectDetector):
    def __init__(self, object_color_map: Dict[ObjectLabel, NormalizedRGB]):
        self._object_color_map = object_color_map

    @property
    def object_color_map(self):
        return self._object_color_map

    def merge_close_contours(self, contours):
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        # Sort bounding boxes by their x-coordinate
        bounding_boxes.sort(key=lambda box: box[0])
        most_left_box, most_right_box = bounding_boxes[0], bounding_boxes[-1]
        # Merge the most left and most right bounding boxes
        x1 = most_left_box[0]
        y1 = most_left_box[1]
        width = most_right_box[0] - most_left_box[0] + most_right_box[2]
        height = most_left_box[3]  # Assuming all boxes have the same height
        merged_box = (x1, y1, width, height)
        # return the box as a contour
        return [np.array([[x1, y1], [x1 + width, y1], [x1 + width, y1 + height], [x1, y1 + height]])]



    def detect(self, image: cv2.typing.MatLike, **kwargs) -> List[BoundedObject]:
        detected_objects = []

        for object_label, color_tuple in self._object_color_map.items():
            # Create a mask for the current color range
            full_rgb_tuple = to_int_rgb(color_tuple)
            mask = find_exact_rgb_color_mask(image, full_rgb_tuple)

            # Find contours for the masked region
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                contours = self.merge_close_contours(contours)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # Save detected object info
                detected_objects.append(
                    BoundedObject(
                        obj_type=object_label.type,
                        name=object_label.name,
                        x_anchor=x,
                        y_anchor=y,
                        width=w,
                        height=h
                    )
                )

        return detected_objects
