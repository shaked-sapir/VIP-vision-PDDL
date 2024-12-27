from typing import List, Dict

import cv2

from src.fluent_classification.colors import to_int_rgb, find_exact_rgb_color_mask, NormalizedRGB
from src.object_detection.base_object_detector import ObjectDetector
from src.object_detection.bounded_object import BoundedObject


# TODO: consider making it a type in a separate file
class ObjectName(str):
    def __new__(cls, value):
        if ':' not in value:
            raise ValueError("ObjectName must include a colon ':'")
        return super().__new__(cls, value)


class ColorObjectDetector(ObjectDetector):
    def __init__(self, object_color_map: Dict[ObjectName, NormalizedRGB]):
        self._object_color_map = object_color_map

    @property
    def object_color_map(self):
        return self._object_color_map

    def detect(self, image: cv2.typing.MatLike, **kwargs) -> List[BoundedObject]:
        detected_objects = []

        for object_name, color_tuple in self._object_color_map.items():
            # Create a mask for the current color range
            full_rgb_tuple = to_int_rgb(color_tuple)
            mask = find_exact_rgb_color_mask(image, full_rgb_tuple)

            # Find contours for the masked region
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # Save detected object info
                detected_objects.append(
                    BoundedObject(
                        obj_type=object_name.split(':')[1],
                        name=object_name.split(':')[0],
                        x_anchor=x,
                        y_anchor=y,
                        width=w,
                        height=h
                    )
                )

        return detected_objects
