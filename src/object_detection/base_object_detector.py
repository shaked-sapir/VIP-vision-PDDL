from abc import ABC, abstractmethod
from typing import List

import cv2

from src.object_detection.bounded_object import BoundedObject


# TODO Later: consider making this an interface instead of a class
class ObjectDetector(ABC):
    @abstractmethod
    def detect(self, image: cv2.typing.MatLike, *args, **kwargs) -> List[BoundedObject]:
        pass
