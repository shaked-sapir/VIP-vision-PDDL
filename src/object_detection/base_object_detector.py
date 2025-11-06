from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

import cv2

from src.object_detection.bounded_object import BoundedObject


# TODO Later: consider making this an interface instead of a class
class ObjectDetector(ABC):
    @abstractmethod
    def detect(self, image: Union[cv2.typing.MatLike, Path, str], *args, **kwargs):
        pass
