from abc import ABC
from typing import List, Dict

from src.object_detection.base_object_detector import ObjectDetector
from src.object_detection.bounded_object import BoundedObject
from src.utils.containers import group_objects_by_key


# TODO Later: make this an actual Base class, with the option to be initialized with proper functions for the predicates - this includes both the constructor and the `classify` nethod
class BaseFluentClassifier(ABC):
    def __init__(self, object_detector: ObjectDetector):
        self.object_detector = object_detector

    def classify(self, image) -> Dict[str, bool]:
        """
         This is the main method of the class: given an image, return for all possible grounded predicates in
         of the problem whether they hold in in the image or not.
         :param image:
         :return: a dict the form of {predicate_name: True/False}
        """
        raise NotImplementedError
