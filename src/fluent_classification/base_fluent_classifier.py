from abc import ABC
from enum import Enum
from pathlib import Path
from typing import List, Dict, Union

from src.object_detection.base_object_detector import ObjectDetector
from src.object_detection.bounded_object import BoundedObject
from src.utils.containers import group_objects_by_key


class PredicateTruthValue(str, Enum):
    TRUE = "true"
    FALSE = "false"
    UNCERTAIN = "uncertain"


class FluentClassifier(ABC):

    # TODO: make this return only str -> float, where the deterministic fluent classifier should always return 1/0 as fluents probabilites
    def classify(self, image_path: Path | str) -> Dict[str, PredicateTruthValue]:
        """
         This is the main method of the class: given an image, return for all possible grounded predicates in
         of the problem whether they hold in the image or not.
         :param image_path:
         :return: a dict the form of {predicate_name: True/False} or {predicate_name: probability}
        """
        raise NotImplementedError
