import itertools
import json
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np

from src.fluent_classification.base_fluent_classifier import FluentClassifier, PredicateTruthValue
from src.object_detection import ColorObjectDetector
from src.object_detection.bounded_object import BoundedObject
from src.typings import ObjectLabel
from src.utils.visualize import draw_objects, load_image


class BlocksContourFluentClassifier(FluentClassifier):

    def __init__(self, object_detector: ColorObjectDetector):
        """
        Initializes the BlocksFluentClassifier with a given object detector.
        :param object_detector: An instance of ColorObjectDetector that detects blocksworld, robots, and tables.
        """
        self.object_detector = object_detector

    @staticmethod
    def _bool_to_predicate_state(value: bool) -> PredicateTruthValue:
        """
        Converts a boolean value to a PredicateState.
        :param value: A boolean value.
        :return: PredicateState.TRUE if value is True, otherwise PredicateState.FALSE.
        """
        return PredicateTruthValue.TRUE if value else PredicateTruthValue.FALSE

    def classify(self, image_path: Path | str) -> Dict[str, PredicateTruthValue]:
        image = load_image(image_path)
        detected_objects = self.object_detector.detect(image)
        predicates = {}

        for obj in detected_objects:
            print(f"Detected a {obj.label} at {obj.bbox.box}")
        blocks = [obj for obj in detected_objects if obj.type == "block"]
        robot = [obj for obj in detected_objects if obj.type == "robot"][0]
        table = [obj for obj in detected_objects if obj.type == "table"][0]

        for block1, block2 in itertools.permutations(blocks, 2):
            predicates[f"on({block1.label},{block2.label})"] = self.is_on_top(block1, block2)

        for block in blocks:
            predicates[f"ontable({block.label})"] = self.is_on_table(block, table)

            # clear -> we can stack upon it.
            predicates[f"clear({block.label})"] = self.is_clear(block, blocks) and not self.is_holding(robot, block)
            predicates[f"holding({block.label})"] = self.is_holding(robot, block)

        predicates[f"handempty({robot.label})"] = self.is_handempty(robot, blocks)
        predicates[f"handfull({robot.label})"] = self.is_handfull(robot, blocks)

        # Convert boolean results to PredicateState
        for key in predicates:
            predicates[key] = self._bool_to_predicate_state(predicates[key])

        return predicates

    """
    This section contains functions defining the spatial relations between objects for the Predicate Detectors.
    Some notes about the objects' geometry for inferring the needed thresholds to determine spatial relations between
    different objects:
    - coordinates are taken from the upper-left corner of the screen, x-axis goes right, y-axis goes down.
    - the (x_anchor, y_anchor) coordinate represents the upper-left corner of the object.
    - boxes that are stacked on one another have a gap of 2 px between their respective bounding boxes.
    - boxes that are stacked on the table are aligned prefectly with it, so there is 0 gap between the two bounding boxes.
    - if box1 is stacked upon box2, then the bottom of box1 should be perfectly aligned with the top of box2.
    - if box1 is placed on the table, then the bottom of box1 should have a gap of 2 px from the top of the table.
    # TODO: format and test later
    """
    @staticmethod
    def is_on_top(obj1: BoundedObject, obj2: BoundedObject):
        return obj1.bbox.on_top(obj2.bbox)

    def is_on_table(self, block: BoundedObject, table: BoundedObject):
        assert block.type == "block" and table.type == "table"

        return self.is_on_top(block, table)

    def is_clear(self, block: BoundedObject, objects: List[BoundedObject]):
        """
        Determines if a box is clear, i.e., no other box is on top of it.
        """
        assert block.type == "block"

        other_blocks = [obj for obj in objects if obj.type == block.type and obj.name != block.name]
        return all(not self.is_on_top(other_block, block) for other_block in other_blocks)

    @staticmethod
    def is_handempty(robot: BoundedObject, objects: List[BoundedObject]):
        """
        Determines if the robot is not holding any box by checking if no box is near its bottom edge.
        """
        assert robot.type == "robot"

        blocks = [obj for obj in objects if obj.type == "block"]
        return all(not robot.bbox.intersects(block.bbox) for block in blocks)

    # Function to check if the robot's hand is full (holding a box)
    def is_handfull(self, robot: BoundedObject, objects: List[BoundedObject]):
        """
        Determines if the robot is holding any box by checking if a box is near its bottom edge.
        """
        # If handempty is False, it means hand is full
        return not self.is_handempty(robot, objects)

    # Function to check if the robot is holding a specific box
    @staticmethod
    def is_holding(robot: BoundedObject, block: BoundedObject):
        return (robot.type == "robot" and
                block.type == "block" and
                robot.bbox.intersects(block.bbox))


if __name__ == "__main__":
    COLOR_MAP = { # this is in normalized RGB
        ObjectLabel('red:block'): (0.9, 0.1, 0.1),
        ObjectLabel('cyan:block'): np.array((0.43758721, 0.891773, 0.96366276)),
        ObjectLabel('blue:block'): (0.15896958, 0.11037514, 0.65632959),
        ObjectLabel('green:block'): (0.1494483 , 0.86812606, 0.16249293),
        ObjectLabel('yellow:block'): (0.94737059, 0.73085581, 0.25394164),
        ObjectLabel('pink:block'): (0.96157015, 0.23170163, 0.94931882),
        ObjectLabel('brown:table'): (0.5, 0.2, 0.0),
        ObjectLabel('gray:robot'): (0.4, 0.4, 0.4)
    }

    # Run the color-based detection pipeline
    image_path = "/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/src/fluent_classification/images/state_0008.png"  # Path to the uploaded image
    image = cv2.imread(image_path) # in BGR format
    color_object_detector = ColorObjectDetector(COLOR_MAP)
    detected_objects = color_object_detector.detect(image)
    # detected_objects = detect_objects_by_color(image, COLOR_MAP)
    detected_objects = {obj.label: obj for obj in detected_objects}

    # Print detected objects information
    for obj in detected_objects.values():
        print(f"Detected a {obj.label} at {obj.bbox.box}")


    # TODO: export the tests to some testing infrastructure (for a start, export to a different file)
    """
    Testing predicates on picture 0008
    """
    blocks_fluent_classifier = BlocksContourFluentClassifier(color_object_detector)
    # is_on_top tests
    assert blocks_fluent_classifier.is_on_top(detected_objects["blue:block"], detected_objects["cyan:block"])
    assert not blocks_fluent_classifier.is_on_top(detected_objects["cyan:block"], detected_objects["blue:block"])  # extreme case: blocksworld are touching but the second is on the first, so the function should fail
    assert not blocks_fluent_classifier.is_on_top(detected_objects["red:block"], detected_objects["cyan:block"])
    assert not blocks_fluent_classifier.is_on_top(detected_objects["green:block"], detected_objects["cyan:block"])  # extreme case: the block held by the robot is never placed on top of any other block
    assert not blocks_fluent_classifier.is_on_top(detected_objects["cyan:block"], detected_objects["cyan:block"])  # extreme case: a block cannot be on top of itself

    # is_clear_tests
    # TODO: add tests for cases in which at least one of the object is not a box
    assert blocks_fluent_classifier.is_clear(detected_objects["blue:block"], list(detected_objects.values()))
    assert not blocks_fluent_classifier.is_clear(detected_objects["red:block"], list(detected_objects.values()))
    assert blocks_fluent_classifier.is_clear(detected_objects["green:block"], list(detected_objects.values()))  # extreme case: for the box being held by the robot
    assert blocks_fluent_classifier.is_clear(detected_objects["gray:robot"], list(detected_objects.values()))  # extreme case: the object is not a box

    # is_on_table tests
    assert blocks_fluent_classifier.is_on_table(detected_objects["red:block"], detected_objects["brown:table"])
    assert blocks_fluent_classifier.is_on_table(detected_objects["pink:block"], detected_objects["brown:table"])
    assert not blocks_fluent_classifier.is_on_table(detected_objects["yellow:block"], detected_objects["brown:table"])
    assert not blocks_fluent_classifier.is_on_table(detected_objects["green:block"], detected_objects["brown:table"])
    assert not blocks_fluent_classifier.is_on_table(detected_objects["brown:table"], detected_objects["brown:table"])  # extreme case: table related to itself (should be validated in the function itself that the first object is a block

    # handempty tests
    assert not blocks_fluent_classifier.is_handempty(detected_objects["gray:robot"], list(detected_objects.values()))

    # handfull tests
    assert blocks_fluent_classifier.is_handfull(detected_objects["gray:robot"], list(detected_objects.values()))

    # is_holding tests
    assert blocks_fluent_classifier.is_holding(detected_objects["gray:robot"], detected_objects["green:block"])
    assert not blocks_fluent_classifier.is_holding(detected_objects["gray:robot"], detected_objects["cyan:block"])
    assert not blocks_fluent_classifier.is_holding(detected_objects["gray:robot"], detected_objects["gray:robot"])  # extreme case: the robot could not hold itself

    print(json.dumps(blocks_fluent_classifier.classify(image), indent=4))
    draw_objects(image, list(detected_objects.values()))
