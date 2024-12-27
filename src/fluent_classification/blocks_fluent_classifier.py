import itertools
import json
from typing import List, Dict

from src.fluent_classification.base_fluent_classifier import BaseFluentClassifier
from src.object_detection.bounded_object import BoundedObject
from src.utils.visualize import draw_objects


class BlocksFluentClassifier(BaseFluentClassifier):

    def classify(self, image) -> Dict[str, bool]:
        predicates = {}
        detected_objects = self.object_detector.detect(image)

        for obj in detected_objects:
            print(f"Detected a {obj.label} at {obj.bounding_box.decompose()}")
        blocks = [obj for obj in detected_objects if obj.obj_type == "block"]
        robot = [obj for obj in detected_objects if obj.obj_type == "robot"][0]
        table = [obj for obj in detected_objects if obj.obj_type == "table"][0]

        for block1, block2 in itertools.permutations(blocks, 2):
            predicates[f"on({block1.label},{block2.label})"] = self.is_on_top(block1, block2)

        for block in blocks:
            predicates[f"ontable({block.label})"] = self.is_on_table(block, table)

            # clear -> we can stack upon it.
            predicates[f"clear({block.label})"] = self.is_clear(block, blocks) and not self.is_holding(robot, block)
            predicates[f"holding({block.label})"] = self.is_holding(robot, block)

        predicates[f"handempty({robot.label})"] = self.is_handempty(robot, blocks)
        predicates[f"handfull({robot.label})"] = self.is_handfull(robot, blocks)

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
    # TODO: for all predicates, see if we can export to spatial_relation file (as all of them should relate to relations between bounding box and not for blocks specifically
    # Function to check if box1 is on top of box2
    def is_on_top(self, obj1: BoundedObject, obj2: BoundedObject, threshold=10):
        """
        Determines if box1 is on top of box2 based on bounding box coordinates.
        """
        x1, y1, w1, h1 = obj1.bounding_box.decompose()  # Coordinates of box1
        x2, y2, w2, h2 = obj2.bounding_box.decompose()  # Coordinates of box2

        # Check if the bottom edge of box1 is close to the top edge of box2 and they are horizontally aligned.
        # it considers box1 to be on top of box2 if box1 bottom's overalp, and not necessarily exactly aligned with the top of box2.
        horizontal_overlap = (x1 < x2 + w2) and (x1 + w1 > x2)
        vertical_gap = y2 - (y1 + h1)

        return horizontal_overlap and 0 < vertical_gap < threshold

    # Function to check if a box is directly on the table
    def is_on_table(self, block: BoundedObject, table: BoundedObject, threshold=10):
        """
        Determines if a box is directly on the table based on bounding box coordinates.
        """
        x, y, w, h = block.bounding_box.decompose()  # Coordinates of the box
        table_x, table_y, table_w, table_h = table.bounding_box.decompose()  # Coordinates of the table

        # Check if the bottom edge of the box is close to the top edge of the table and they are horizontally aligned
        vertical_gap = table_y - (y + h)
        horizontal_overlap = (x < table_x + table_w) and (x + w > table_x)

        return horizontal_overlap and 0 < vertical_gap < threshold

    def is_clear(self, block: BoundedObject, objects: List[BoundedObject], threshold=10):
        """
        Determines if a box is clear, i.e., no other box is on top of it.
        """
        for obj in objects:
            # Skip the box itself
            if obj.bounding_box.decompose() == block.bounding_box.decompose():
                continue

            # Check if any other box is on top of this box
            if self.is_on_top(obj, block, threshold):
                return False
        return True

    def is_handempty(self, robot: BoundedObject, objects: List[BoundedObject]):
        """
        Determines if the robot is not holding any box by checking if no box is near its bottom edge.
        """
        # x_robot, y_robot, w_robot, h_robot = robot  # Coordinates of the robot

        # Check if there is any box within holding range of the robot
        for obj in objects:
            if obj.obj_type == "block":
                if robot.bounding_box.intersects(obj.bounding_box):
                    return False  # If there's an intersection, the robot's hand is not empty
        return True

    # Function to check if the robot's hand is full (holding a box)
    def is_handfull(self, robot: BoundedObject, objects: List[BoundedObject]):
        """
        Determines if the robot is holding any box by checking if a box is near its bottom edge.
        """
        # If handempty is False, it means hand is full
        return not self.is_handempty(robot, objects)

    # Function to check if the robot is holding a specific box
    def is_holding(self, robot: BoundedObject, block: BoundedObject):
        return (robot.obj_type == "robot" and
                block.obj_type == "block" and
                robot.bounding_box.intersects(block.bounding_box))


if __name__ == "__main__":
    COLOR_MAP = { # this is in normalized RGB
        'red:block': (0.9, 0.1, 0.1),
        'cyan:block': np.array((0.43758721, 0.891773, 0.96366276)),
        'blue:block': (0.15896958, 0.11037514, 0.65632959),
        'green:block': (0.1494483 , 0.86812606, 0.16249293),
        'yellow:block': (0.94737059, 0.73085581, 0.25394164),
        'pink:block': (0.96157015, 0.23170163, 0.94931882),
        'brown:table': (0.5,0.2,0.0),
        'gray:robot': (0.4, 0.4, 0.4)
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
        print(f"Detected a {obj.label} at {obj.bounding_box.decompose()}")


    # TODO: export the tests to some testing infrastructure (for a start, export to a different file)
    """
    Testing predicates on picture 0008
    """
    blocks_fluent_classifier = BlocksFluentClassifier(color_object_detector)
    # is_on_top tests
    assert blocks_fluent_classifier.is_on_top(detected_objects["blue:block"], detected_objects["cyan:block"])
    assert not blocks_fluent_classifier.is_on_top(detected_objects["cyan:block"], detected_objects["blue:block"])  # extreme case: blocks are touching but the second is on the first, so the function should fail
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
