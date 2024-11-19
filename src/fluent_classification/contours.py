from typing import List, Tuple, Dict
import itertools
import cv2
import json
import numpy as np


class BoundingBox:

    def __init__(self, x_anchor: int, y_anchor: int, width: int, height: int):
        """
        :param x_anchor: the x coordinate of the upper-left corner of the bounding box
        :param y_anchor: the y coordinate of the upper-left corner of the bounding box
        :param width: the width of the bounding box
        :param height: the height of the bounding box
        """
        self.x_anchor = x_anchor
        self.y_anchor = y_anchor
        self.width = width
        self.height = height

    def decompose(self) -> Tuple[int, int, int, int]:
        return self.x_anchor, self.y_anchor, self.width, self.height

    def intersects(self, other: "BoundingBox") -> bool:
        return not (
                self.x_anchor + self.width < other.x_anchor or  # self is to the left of other
                self.x_anchor > other.x_anchor + other.width or  # self is to the right of other
                self.y_anchor > other.y_anchor + other.height or # self is below other
                self.y_anchor + self.height < other.y_anchor  # self is above other
        )


# TODO: extract this to the types dir
# TODO: make this a DTO because there are a lot of properties to the object
class VisualObject:
    def __init__(self, obj_type: str, color: str, x_anchor: int, y_anchor: int, width: int, height: int):
        self.obj_type = obj_type
        self.color = color
        self.label = f"{self.obj_type}:{self.color}"
        self.bounding_box = BoundingBox(x_anchor, y_anchor, width, height)

# Define color ranges in HSV for each object in the scene, including cyan
COLOR_RANGES = {
    'green_block': ((50, 100, 100), (70, 255, 255)),
    'blue_block': ((100, 100, 100), (130, 255, 255)),
    'cyan_block': ((85, 100, 100), (95, 255, 255)),
    'red_block': ((0, 100, 100), (10, 255, 255)),
    'yellow_block': ((20, 100, 100), (30, 255, 255)),
    'pink_block': ((140, 100, 100), (170, 255, 255)),
    'brown_table': ((10, 100, 20), (20, 255, 200)),
    'gray_robot': ((0, 0, 60), (180, 50, 130))
}


def get_image_predicates(image: cv2.typing.MatLike) -> Dict[str, bool]:
    """
    function that returns the grounded predicates appearing in the image.
    it creates all the possible groundings using all objects detected in the image, then uses
    predicate classifiers to determine whether the predicates hold.
    :param image: the image to predict on
    :return: a dictionary of the form {<grounded_predicate>: True/False}
    """
    predicates = {}
    detected_objects = detect_objects_by_color(image)
    blocks = [obj for obj in detected_objects if obj.obj_type == "block"]
    robot = [obj for obj in detected_objects if obj.obj_type == "robot"][0]
    table = [obj for obj in detected_objects if obj.obj_type == "table"][0]

    for block1, block2 in itertools.permutations(blocks, 2):
        predicates[f"on {block1.label} {block2.label}"] = is_on_top(block1, block2)

    for block in blocks:
        predicates[f"ontable {block.label}"] = is_on_table(block, table)
        predicates[f"clear {block.label}"] = is_clear(block, blocks)
        predicates[f"holding {block.label}"] = is_holding(robot, block)

    predicates[f"handempty {robot.label}"] = is_handempty(robot, blocks)
    predicates[f"handfull {robot.label}"] = is_handfull(robot, blocks)

    return predicates

# Function to detect objects by color
def detect_objects_by_color(image: cv2.typing.MatLike):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_copy = image.copy()

    detected_objects = []

    for object_name, (lower, upper) in COLOR_RANGES.items():
        # Create a mask for the current color range
        mask = cv2.inRange(hsv_image, lower, upper)

        # Find contours for the masked region
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Save detected object info
            detected_objects.append(
                VisualObject(
                    obj_type=object_name.split('_')[1],
                    color=object_name.split('_')[0],
                    x_anchor=x,
                    y_anchor=y,
                    width=w,
                    height=h
                )
            )

    return detected_objects


def draw_objects(image: cv2.typing.MatLike, detected_objects: List[VisualObject]):
    image_copy = image.copy()

    for obj in detected_objects:
        x_anchor, y_anchor, width, height = obj.bounding_box.decompose()
        color, obj_type, label = (obj.color, obj.obj_type, obj.label)

        font_scale = min(width, height) / 100  # Scale font based on box size
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        text_x = x_anchor + 5  # Slight padding inside the box
        text_y = y_anchor + height - 5 if text_size[1] < height else y_anchor + height // 2  # Center the text vertically if box is small

        cv2.rectangle(image_copy, (x_anchor, y_anchor), (x_anchor + width, y_anchor + height), color=(0, 255, 0), thickness=2)
        cv2.putText(image_copy, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 1)

    # Show the image with detected objects
    cv2.imshow("Detected Objects", image_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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


# Function to check if box1 is on top of box2
def is_on_top(obj1: VisualObject, obj2: VisualObject, threshold=10):
    """
    Determines if box1 is on top of box2 based on bounding box coordinates.
    """
    x1, y1, w1, h1 = obj1.bounding_box.decompose()  # Coordinates of box1
    x2, y2, w2, h2 = obj2.bounding_box.decompose()  # Coordinates of box2

    # Check if the bottom edge of box1 is close to the top edge of box2 and they are horizontally aligned.
    # it considers box1 to be on top of box2 if box1 bottom's overalp, and not necessarily exactly aligned with the top of box2.
    horizontal_overlap = (x1 < x2 + w2) and (x1 + w1 > x2)
    vertical_gap = y2-(y1+h1)

    return horizontal_overlap and 0 < vertical_gap < threshold


# Function to check if a box is directly on the table
def is_on_table(block: VisualObject, table: VisualObject, threshold=10):
    """
    Determines if a box is directly on the table based on bounding box coordinates.
    """
    x, y, w, h = block.bounding_box.decompose()  # Coordinates of the box
    table_x, table_y, table_w, table_h = table.bounding_box.decompose()  # Coordinates of the table

    # Check if the bottom edge of the box is close to the top edge of the table and they are horizontally aligned
    vertical_gap = table_y-(y + h)
    horizontal_overlap = (x < table_x + table_w) and (x + w > table_x)

    return horizontal_overlap and 0 < vertical_gap < threshold


def is_clear(block: VisualObject, objects: List[VisualObject], threshold=10):
    """
    Determines if a box is clear, i.e., no other box is on top of it.
    """
    for obj in objects:
        # Skip the box itself
        if obj.bounding_box.decompose() == block.bounding_box.decompose():
            continue

        # Check if any other box is on top of this box
        if is_on_top(obj, block, threshold):
            return False
    return True


def is_handempty(robot: VisualObject, objects: List[VisualObject]):
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
def is_handfull(robot, objects: List[VisualObject]):
    """
    Determines if the robot is holding any box by checking if a box is near its bottom edge.
    """
    # If handempty is False, it means hand is full
    return not is_handempty(robot, objects)


# Function to check if the robot is holding a specific box
def is_holding(robot:VisualObject, block: VisualObject):
    return (robot.obj_type == "robot" and
            block.obj_type == "block" and
            robot.bounding_box.intersects(block.bounding_box))


# Run the color-based detection pipeline
image_path = "/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/src/domains/blocks_images/state_0008.png"  # Path to the uploaded image
image = cv2.imread(image_path)

detected_objects = detect_objects_by_color(image)
detected_objects = {obj.label: obj for obj in detected_objects}

# Print detected objects information
for obj in detected_objects.values():
    print(f"Detected a {obj.label} at {obj.bounding_box.decompose()}")


"""
Testing predicates on picture 0008
"""

# is_on_top tests
assert is_on_top(detected_objects["block:blue"], detected_objects["block:cyan"])
assert not is_on_top(detected_objects["block:cyan"], detected_objects["block:blue"])  # extreme case: blocks are touching but the second is on the first, so the function should fail
assert not is_on_top(detected_objects["block:red"], detected_objects["block:cyan"])
assert not is_on_top(detected_objects["block:green"], detected_objects["block:cyan"])  # extreme case: the block held by the robot is never placed on top of any other block
assert not is_on_top(detected_objects["block:cyan"], detected_objects["block:cyan"])  # extreme case: a block cannot be on top of itself

# is_clear_tests
# TODO: add tests for cases in which at least one of the object is not a box
assert is_clear(detected_objects["block:blue"], list(detected_objects.values()))
assert not is_clear(detected_objects["block:red"], list(detected_objects.values()))
assert is_clear(detected_objects["block:green"], list(detected_objects.values()))  # extreme case: for the box being held by the robot
assert is_clear(detected_objects["robot:gray"], list(detected_objects.values()))  # extreme case: the object is not a box

# is_on_table tests
assert is_on_table(detected_objects["block:red"], detected_objects["table:brown"])
assert is_on_table(detected_objects["block:pink"], detected_objects["table:brown"])
assert not is_on_table(detected_objects["block:yellow"], detected_objects["table:brown"])
assert not is_on_table(detected_objects["block:green"], detected_objects["table:brown"])
assert not is_on_table(detected_objects["table:brown"], detected_objects["table:brown"])  # extreme case: table related to itself (should be validated in the function itself that the first object is a block

# handempty tests
assert not is_handempty(detected_objects["robot:gray"], list(detected_objects.values()))

# handfull tests
assert is_handfull(detected_objects["robot:gray"], list(detected_objects.values()))

# is_holding tests
assert is_holding(detected_objects["robot:gray"], detected_objects["block:green"])
assert not is_holding(detected_objects["robot:gray"], detected_objects["block:cyan"])
assert not is_holding(detected_objects["robot:gray"], detected_objects["robot:gray"])  # extreme case: the robot could not hold itself

print(json.dumps(get_image_predicates(image), indent=4))
draw_objects(image, list(detected_objects.values()))
