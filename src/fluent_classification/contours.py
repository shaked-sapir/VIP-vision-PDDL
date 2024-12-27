from typing import List, Tuple, Dict
import itertools
import cv2
import json
import numpy as np

from src.fluent_classification.blocks_fluent_classifier import BlocksFluentClassifier
from src.fluent_classification.colors import find_exact_rgb_color_mask, to_int_rgb, NormalizedRGB
from src.object_detection.bounded_object import BoundedObject
from src.object_detection.color_object_detector import ColorObjectDetector, ObjectName

# # TODO IMM: this should be a part of the ObjectDetector module as it creates the bounding box of the detected objects.
# class BoundingBox:
#
#     def __init__(self, x_anchor: int, y_anchor: int, width: int, height: int): #TODO LATER: check if it could be float
#         """
#         :param x_anchor: the x coordinate of the upper-left corner of the bounding box
#         :param y_anchor: the y coordinate of the upper-left corner of the bounding box
#         :param width: the width of the bounding box
#         :param height: the height of the bounding box
#         """
#         self.x_anchor = x_anchor
#         self.y_anchor = y_anchor
#         self.width = width
#         self.height = height
#
#     def decompose(self) -> Tuple[int, int, int, int]:
#         return self.x_anchor, self.y_anchor, self.width, self.height
#
#     def intersects(self, other: "BoundingBox") -> bool:
#         return not (
#                 self.x_anchor + self.width < other.x_anchor or  # self is to the left of other
#                 self.x_anchor > other.x_anchor + other.width or  # self is to the right of other
#                 self.y_anchor > other.y_anchor + other.height or # self is below other
#                 self.y_anchor + self.height < other.y_anchor  # self is above other
#         )
#
#
# # TODO IMM: this also should be a part of the ObjectDetector module.
# # TODO: make this a DTO because there are a lot of properties to the object
# class BoundedObject:
#     def __init__(self, obj_type: str, name: str, x_anchor: int, y_anchor: int, width: int, height: int):
#         self.obj_type = obj_type
#         self.name = name
#         self.label = f"{self.name}:{self.obj_type}"
#         self.bounding_box = BoundingBox(x_anchor, y_anchor, width, height)


# TODO IMM: this is actually the `classify fluents` algorithm
# TODO: this is specific to blocks and therefore should be in an ad-hoc file related to blocks
"""
so as same as in the detect_objects_by_color,  we need to have a base `FluentClassifier` class (or interface)
and then create an actual instance for the blocks.
"""
# def get_image_predicates(image: cv2.typing.MatLike, object_color_map: Dict[ObjectName, NormalizedRGB]) -> Dict[str, bool]:
#     """
#     function that returns the grounded predicates appearing in the image.
#     it creates all the possible groundings using all objects detected in the image, then uses
#     predicate classifiers to determine whether the predicates hold.
#     :param image: the image to predict on
#     :param object_color_map: mapping between an object to its color so we can detect it in the image.
#     :return: a dictionary of the form {<grounded_predicate>: True/False}
#     """
#     predicates = {}
#     color_object_detector = ColorObjectDetector(object_color_map)
#     detected_objects = color_object_detector.detect(image)
#     # detected_objects = detect_objects_by_color(image, object_color_map)
#     for obj in detected_objects:
#         print(f"Detected a {obj.label} at {obj.bounding_box.decompose()}")
#     blocks = [obj for obj in detected_objects if obj.obj_type == "block"]
#     robot = [obj for obj in detected_objects if obj.obj_type == "robot"][0]
#     table = [obj for obj in detected_objects if obj.obj_type == "table"][0]
#
#     for block1, block2 in itertools.permutations(blocks, 2):
#         predicates[f"on({block1.label},{block2.label})"] = is_on_top(block1, block2)
#
#     for block in blocks:
#         predicates[f"ontable({block.label})"] = is_on_table(block, table)
#
#         # clear -> we can stack upon it.
#         predicates[f"clear({block.label})"] = is_clear(block, blocks) and not is_holding(robot, block)
#         predicates[f"holding({block.label})"] = is_holding(robot, block)
#
#     predicates[f"handempty({robot.label})"] = is_handempty(robot, blocks)
#     predicates[f"handfull({robot.label})"] = is_handfull(robot, blocks)
#
#     return predicates


# # TODO IMM: consider to export this one to some external file, and make it the ObjectDetector module
# """
# notice that this is only valid for domains in which objects could be detected solely by color, so i need to define
# a base class for an ObjectDetector (maybe more of interface) and then implement this one as a child class.
# """
# def detect_objects_by_color(image: cv2.typing.MatLike, color_map: Dict[str, NormalizedRGB]) -> List[BoundedObject]:
#     detected_objects = []
#
#     for object_name, color_tuple in color_map.items():
#         # Create a mask for the current color range
#         full_rgb_tuple = to_int_rgb(color_tuple)
#         mask = find_exact_rgb_color_mask(image, full_rgb_tuple)
#
#         # Find contours for the masked region
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         for contour in contours:
#             x, y, w, h = cv2.boundingRect(contour)
#
#             # Save detected object info
#             detected_objects.append(
#                 BoundedObject(
#                     obj_type=object_name.split(':')[1],
#                     name=object_name.split(':')[0],
#                     x_anchor=x,
#                     y_anchor=y,
#                     width=w,
#                     height=h
#                 )
#             )
#
#     return detected_objects


# TODO IMM2: extract to drawing file, this is general to objects (and specifically bounding boxes) and not specific to blocks





# # TODO: for all predicates, see if we can export to spatial_relation file (as all of them should relate to relations between bounding box and not for blocks specifically
# # Function to check if box1 is on top of box2
# def is_on_top(obj1: BoundedObject, obj2: BoundedObject, threshold=10):
#     """
#     Determines if box1 is on top of box2 based on bounding box coordinates.
#     """
#     x1, y1, w1, h1 = obj1.bounding_box.decompose()  # Coordinates of box1
#     x2, y2, w2, h2 = obj2.bounding_box.decompose()  # Coordinates of box2
#
#     # Check if the bottom edge of box1 is close to the top edge of box2 and they are horizontally aligned.
#     # it considers box1 to be on top of box2 if box1 bottom's overalp, and not necessarily exactly aligned with the top of box2.
#     horizontal_overlap = (x1 < x2 + w2) and (x1 + w1 > x2)
#     vertical_gap = y2-(y1+h1)
#
#     return horizontal_overlap and 0 < vertical_gap < threshold
#
#
# # Function to check if a box is directly on the table
# def is_on_table(block: BoundedObject, table: BoundedObject, threshold=10):
#     """
#     Determines if a box is directly on the table based on bounding box coordinates.
#     """
#     x, y, w, h = block.bounding_box.decompose()  # Coordinates of the box
#     table_x, table_y, table_w, table_h = table.bounding_box.decompose()  # Coordinates of the table
#
#     # Check if the bottom edge of the box is close to the top edge of the table and they are horizontally aligned
#     vertical_gap = table_y-(y + h)
#     horizontal_overlap = (x < table_x + table_w) and (x + w > table_x)
#
#     return horizontal_overlap and 0 < vertical_gap < threshold
#
#
# def is_clear(block: BoundedObject, objects: List[BoundedObject], threshold=10):
#     """
#     Determines if a box is clear, i.e., no other box is on top of it.
#     """
#     for obj in objects:
#         # Skip the box itself
#         if obj.bounding_box.decompose() == block.bounding_box.decompose():
#             continue
#
#         # Check if any other box is on top of this box
#         if is_on_top(obj, block, threshold):
#             return False
#     return True
#
#
# def is_handempty(robot: BoundedObject, objects: List[BoundedObject]):
#     """
#     Determines if the robot is not holding any box by checking if no box is near its bottom edge.
#     """
#     # x_robot, y_robot, w_robot, h_robot = robot  # Coordinates of the robot
#
#     # Check if there is any box within holding range of the robot
#     for obj in objects:
#         if obj.obj_type == "block":
#             if robot.bounding_box.intersects(obj.bounding_box):
#                 return False  # If there's an intersection, the robot's hand is not empty
#     return True
#
#
# # Function to check if the robot's hand is full (holding a box)
# def is_handfull(robot, objects: List[BoundedObject]):
#     """
#     Determines if the robot is holding any box by checking if a box is near its bottom edge.
#     """
#     # If handempty is False, it means hand is full
#     return not is_handempty(robot, objects)
#
#
# # Function to check if the robot is holding a specific box
# def is_holding(robot:BoundedObject, block: BoundedObject):
#     return (robot.obj_type == "robot" and
#             block.obj_type == "block" and
#             robot.bounding_box.intersects(block.bounding_box))


