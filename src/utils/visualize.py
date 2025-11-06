from pathlib import Path
from typing import List
from typing import Tuple, Union

import cv2
import numpy as np

from src.object_detection.bounded_object import BoundedObject

NormalizedRGB = Union[Tuple[float, float, float], np.array]
IntRGB = Union[Tuple[int, int, int], np.array]


def draw_objects(image: cv2.typing.MatLike, objects: List[BoundedObject]):
    image_copy = image.copy()

    for obj in objects:
        x_anchor, y_anchor, width, height = obj.bbox.box
        name, obj_type, label = (obj.name, obj.type, obj.label)

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


def to_int_rgb(normalized_rgb_triplet: NormalizedRGB) -> IntRGB:
    return tuple(round(v*255) for v in normalized_rgb_triplet)


def find_exact_rgb_color_mask(image_bgr: cv2.typing.MatLike, int_rgb_triplet: IntRGB):
    """
    This function finds a specific color mask in the given image and returns it.
    :param image_bgr: the source image. as it is coming from cv2, we assume that it is in BGR format and
                      currently supporting this format only.
    :param int_rgb_triplet: the channels of the color mask to find
    :return: relevant color mask. (could be covering multiple areas in the image in case there are such.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Create masks for each channel comparison
    mask_r = cv2.compare(image_rgb[:, :, 0], int_rgb_triplet[0], cv2.CMP_EQ)
    mask_g = cv2.compare(image_rgb[:, :, 1], int_rgb_triplet[1], cv2.CMP_EQ)
    mask_b = cv2.compare(image_rgb[:, :, 2], int_rgb_triplet[2], cv2.CMP_EQ)

    # Combine masks to ensure all channels match
    mask = cv2.bitwise_and(mask_r, cv2.bitwise_and(mask_g, mask_b))

    return mask


def load_image(image_path: str) -> cv2.typing.MatLike:
    return cv2.imread(image_path)
