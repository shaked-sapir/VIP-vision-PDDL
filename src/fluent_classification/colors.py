from typing import Tuple, Union

import cv2
import numpy as np

NormalizedRGB = Union[Tuple[float, float, float], np.array]
IntRGB = Union[Tuple[int, int, int], np.array]


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
