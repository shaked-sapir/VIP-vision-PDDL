from typing import Tuple, Union

import cv2
import numpy as np

NormalizedRGB = Union[Tuple[float, float, float], np.array]
IntRGB = Union[Tuple[int, int, int], np.array]

def to_int_rgb(normalized_rgb_triplet: NormalizedRGB) -> IntRGB:
    return tuple(round(v*255) for v in normalized_rgb_triplet)


def find_exact_rgb_color_mask(image, int_rgb_triplet: IntRGB):
    # Create masks for each channel comparison
    mask_r = cv2.compare(image[:, :, 2], int_rgb_triplet[0], cv2.CMP_EQ)  # Red channel
    mask_g = cv2.compare(image[:, :, 1], int_rgb_triplet[1], cv2.CMP_EQ)  # Green channel
    mask_b = cv2.compare(image[:, :, 0], int_rgb_triplet[2], cv2.CMP_EQ)  # Blue channel

    # Combine masks to ensure all channels match
    mask = cv2.bitwise_and(mask_r, cv2.bitwise_and(mask_g, mask_b))
    return mask
