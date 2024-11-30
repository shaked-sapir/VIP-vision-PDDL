from typing import Tuple

import cv2


def to_int_rgb(normalized_rgb_tuple: Tuple[float, float, float]) -> Tuple[int, ...]:
    return tuple(round(v*255) for v in normalized_rgb_tuple)


def find_exact_rgb_color_mask_with_compare(image, rgb_tuple):
    # Create masks for each channel comparison
    mask_r = cv2.compare(image[:, :, 2], rgb_tuple[0], cv2.CMP_EQ)  # Red channel
    mask_g = cv2.compare(image[:, :, 1], rgb_tuple[1], cv2.CMP_EQ)  # Green channel
    mask_b = cv2.compare(image[:, :, 0], rgb_tuple[2], cv2.CMP_EQ)  # Blue channel

    # Combine masks to ensure all channels match
    mask = cv2.bitwise_and(mask_r, cv2.bitwise_and(mask_g, mask_b))
    return mask
