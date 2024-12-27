from typing import List

import cv2

from src.object_detection.bounded_object import BoundedObject


def draw_objects(image: cv2.typing.MatLike, objects: List[BoundedObject]):
    image_copy = image.copy()

    for obj in objects:
        x_anchor, y_anchor, width, height = obj.bounding_box.decompose()
        name, obj_type, label = (obj.name, obj.obj_type, obj.label)

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
