from typing import List

import cv2
import numpy as np

# TODO: extract this to the types dir
# TODO: make this a DTO because there are a lot of properties to the object
class VisualObject:
    def __init__(self, obj_type: str, color: str, x_anchor: int, y_anchor: int, width: int, height: int):
        self.obj_type = obj_type
        self.color = color
        self.x_anchor = x_anchor
        self.y_anchor = y_anchor
        self.width = width
        self.height = height
        self.label = f"{self.obj_type}:{self.color}"
        self.bounding_box = (x_anchor, y_anchor, width, height)

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
        x_anchor, y_anchor, width, height, color, obj_type, label = (
            obj.x_anchor, obj.y_anchor, obj.width, obj.height, obj.color, obj.obj_type, obj.label
        )
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


# Run the color-based detection pipeline
image_path = "/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/src/domains/blocks_images/state_0008.png"  # Path to the uploaded image
image = cv2.imread(image_path)

detected_objects = detect_objects_by_color(image)

# Print detected objects information
for obj in detected_objects:
    print(f"Detected a {obj.label} at {obj.bounding_box}")

draw_objects(image, detected_objects)