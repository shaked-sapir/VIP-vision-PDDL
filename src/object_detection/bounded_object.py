from typing import Tuple


class BoundingBox:

    def __init__(self, x_anchor: int, y_anchor: int, width: int, height: int):  # TODO LATER: check if it could be float
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
                self.y_anchor > other.y_anchor + other.height or  # self is below other
                self.y_anchor + self.height < other.y_anchor  # self is above other
        )


# TODO: make this a DTO because there are a lot of properties to the object
class BoundedObject:
    def __init__(self, obj_type: str, name: str, x_anchor: int, y_anchor: int, width: int, height: int):
        self.obj_type = obj_type
        self.name = name
        self.label = f"{self.name}:{self.obj_type}"
        self.bounding_box = BoundingBox(x_anchor, y_anchor, width, height)
