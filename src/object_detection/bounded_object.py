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

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return self.x_anchor, self.y_anchor, self.width, self.height

    def intersects(self, other: "BoundingBox") -> bool:
        """
        Determines if `self` box intersects with `other` box.

        type of spatial relation: Symmetrical.

        :param other: other bounding box
        :return: True if `self` box intersects with `other` box, False otherwise
        """
        return not (
                self.x_anchor + self.width < other.x_anchor or  # self is to the left of other
                self.x_anchor > other.x_anchor + other.width or  # self is to the right of other
                self.y_anchor > other.y_anchor + other.height or  # self is below other
                self.y_anchor + self.height < other.y_anchor  # self is above other
        )

    def on_top(self, other: "BoundingBox", threshold=10) -> bool:
        """
        Determines if `self` box is on top of `other` box based on bounding box coordinates by checking if the bottom
        edge of `self` box is close (within some threshold) to the top edge of `other` box and they are horizontally aligned.
        it considers `self` box to be on top of `other` box if `self` box overlaps with the top of `other` box and
        not necessarily perfectly aligned with it.

        type of spatial relation: Asymmetrical.

        :param other: other bounding box
        :param threshold: threshold for tolerating "closeness" in vertical distance between boxes
        :return: True if self is on top edge of other bounding box, False otherwise
        """
        x1, y1, w1, h1 = self.box
        x2, y2, w2, h2 = other.box

        horizontal_overlap = (x1 < x2 + w2) and (x1 + w1 > x2)
        vertical_gap = y2 - (y1 + h1)

        return horizontal_overlap and 0 < vertical_gap < threshold


class BoundedObject:
    def __init__(self, obj_type: str, name: str, x_anchor: int, y_anchor: int, width: int, height: int):
        self._type = obj_type
        self._name = name
        self.label = f"{self._name}:{self._type}" # TODO Maybe: turn this into an ObjectLabel type
        self.bounding_box = BoundingBox(x_anchor, y_anchor, width, height)

    @property
    def type(self) -> str:
        return self._type

    @property
    def name(self):
        return self._name
