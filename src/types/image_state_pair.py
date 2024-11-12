from typing import Tuple

from . import State
from .image import Image


class ImageStatePair:
    def __init__(self, image: Image, state: State):
        self.image = image
        self.state = state

    def get_image(self) -> Image:
        return self.image

    def get_state(self) -> State:
        return self.state

    def get_pair(self) -> Tuple[Image, State]:
        return self.image, self.state
