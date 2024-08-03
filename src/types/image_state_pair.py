from .image import Image
from .state import State

class ImageStatePair:
    image: Image
    state: State
    def __init__(self, image: Image, state: State):
        self.image = image
        self.state = state