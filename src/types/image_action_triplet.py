from .action_triplet import ActionTriplet
from .action import Action
from .image import Image


class ImageActionTriplet(ActionTriplet):
    def __init__(self, prev_image: Image, action: Action, next_image: Image):
        super().__init__(prev_state=prev_image, action=action, next_state=next_image)

    @staticmethod
    def _validate_triplet(prev_state: Image, action: Action, next_state: Image):
        raise NotImplementedError  # TODO: implement
