from abc import ABC, abstractmethod
from typing import Tuple

from .action import Action


class StateRepresentation:
    """
    This class is meant to be a parent class for possible state representations,
    and should not be instantiated directly.
    """
    pass



class ActionTriplet(ABC):
    def __init__(self, prev_state: StateRepresentation, action: Action, next_state: StateRepresentation):
        self._validate_triplet(prev_state, action, next_state)

        self.prev_state = prev_state
        self.action = action
        self.next_state = next_state


    @staticmethod
    @abstractmethod
    def _validate_triplet(prev_state: StateRepresentation, action: Action, next_state: StateRepresentation):
        pass

    def get_prev_state(self) -> StateRepresentation:
        return self.prev_state

    def get_action(self) -> Action:
        return self.action

    def get_next_state(self) -> StateRepresentation:
        return self.next_state

    def get_triplet(self) -> Tuple[StateRepresentation, Action, StateRepresentation]:
        return self.prev_state, self.action, self.next_state
