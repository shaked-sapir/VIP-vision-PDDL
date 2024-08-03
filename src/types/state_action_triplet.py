from .state import State
from .action import Action
from functools import reduce
from operator import and_

class StateActionTriplet:
    def __init__(self, prev_state: State, action: Action, next_state: State):
        self._validate_triplet(prev_state, action, next_state)

        self.prev_state = prev_state
        self.action = action
        self.next_state = next_state

    @staticmethod
    def _validate_triplet(prev_state: State, action: Action, next_state: State):
        assert reduce(and_, [fluent in prev_state.fluents for fluent in action.preconditions])
        assert reduce(and_, [fluent in next_state.fluents for fluent in action.effects])
