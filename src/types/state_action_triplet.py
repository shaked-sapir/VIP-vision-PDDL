from .action_triplet import ActionTriplet
from .state import State
from .action import Action
from functools import reduce
from operator import and_

class StateActionTriplet(ActionTriplet):
    def __init__(self, prev_state: State, action: Action, next_state: State):
        super().__init__(prev_state, action, next_state)

    @staticmethod
    # TODO: check if it fits to the final definition of the ActionSchema
    def _validate_triplet(prev_state: State, action: Action, next_state: State):
        assert reduce(and_, [fluent in prev_state.fluents for fluent in action.preconditions])
        assert reduce(and_, [fluent in next_state.fluents for fluent in action.effects])
