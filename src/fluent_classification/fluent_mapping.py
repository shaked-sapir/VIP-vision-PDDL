from typing import Set

from src.types import Fluent


class FluentMapping:
    def __init__(self, num_states: int):
        self.num_states = num_states
        self.mapping = {i: set() for i in range(num_states)}

    def get_fluents_of_state(self, state_index: int) -> Set[Fluent]:
        return self.mapping[state_index]
