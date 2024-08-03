from typing import List

from .fluent import Fluent

class Action:
    preconditions: List[Fluent]
    effects: List[Fluent]
    def __init__(self, preconditions: List[Fluent], effects: List[Fluent]):
        self.preconditions = preconditions
        self.effects = effects



