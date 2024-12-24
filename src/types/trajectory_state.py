from typing import List


class TrajectoryState:
    literals: List[str]
    objects: List[str]
    goal: List[str]

    def __init__(self, literals: List[str], objects: List[str], goal:List[str]):
        self.literals = literals
        self.objects = objects
        self.goal = goal
