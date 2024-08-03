class State:
    def __init__(self, is_initial: bool=False, is_goal:bool =False, fluents = None):
        self.is_initial=is_initial
        self.is_goal = is_goal
        self.fluents = fluents

    def add_fluents(self, fluents: Any):
        raise NotImplementedError

    def remove_fluents(self, fluents: Any):
        raise NotImplementedError
