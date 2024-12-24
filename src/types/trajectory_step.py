from src.types.trajectory_state import TrajectoryState


class TrajectoryStep:
    step: int
    current_state: TrajectoryState
    ground_action: str
    operator_object_assignment: str
    lifted_preconds: str
    next_state: TrajectoryState

    def __init__(self,
                 step: int,
                 current_state: TrajectoryState,
                 ground_action: str,
                 operator_object_assignment: str,
                 lifted_preconds: str,
                 next_state: TrajectoryState):
        self.step = step
        self.current_state = current_state
        self.ground_action = ground_action
        self.operator_object_assignment = operator_object_assignment
        self.lifted_preconds = lifted_preconds
        self.next_state = next_state
