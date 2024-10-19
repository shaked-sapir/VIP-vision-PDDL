from pddlgym.structs import State as PGState

from src.types.action_triplet import StateRepresentation


class State(PGState, StateRepresentation):
    """
        At the moment, this class implements the exact same logic as PDDLGym's State.
        It includes signals for goal states, literals ("fluents") and objects composing the state.
    """
    pass
