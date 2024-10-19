from typing import List, Sequence, Tuple, TypeVar

import imageio
import matplotlib.pyplot as plt
import pddlgym
from pddlgym.core import PDDLEnv

from src.types import State, Action, StateRepresentation, ActionTriplet

AT = TypeVar("AT", bound=ActionTriplet)
SR = TypeVar("SR", bound=StateRepresentation)
class BaselineImageTrajectoryHandler:


    def __init__(self, action_steps=100):
        """
        This is a class for the baseline image trajectory handler that creates a
        trajectory at random from a given `pddlgym.PDDLGym` environment.

        Attributes:
            - action_steps (int): number of actions to take when generating the trajectory
            - problem_index (int): index of the problem in the to-be-given domain, so we can
            use this to reproduce the trajectory for a given domain if needed.
        Methods:
            - get_image_trajectory(self, path: str) -> List[Image]: returns a trajectory of images
              listed in the provided path.
        """
        super().__init__()
        self.action_steps = action_steps # TODO LATER: decide if this one should be determined per trajectory creation, instead of to be fixed to all trajectories

    def create_image_trajectory(self, gymenv_name: str, problem_index: int = None) -> Tuple[Sequence[State], Sequence[Action]]:
        env: PDDLEnv = pddlgym.make(gymenv_name)
        if problem_index:
            env.fix_problem_index(problem_index)

        obs, info = env.reset()

        actions: List[Action] = []
        gold_states: List[State] = [obs]

        # TODO:  might need an enumeration method for all the grounded states and actions.

        for step in range(self.action_steps):
            # TODO: see how to save the image as well in a sequence to be returned from this method
            imageio.mimsave(f"../data/{gymenv_name}/{step}.png", env.render())
            plt.close()
            gold_states.append(obs)

            action = env.action_space.sample(obs)
            actions.append(action.pddl_str())

            obs = env.step(action)[0]

        return gold_states, actions

    @staticmethod
    def build_trajectory(state_list: Sequence[SR], action_list: Sequence[Action]) -> Sequence[AT]:
        """
        Builds a trajectory of ActionTriplet objects of size `len(action_list)`
        :param state_list: the list of state representations. must satisfy `len(state_list) == len(action_list)+1`
        :param action_list: the list of actions transitioning between states in state_list such that action_list[i] is
        the action transitioning between state_list[i] and state_list[i+1]
        :return: a sequence of ActionTriplet objects, representing the proper triplets ordered sequence.
        """

        assert len(state_list) == len(action_list)+1

        return [ActionTriplet(state_list[i], action_list[i], state_list[i+1]) for i in range(len(action_list))]

    @staticmethod
    def split_trajectory(triplets_list: Sequence[ActionTriplet]) -> Tuple[Sequence[StateRepresentation], Sequence[Action]]:
        assert len (triplets_list) > 0

        states: Sequence[StateRepresentation] = [triplets_list[0].prev_state] + \
                                                [triplet.next_state for triplet in triplets_list]

        actions: Sequence[Action] = [triplet.action for triplet in triplets_list]

        return states, actions