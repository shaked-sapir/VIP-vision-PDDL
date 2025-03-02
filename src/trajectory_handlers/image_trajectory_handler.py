import json
import math
import os
from pathlib import Path
from typing import Union

import cv2
import pddlgym
from PIL import Image
from pddlgym.core import _select_operator
from pddlgym.parser import Operator
from pddlgym.structs import State, Literal

from src.types import TrajectoryState, TrajectoryStep
from src.utils.containers import serialize
from src.utils.os import create_dir_from_root
from src.utils.pddl import set_problem_by_name


class ImageTrajectoryHandler:
    #TODO IMM: limit the rendering if the num_steps exceeds the trajectory_size_limit
    def __init__(self, domain_name: str, pddlgym_domain_name: str, trajectory_size_limit: int = 1000):
        self.domain_name = domain_name
        self.pddl_env = pddlgym.make(pddlgym_domain_name)
        size_limit_bit_count = math.ceil(math.log10(trajectory_size_limit))
        self.seq_idx_format = f'0{size_limit_bit_count+1}d'
        self.trajectory_size_limit = trajectory_size_limit
        self.trials_dir: str = os.path.join("src/trials", self.domain_name)
        create_dir_from_root(self.trials_dir)

    def create_image(self, image_output_dir: Union[Path, str], image_sequential_idx: int) -> None:
        """
        renders a single image from the current state of the environment and saves it.
        :param image_output_dir: directory to save the image to
        :param image_sequential_idx: the sequential index of the image in the trajectory, 0-indexed. e.g., 0 for the
                                     first image, 1 for the second image, etc.
        :return: (None)
        """
        img = self.pddl_env.render(mode='rgb_array')
        img_pil = Image.fromarray(img)
        img_path = os.path.join(image_output_dir, f"state_{image_sequential_idx:{self.seq_idx_format}}.png")
        img_pil.save(img_path)
        # with open(img_path, "rb") as f:
        #     os.fsync(f.fileno())  # Ensure immediate disk write

    def load_image(self, image_output_dir: Path, image_sequential_index: int) -> cv2.typing.MatLike:
        return cv2.imread(os.path.join(image_output_dir, f"state_{image_sequential_index:{self.seq_idx_format}}.png"))

    @staticmethod
    def _create_trajectory_state(obs: State) -> TrajectoryState:
        return TrajectoryState(
            literals=[str(literal) for literal in obs.literals],
            objects=[str(obj) for obj in obs.objects],
            goal=[str(literal) for literal in obs.goal.literals]
        )

    def _create_trajectory_step(self, curr_obs: State, action: Literal, action_index: int, next_obs: State) -> TrajectoryStep:
        selected_operator, assignment = _select_operator(curr_obs, action, self.pddl_env.domain)
        grounded_action = self._ground_action(selected_operator, assignment)
        return TrajectoryStep(
            step=action_index,
            current_state=self._create_trajectory_state(curr_obs),
            ground_action=grounded_action,

            # TODO later: the _select_operator seems to make it a "safe" action, but the blocksworld is not a domain prone to unsafety - discuss with Roni
            operator_object_assignment=assignment,
            lifted_preconds=str(self.pddl_env.domain.operators[selected_operator.name].preconds.literals),
            next_state=self._create_trajectory_state(next_obs)
        )

    def _ground_action(self, operator: Operator, assignment: dict) -> str:
        action_name: str = operator.name
        grounded_params = [str(assignment[lifted_param]) for lifted_param in operator.params]
        return f"{action_name}({', '.join(grounded_params)})"

    def create_image_trajectory(self, problem_name: str, num_steps: int = 100):
        """
        This method creates a trajectory of randomly-taken actions within a pddlgym environment, using a specific
        problem, and saves the trajectory and its images into a directory.

        :param problem_name: The problem for which to create the trajectory
        :param trajectory_output_dir: the output directory for the trajectory images
        :param num_steps: the number of steps (actions) to take when generating the trajectory.
        :return: (None)
        """
        if num_steps > self.trajectory_size_limit:
            raise ValueError(f"cannot have more than {self.trajectory_size_limit} steps!")

        set_problem_by_name(self.pddl_env, problem_name)
        obs, info = self.pddl_env.reset()

        # ATM, only a single trial is allowed for a certain problem. TODO LATER: allow multiple trials per problem
        # TODO NOW: extract folder creations to an ad-hoc function
        problem_trial_dir: Path = create_dir_from_root(os.path.join(self.trials_dir, problem_name.split(".pddl")[0]))
        problem_trial_image_dir: Path = create_dir_from_root(os.path.join(problem_trial_dir, "images"))
        # problem_trial_image_dir: str = os.path.join(problem_trial_dir, "images")

        # create_dir_from_root(problem_trial_dir)
        # create_dir_from_root(problem_trial_image_dir)

        # os.makedirs(problem_trial_dir, exist_ok=True)
        # os.makedirs(problem_trial_image_dir, exist_ok=True)
        self.create_image(problem_trial_image_dir, 0)

        trajectory_log_file_path = os.path.join(problem_trial_dir, "trajectory.json")

        trajectory = []
        new_obs = obs

        for i in range(1, num_steps + 1):
            obs = new_obs

            # Sample a random valid action (action affecting the state) from the set of valid actions
            while new_obs == obs:
                action = self.pddl_env.action_space.sample(obs)
                new_obs, _, done, _, _ = self.pddl_env.step(action)

            self.create_image(problem_trial_image_dir, i)
            state_action_entry = self._create_trajectory_step(curr_obs=obs,
                                                              action=action,
                                                              action_index=i,
                                                              next_obs=new_obs)
            trajectory.append(state_action_entry)

            if done:
                break

        # Save the states and actions to the log file
        with open(trajectory_log_file_path, 'w') as log_file:
            json.dump(serialize(trajectory), log_file, indent=4)

        print(f"Images saved to the directory '{problem_trial_image_dir}'")
        print(f"Trajectory log saved to '{trajectory_log_file_path}'")

    # def create_image_trajectory(self, problem_name: str, trajectory_output_dir: Path, num_steps: int = 100):
    #     """
    #     This method creates a trajectory of randomly-taken actions within a pddlgym environment, using a specific
    #     problem, and saves the trajectory and its images into a directory.
    #
    #     :param problem_name: The problem for which to create the trajectory
    #     :param trajectory_output_dir: the output directory for the trajectory images
    #     :param num_steps: the number of steps (actions) to take when generating the trajectory.
    #     :return: (None)
    #     """
    #     if num_steps > self.trajectory_size_limit:
    #         raise ValueError(f"cannot have more than {self.trajectory_size_limit} steps!")
    #
    #     set_problem_by_name(self.pddl_env, problem_name)
    #     obs, info = self.pddl_env.reset()
    #     self.create_image(trajectory_output_dir, 0)
    #
    #     os.makedirs(trajectory_output_dir, exist_ok=True)
    #     trajectory_log_file_path = os.path.join(trajectory_output_dir, "trajectory.json")
    #
    #     trajectory = []
    #     new_obs = obs
    #
    #     for i in range(1, num_steps + 1):
    #         obs = new_obs
    #
    #         # Sample a random valid action (action affecting the state) from the set of valid actions
    #         while new_obs == obs:
    #             action = self.pddl_env.action_space.sample(obs)
    #             new_obs, _, done, _, _ = self.pddl_env.step(action)
    #
    #         self.create_image(trajectory_output_dir, i)
    #         state_action_entry = self._create_trajectory_step(curr_obs=obs,
    #                                                           action=action,
    #                                                           action_index=i,
    #                                                           next_obs=new_obs)
    #         trajectory.append(state_action_entry)
    #
    #         if done:
    #             break
    #
    #     # Save the states and actions to the log file
    #     with open(trajectory_log_file_path, 'w') as log_file:
    #         json.dump(serialize(trajectory), log_file, indent=4)
    #
    #     print(f"Images saved to the directory '{trajectory_output_dir}'")
    #     print(f"Trajectory log saved to '{trajectory_log_file_path}'")
