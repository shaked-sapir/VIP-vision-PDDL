import json
import math
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import List

import cv2
import pddlgym
from PIL import Image
from matplotlib import pyplot as plt
from pddlgym.core import _select_operator
from gym.envs.registration import register
from pddlgym.structs import State, Literal

from src.action_model.pddl2gym_parser import parse_image_predicate_to_gym, is_positive_gym_predicate, \
    is_unknown_gym_predicate, get_predicate_base_form
from src.fluent_classification.base_fluent_classifier import FluentClassifier
from src.object_detection.base_object_detector import ObjectDetector
from src.typings import TrajectoryState, TrajectoryStep
from src.utils.containers import serialize
from src.utils.pddl import set_problem_by_name, ground_action, build_trajectory_file




class ImageTrajectoryHandler(ABC):
    """
    Abstract class for creating image trajectories.
    Each domain must have its own subclass and implement at least the "init_visual_resources" method, which
    takes care of initializing the components used for visual analysis of images composing the trajectories:
    object detectors, predicate classifiers, etc.
    """

    object_detector: ObjectDetector
    fluent_classifier: FluentClassifier

    def __init__(self, domain_name: str, trajectory_size_limit: int = 1000, object_detector: ObjectDetector = None,
                 fluent_classifier: FluentClassifier = None):
        self.domain_name = domain_name
        self.pddl_env = pddlgym.make(domain_name)

        size_limit_bit_count = math.ceil(math.log10(trajectory_size_limit))
        self.seq_idx_format = f'0{size_limit_bit_count + 1}d'
        self.trajectory_size_limit = trajectory_size_limit
        self.object_detector = object_detector
        self.fluent_classifier = fluent_classifier

    def create_image(self, image_output_dir: Path, image_sequential_idx: int) -> None:
        """
        renders a single image from the current state of the environment and saves it.
        :param image_output_dir: directory to save the image to
        :param image_sequential_idx: the sequential index of the image in the trajectory, 0-indexed. e.g., 0 for the
                                     first image, 1 for the second image, etc.
        :return: (None)
        """
        img = self.pddl_env.render(mode='rgb_array')
        plt.close('all')

        # normalize for float renderers
        if img.dtype != "uint8":
            img = (img * 255).clip(0, 255).astype("uint8")

        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(image_output_dir, f"state_{image_sequential_idx:{self.seq_idx_format}}.png"))

        return

    #  TODO: maybe use same approach (either PIL or cv2) for both create_image and load_image
    @staticmethod
    def get_image_full_path(image_dir: Path, image_name: str) -> str:
        return os.path.join(image_dir, image_name)

    def get_image_path_by_index(self, image_dir: Path, image_sequential_index: int) -> str:
        image_name: str = f"state_{image_sequential_index:{self.seq_idx_format}}.png"
        return self.get_image_full_path(image_dir, image_name)

    @staticmethod
    def _create_trajectory_state(obs: State) -> TrajectoryState:
        return TrajectoryState(
            literals=[str(literal) for literal in obs.literals],
            objects=[str(obj) for obj in obs.objects],
            goal=[str(literal) for literal in obs.goal.literals]
        )

    @staticmethod
    def _rename_ground_action(ground_action: str) -> str:
        # each child class can override this method to rename the ground action from pddlgym format to its own format
        return ground_action

    def _create_trajectory_step(self, curr_obs: State, action: Literal, action_index: int,
                                next_obs: State) -> TrajectoryStep:
        selected_operator, assignment = _select_operator(curr_obs, action, self.pddl_env.domain)
        grounded_action = ground_action(selected_operator, assignment)
        return TrajectoryStep(
            step=action_index,
            current_state=self._create_trajectory_state(curr_obs),
            ground_action=grounded_action,
            operator_object_assignment=assignment,
            lifted_preconds=str(self.pddl_env.domain.operators[selected_operator.name].preconds.literals),
            next_state=self._create_trajectory_state(next_obs)
        )

    def image_trajectory_pipeline(self, problem_name: str, actions: List[str], images_path: Path) -> List[dict]:
        """
        runs the pipeline of creating an imaged trajectory for a given domain and problem:
        1. if needed, initializes the visual components - object detectors, fluent classifiers, etc
        2. creates a belief representation of the states from the images (imaged trajectory)
        3. creates a .trajectory file from the belief representation, that could be fed into planning
        algorithms

        :param problem_name: name of specific problem to generate trajectory for
        :param actions: list of ground actions taken in the trajectory which is being represented by images
        :param images_path: the path for storing the method outcomes like trajectory images, GT_trajectory info, etc.
        :return: None
        """
        if not self.object_detector or not self.fluent_classifier:
            self.init_visual_components(images_path / "state_0000.png")

        imaged_trajectory = self.construct_trajectory_from_images(
            images_path=images_path, ground_actions=actions)
        build_trajectory_file(imaged_trajectory, problem_name, images_path)
        return imaged_trajectory

    @abstractmethod
    def init_visual_components(self, *args, **kwargs) -> None:
        """
        Initialize the object detector and fluent classifier components, if they are not provided in the constructor.
        if needed - initialize more essential components, and this should be
        specific for each domain.
        :return:  None
        """
        pass

    def create_trajectory_from_gym(self, problem_name: str, images_output_path: Path,
                                   num_steps: int = 100) -> List[str]:

        """
        This method creates a trajectory of randomly-taken actions within a pddlgym environment, using a specific
        problem, and does the following:
        1. saves the trajectory of the problem to the trajectory in verbose format, to serve as GT
        2. saves the image sequence of the trajectory to the specified directory

        :param problem_name: name of specific problem to generate trajectory for
        :param images_output_path: the path for storing the method outcomes like trajectory images, GT_trajectory info, etc.
        :param num_steps: number of random actions to be taken for the trajectory
        :return: action sequence of the trajectory
        """
        if num_steps > self.trajectory_size_limit:
            raise ValueError(f"cannot have more than {self.trajectory_size_limit} steps!")

        set_problem_by_name(self.pddl_env, problem_name)
        obs, info = self.pddl_env.reset()

        os.makedirs(images_output_path, exist_ok=True)
        trajectory_log_file_path = os.path.join(images_output_path, f"{problem_name}_trajectory.json")

        GT_trajectory: list[TrajectoryStep] = []
        ground_actions: list[str] = []
        new_obs = obs
        good_action = True
        self.create_image(images_output_path, 0)

        for i in range(1, num_steps + 1):
            obs = new_obs

            # Sample a random valid action (action affecting the state) from the set of valid actions
            while new_obs == obs or not good_action:
                try:
                    action = self.pddl_env.action_space.sample(obs)
                    env_temp = deepcopy(self.pddl_env)
                    new_obs, _, done, _, _ = env_temp.step(action)
                    _ = self.pddl_env.action_space.sample(new_obs) # will throw exception if got into invalid state
                    good_action = True
                    new_obs, _, done, _, _ = self.pddl_env.step(action)
                except Exception:
                    good_action = False
                    continue

            self.create_image(images_output_path, i)
            trajectory_step: TrajectoryStep = self._create_trajectory_step(curr_obs=obs,
                                                                              action=action,
                                                                              action_index=i,
                                                                              next_obs=new_obs)
            GT_trajectory.append(trajectory_step)
            ground_actions.append(trajectory_step.ground_action)
            if done:
                break

        GT_trajectory = serialize(GT_trajectory)
        # Save the states and actions to the log file
        with open(trajectory_log_file_path, 'w') as log_file:
            json.dump(GT_trajectory, log_file, indent=4)

        print(f"Images saved to the directory '{images_output_path}'")
        print(f"Trajectory log saved to '{trajectory_log_file_path}'")

        return ground_actions

    def construct_trajectory_from_images(self,
                                         images_path: Path, ground_actions: List[str], action_model=None
                                         ) -> List[dict]:
        imaged_trajectory = []

        # Process first state separately
        first_image_path = self.get_image_path_by_index(images_path, 0)
        current_state_predicates = self.fluent_classifier.classify(first_image_path)

        # Process each transition
        for i, action in enumerate(ground_actions):
            # Load next image and classify predicates
            next_image_path = self.get_image_path_by_index(images_path, i + 1)
            next_state_predicates = self.fluent_classifier.classify(next_image_path)

            # Convert predicates to PDDL format for trajectory
            current_literals = [parse_image_predicate_to_gym(pred, truth_value)
                                for pred, truth_value in current_state_predicates.items()]
            next_literals = [parse_image_predicate_to_gym(pred, truth_value)
                             for pred, truth_value in next_state_predicates.items()]

            action = self._rename_ground_action(action)

            # Build trajectory step
            trajectory_step = {
                "step": i + 1,
                "current_state": {
                    "literals": [pred for pred in current_literals if is_positive_gym_predicate(pred)],
                    "unknown": [pred for pred in current_literals if is_unknown_gym_predicate(pred)]
                },
                "ground_action": action,
                "next_state": {
                    "literals": [pred for pred in next_literals if is_positive_gym_predicate(pred)],
                    "unknown": [pred for pred in next_literals if is_unknown_gym_predicate(pred)]
                }
            }

            imaged_trajectory.append(trajectory_step)

            # Optimization: reuse next_state as current_state for next iteration
            current_state_predicates = next_state_predicates

        return imaged_trajectory
