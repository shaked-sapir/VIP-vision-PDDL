import json
import math
import os
from pathlib import Path
from typing import List, Tuple
from abc import ABC, abstractmethod
import cv2
import pddlgym
from PIL import Image
from pddlgym.core import _select_operator
from pddlgym.structs import State, Literal

from src.action_model.pddl2gym_parser import parse_image_predicate_to_gym, is_positive_gym_predicate
from src.fluent_classification.base_fluent_classifier import FluentClassifier
from src.object_detection.base_object_detector import ObjectDetector
from src.types import TrajectoryState, TrajectoryStep
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
        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(image_output_dir, f"state_{image_sequential_idx:{self.seq_idx_format}}.png"))
        return

    def load_image(self, image_output_dir: Path, image_sequential_index: int) -> cv2.typing.MatLike:
        return cv2.imread(os.path.join(image_output_dir, f"state_{image_sequential_index:{self.seq_idx_format}}.png"))

    @staticmethod
    def _create_trajectory_state(obs: State) -> TrajectoryState:
        return TrajectoryState(
            literals=[str(literal) for literal in obs.literals],
            objects=[str(obj) for obj in obs.objects],
            goal=[str(literal) for literal in obs.goal.literals]
        )

    def _create_trajectory_step(self, curr_obs: State, action: Literal, action_index: int,
                                next_obs: State) -> TrajectoryStep:
        selected_operator, assignment = _select_operator(curr_obs, action, self.pddl_env.domain)
        grounded_action = ground_action(selected_operator, assignment)
        return TrajectoryStep(
            step=action_index,
            current_state=self._create_trajectory_state(curr_obs),
            ground_action=grounded_action,

            # TODO later: the _select_operator seems to make it a "safe" action,
            #  but the blocksworld is not a domain prone to unsafety - discuss with Roni
            operator_object_assignment=assignment,
            lifted_preconds=str(self.pddl_env.domain.operators[selected_operator.name].preconds.literals),
            next_state=self._create_trajectory_state(next_obs)
        )

    def image_trajectory_pipeline(self, problem_name: str, output_path: Path, num_steps: int = 100) -> None:
        """
        runs the pipeline of creating an imaged trajectory for a given domain and problem:
        1. initializes the specific problem in the domain and runs a number of random steps to
        generate a sequence of states, recording the images representing them
        2. initializes the visual components - object detectors, fluent classifiers, etc
        3. creates a belief representation of the states from the images (imaged trajectory)
        4. creates a .trajectory file from the belief representation, that could be fed into planning
        algorithms

        :param problem_name: name of specific problem to generate trajectory for
        :param output_path: the path for storing the method outcomes like trajectory images, GT_trajectory info, etc.
        :param num_steps: number of random actions to be taken for the trajectory
        :return: None
        """
        actions = self.create_trajectory_from_gym(problem_name, output_path, num_steps)

        # TODO: putting it here is a hack because the color detector has to be initialized only after deciding on
        #  the problem to be solved, potentially we need to initialize it in the constructor
        self._init_visual_components()
        imaged_trajectory = self.construct_trajectory_from_images(images_path=output_path, ground_actions=actions)
        build_trajectory_file(imaged_trajectory, output_path)
        return

    @abstractmethod
    def _init_visual_components(self) -> None:
        """
        Initialize the object detector and fluent classifier components.
        if needed - initialize more essential components, and this should be
        specific for each domain.
        :return:  None
        """
        pass

    def create_trajectory_from_gym(self, problem_name: str, output_path: Path,
                                   num_steps: int = 100) -> List[str]:

        """
        This method creates a trajectory of randomly-taken actions within a pddlgym environment, using a specific
        problem, and does the following:
        1. saves the trajectory of the problem to the trajectory in verbose format, to serve as GT
        2. saves the image sequence of the trajectory to the specified directory

        :param problem_name: name of specific problem to generate trajectory for
        :param output_path: the path for storing the method outcomes like trajectory images, GT_trajectory info, etc.
        :param num_steps: number of random actions to be taken for the trajectory
        :return: action sequence of the trajectory
        """
        if num_steps > self.trajectory_size_limit:
            raise ValueError(f"cannot have more than {self.trajectory_size_limit} steps!")

        set_problem_by_name(self.pddl_env, problem_name)
        obs, info = self.pddl_env.reset()

        os.makedirs(output_path, exist_ok=True)
        trajectory_log_file_path = os.path.join(output_path, "trajectory.json")

        GT_trajectory: list[TrajectoryStep] = []
        ground_actions: list[str] = []
        new_obs = obs
        self.create_image(output_path, 0)

        for i in range(1, num_steps + 1):
            obs = new_obs

            # Sample a random valid action (action affecting the state) from the set of valid actions
            while new_obs == obs:
                action = self.pddl_env.action_space.sample(obs)
                new_obs, _, done, _, _ = self.pddl_env.step(action)

            self.create_image(output_path, i)
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

        print(f"Images saved to the directory '{output_path}'")
        print(f"Trajectory log saved to '{trajectory_log_file_path}'")

        return ground_actions

    def construct_trajectory_from_images(self,
                                         images_path: Path, ground_actions: List[str], action_model=None) -> List[dict]:
        imaged_trajectory = []
        for i, action in enumerate(ground_actions):
            current_state_image = self.load_image(images_path, i)
            current_state_image_predicates = self.fluent_classifier.classify(current_state_image)
            current_state_image_pddl_predicates: List[str] = [parse_image_predicate_to_gym(pred, holds_in_image) for
                                                              pred, holds_in_image in
                                                              current_state_image_predicates.items()]

            next_state_image = self.load_image(images_path, i + 1)
            next_state_image_predicates = self.fluent_classifier.classify(next_state_image)
            next_state_image_pddl_predicates: List[str] = [parse_image_predicate_to_gym(pred, holds_in_image) for
                                                           pred, holds_in_image in
                                                           next_state_image_predicates.items()]
            # TODO: add this when the action model procedures are ready
            #  remove_effects(next_state_image_pddl_predicates, action, action_model)
            imaged_trajectory.append({
                "step": i + 1,
                "current_state": {
                    "literals": [pred for pred in current_state_image_pddl_predicates if
                                 is_positive_gym_predicate(pred)]
                },
                "ground_action": action,
                "next_state": {
                    "literals": [pred for pred in next_state_image_pddl_predicates if is_positive_gym_predicate(pred)]
                },
            })
        return imaged_trajectory
