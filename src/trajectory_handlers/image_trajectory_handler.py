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
from src.utils.pddl import set_problem_by_name, parse_gym_to_pddl_ground_action, parse_gym_to_pddl_literal, \
    ground_action


class ImageTrajectoryHandler(ABC):
    #TODO IMM: limit the rendering if the num_steps exceeds the trajectory_size_limit
    # TODO: add documentation about the class for being a class responsible for making trajectories
    #       from gym domains

    object_detector: ObjectDetector
    fluent_classifier: FluentClassifier

    def __init__(self, domain_name: str, trajectory_size_limit: int = 1000, object_detector: ObjectDetector = None,
                 fluent_classifier: FluentClassifier = None):
        self.pddl_env = pddlgym.make(domain_name)

        size_limit_bit_count = math.ceil(math.log10(trajectory_size_limit))
        self.seq_idx_format = f'0{size_limit_bit_count+1}d'
        self.trajectory_size_limit = trajectory_size_limit
        self.object_detector = object_detector
        self.fluent_classifier = fluent_classifier

    def create_image(self, image_output_dir: Path, image_sequential_idx: int):
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

    def _create_trajectory_step(self, curr_obs: State, action: Literal, action_index: int, next_obs: State) -> TrajectoryStep:
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

    def image_trajectory_pipeline(self, domain_name: str, problem_name: str,
                                   num_steps: int = 100, output_dir: Path = None) -> None:
        GT_trajectory, actions, trajectory_path = self.create_trajectory_from_gym(domain_name, problem_name, num_steps, output_dir)
        self._init_visual_components()
        imaged_trajectory = self.construct_trajectory_from_images(fluent_classifier=None, images_path=trajectory_path, ground_actions=actions)
        self.build_trajectory_file(imaged_trajectory)
        return imaged_trajectory, trajectory_path

    #  TODO: putting it here is a hack because the color detector has to be initialized only after deciding on the problem
    #  to be solved, potentially we need to initialize it in the constructor
    def _init_visual_components(self) -> None:
        """
        Initialize the object detector and fluent classifier components.
        if needed - initialize more essential components, and this should be
        specific for each domain.
        :return:  None
        """
        pass

    def create_trajectory_from_gym(self, domain_name: str, problem_name: str,
                                   num_steps: int = 100, output_dir: Path = None) -> Tuple[List[TrajectoryStep], List[str], Path]:

        # TODO IMM: this function should also generate the .trajectory file in pddl format, from which we are going to
        # load the trajectory using SAm_LEARNING parser. we do have to, though, keep the gt format for later comparisons if needed.
        # this one is instead of reconstructing Observations from this GT by force.. redundant.

        # TODO: fix documentation when refactoring is done
        """
        This method creates a trajectory of randomly-taken actions within a pddlgym environment, using a specific
        problem, and does the following:
        1. saves the trajectory of the problem to the trajectory in verbose format, to serve as GT
        2. saves the image sequence of the trajectory to the specified directory

        :param problem_name: The problem for which to create the trajectory
        :param trajectory_output_dir: the output directory for the trajectory images
        :param num_steps: the number of steps (actions) to take when generating the trajectory.
        :param output_dir: the directory to put all created resources of this method: image stream and GT_trajectory
        :return: (None)
        """
        if num_steps > self.trajectory_size_limit:
            raise ValueError(f"cannot have more than {self.trajectory_size_limit} steps!")

        set_problem_by_name(self.pddl_env, problem_name)
        obs, info = self.pddl_env.reset()
        output_dir = output_dir if output_dir else Path(f"{domain_name}_{problem_name}")
        os.makedirs(output_dir, exist_ok=True)
        self.create_image(output_dir, 0)

        trajectory_log_file_path = os.path.join(output_dir, "trajectory.json")

        GT_trajectory = []
        actions = []
        new_obs = obs

        for i in range(1, num_steps + 1):
            obs = new_obs

            # Sample a random valid action (action affecting the state) from the set of valid actions
            while new_obs == obs:
                action = self.pddl_env.action_space.sample(obs)
                new_obs, _, done, _, _ = self.pddl_env.step(action)

            self.create_image(output_dir, i)
            state_action_entry = self._create_trajectory_step(curr_obs=obs,
                                                              action=action,
                                                              action_index=i,
                                                              next_obs=new_obs)
            GT_trajectory.append(state_action_entry)
            actions.append(action)
            if done:
                break

        GT_trajectory = serialize(GT_trajectory)
        # Save the states and actions to the log file
        with open(trajectory_log_file_path, 'w') as log_file:
            json.dump(GT_trajectory, log_file, indent=4)

        print(f"Images saved to the directory '{output_dir}'")
        print(f"Trajectory log saved to '{trajectory_log_file_path}'")

        return GT_trajectory, actions, output_dir

    def construct_trajectory_from_images(self, fluent_classifier,
                                         images_path: Path, ground_actions: List[str], action_model=None) -> List[dict]:
        imaged_trajectory = []
        for i, action in enumerate(ground_actions):
            current_state_image = self.load_image(images_path, i)
            current_state_image_predicates = fluent_classifier.classify(current_state_image)
            current_state_image_pddl_predicates: List[str] = [parse_image_predicate_to_gym(pred, holds_in_image) for
                                                              pred, holds_in_image in
                                                              current_state_image_predicates.items()]

            next_state_image = self.load_image(images_path, i + 1)
            next_state_image_predicates = fluent_classifier.classify(next_state_image)
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

    #  TODO: move this to `utils.pddl` file
    def build_trajectory_file(self, trajectory: List[dict], output_path="trajectory.trajectory") -> None:
        trajectory_lines = ["("]  # the opener of the trajectory file

        # Step 0: Write the initial state from current_state of the first entry
        init_literals = trajectory[0]['current_state']['literals']
        init_literals_parsed = ' '.join(parse_gym_to_pddl_literal(lit) for lit in init_literals)
        trajectory_lines.append(f"(:init {init_literals_parsed})")

        # Step 1: Write the first operator (from first entry)
        ground_action = trajectory[0]['ground_action']
        ground_action_parsed = parse_gym_to_pddl_ground_action(ground_action)
        trajectory_lines.append(f"(operator: {ground_action_parsed})")

        # Then continue: For each NEXT state and NEXT action
        for i in range(1, len(trajectory)):
            step_info = trajectory[i]

            # Write the state
            current_literals = step_info['current_state']['literals']
            current_literals_parsed = ' '.join(parse_gym_to_pddl_literal(lit) for lit in current_literals)
            trajectory_lines.append(f"(:state {current_literals_parsed})")

            # Write the operator
            ground_action = step_info['ground_action']
            ground_action_parsed = parse_gym_to_pddl_ground_action(ground_action)
            trajectory_lines.append(f"(operator: {ground_action_parsed})")

        # Finally write the last :state after the last action
        final_state_literals = trajectory[-1]['next_state']['literals']
        final_state_literals_parsed = ' '.join(parse_gym_to_pddl_literal(lit) for lit in final_state_literals)
        trajectory_lines.append(f"(:state {final_state_literals_parsed})")

        trajectory_lines.append(")")  # the closer of the trajectory file

        # Save to file
        with open(output_path, "w") as f:
            f.write('\n'.join(trajectory_lines))

        print(f"Trajectory saved to {output_path}")
