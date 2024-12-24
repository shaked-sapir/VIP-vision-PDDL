import json
from pathlib import Path
from typing import Dict, List

import cv2

import pddlgym
from pddlgym.rendering.blocks import _block_name_to_color

from src.action_model.GYM2SAM_parser import create_observation_from_trajectory
from src.fluent_classification.colors import NormalizedRGB
from src.fluent_classification.contours import get_image_predicates
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser
from pddl_plus_parser.models import (
    Observation,
    Domain, Problem
)

from src.trajectory_handlers import ImageTrajectoryHandler

BLOCKS_DOMAIN_FILE_PATH = Path("blocks.pddl")
BLOCKS_PROBLEM_DIR_PATH = Path("problems")
BLOCKS_OUTPUT_DIR_PATH = Path("blocks_images")

NEGATION_PREFIX = "Not"


def parse_image_predicate_to_gym(predicate_str: str, is_holding_in_image: bool) -> str:
    """
    Parse a predicate extracted from an image to pddlgym format.
    :param predicate_str: the string representing the predicate, e.g. `holding(e:block)`
    :param is_holding_in_image: whether the predicate holds in the image it was extracted from
    :return: updated string representing the predicate
    """
    return predicate_str if is_holding_in_image else f"{NEGATION_PREFIX}{predicate_str}"


def is_positive_gym_predicate(predicate_str: str) -> bool:
    return NEGATION_PREFIX not in predicate_str


def create_imaged_trajectory(images_path: Path, ground_actions: List[str], object_name_to_color: Dict[str, str]):
    imaged_trajectory = []
    for i, action in enumerate(ground_actions):
        current_state_image = cv2.imread(f"{images_path}/state_{i:04d}.png")  # in BGR format
        current_state_image_predicates = get_image_predicates(current_state_image, object_name_to_color)
        current_state_image_pddl_predicates: List[str] = [parse_image_predicate_to_gym(pred, holds_in_image) for
                                                          pred, holds_in_image in
                                                          current_state_image_predicates.items()]

        next_state_image = cv2.imread(f"{images_path}/state_{i + 1:04d}.png")  # in BGR format
        next_state_image_predicates = get_image_predicates(next_state_image, object_name_to_color)
        next_state_image_pddl_predicates: List[str] = [parse_image_predicate_to_gym(pred, holds_in_image) for
                                                       pred, holds_in_image in
                                                       next_state_image_predicates.items()]
        imaged_trajectory.append({
            "step": i + 1,
            "current_state": {
                "literals": [pred for pred in current_state_image_pddl_predicates if is_positive_gym_predicate(pred)]
            },
            "ground_action": action,
            "next_state": {
                "literals": [pred for pred in next_state_image_pddl_predicates if is_positive_gym_predicate(pred)]
            },
        })
    return imaged_trajectory



# TODO: make this a class TrajectoryHandler. inject: necessary paths, domain, problems.. whatever necessarry.
def alg(domain_name: str, num_steps: int, output_dir: Path, problem_name: str):
    """
    This the main workflow of predicates classification within an image in the blocks world.
    :param domain_name: the name of the domain for the environment
    :param num_steps: the number of steps (actions) we want to perform in the environment (length of the trajectory)
    :param output_dir: the output directory name we want to save the images to
    :param problem: the problem instance we desire to make trajectory from #TODO: find a way to make the problem injectable
    :return:
    """

    print("alg started")

    """
    extracting colors of objects from trajectory so we can detect the objects in the image
    """
    # TODO:  the mapping is generated at problem initialization, so it has to be returned from the trajectory making process
    object_name_to_color: Dict[str, NormalizedRGB] = {
        **{str(obj): color for obj, color in _block_name_to_color.items()},
        "robot:robot": (0.4, 0.4, 0.4),
        "table:table": (0.5, 0.2, 0.0)
    }

    with open(f"{BLOCKS_OUTPUT_DIR_PATH}/trajectory.json", 'r') as file:
        GT_trajectory = json.load(file)

    print(f"Object name to color map: {object_name_to_color}")

    pddl_plus_blocks_domain: Domain = DomainParser(BLOCKS_DOMAIN_FILE_PATH).parse_domain()
    pddl_plus_blocks_problem: Problem = ProblemParser(Path(f"{BLOCKS_PROBLEM_DIR_PATH}/{problem_name}"),
                                                      pddl_plus_blocks_domain).parse_problem()
    GT_observation: Observation = create_observation_from_trajectory(GT_trajectory, pddl_plus_blocks_domain,
                                                                     pddl_plus_blocks_problem)

    # Output the resulting Observation object
    print("printing GT observation:")
    for component in GT_observation.components:
        print(str(component))

    print("*****************************")

    grounded_actions = [step["ground_action"] for step in GT_trajectory]
    imaged_trajectory_info = create_imaged_trajectory(output_dir, grounded_actions, object_name_to_color)
    imaged_observation: Observation = create_observation_from_trajectory(imaged_trajectory_info,
                                                                         pddl_plus_blocks_domain,
                                                                         pddl_plus_blocks_problem)

    # Output the resulting Observation object
    print("printing imaged observation:")
    for component in imaged_observation.components:
        print(str(component))

    print("*****************************")


if __name__ == "__main__":
    pddl_blocks_env = pddlgym.make("PDDLEnvBlocks-v0")
    trajectory_handler = ImageTrajectoryHandler(pddl_blocks_env)
    trajectory_handler.create_image_trajectory(problem_name='problem9.pddl',
                                               trajectory_output_dir=BLOCKS_OUTPUT_DIR_PATH,
                                               num_steps=15)

    # TODO TOMORROW: handle the part of the prediates after the image trajectyory saving.
    alg(domain_name="PDDLEnvBlocks-v0", num_steps=15, output_dir=BLOCKS_OUTPUT_DIR_PATH, problem_name='problem9.pddl')
