import json
from pathlib import Path
from typing import Dict, List

import cv2

import pddlgym
from pddlgym.rendering.blocks import _block_name_to_color

from src.action_model.GYM2SAM_parser import create_observation_from_trajectory
from src.fluent_classification.blocks_fluent_classifier import BlocksFluentClassifier
from src.fluent_classification.colors import NormalizedRGB
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser
from pddl_plus_parser.models import (
    Observation,
    Domain, Problem
)

from src.object_detection import ColorObjectDetector
from src.object_detection.color_object_detector import ObjectName
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


# TODO IMM: actually this is the `construct states` algorithm - and this one should
"""
so this function has to also get the current action model as a parameter, so we can use it to add effects 
(at first) of an action to a `next_state` and  preconditions (later, maybe) of an action to a `previous_state`.

notice: we may need to have access to a running SAM algorithm beforehand so for the time being the current 
situation might be sufficient. 
"""
def construct_states_from_images(fluent_classifier, images_path: Path, ground_actions: List[str]):
    imaged_trajectory = []
    for i, action in enumerate(ground_actions):
        current_state_image = cv2.imread(f"{images_path}/state_{i:04d}.png")  # in BGR format
        current_state_image_predicates = fluent_classifier.classify(current_state_image)
        current_state_image_pddl_predicates: List[str] = [parse_image_predicate_to_gym(pred, holds_in_image) for
                                                          pred, holds_in_image in
                                                          current_state_image_predicates.items()]

        next_state_image = cv2.imread(f"{images_path}/state_{i + 1:04d}.png")  # in BGR format
        next_state_image_predicates = fluent_classifier.classify(next_state_image)
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



def alg(output_dir: Path, problem_name: str, fluent_classifier):

    print("alg started")



    with open(f"{BLOCKS_OUTPUT_DIR_PATH}/trajectory.json", 'r') as file:
        GT_trajectory = json.load(file)

    print(f"Object name to color map: {fluent_classifier.object_detector.object_color_map}")

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
    imaged_trajectory_info = construct_states_from_images(fluent_classifier, output_dir, grounded_actions)
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
    """
        extracting colors of objects from trajectory so we can detect the objects in the image
        """
    # TODO:  the mapping is generated at problem initialization, so it has to be returned from the trajectory making process
    object_name_to_color: Dict[ObjectName, NormalizedRGB] = {
        **{ObjectName(str(obj)): color for obj, color in _block_name_to_color.items()},
        ObjectName("robot:robot"): (0.4, 0.4, 0.4),
        ObjectName("table:table"): (0.5, 0.2, 0.0)
    }

    blocks_object_detector = ColorObjectDetector(object_name_to_color)
    blocks_fluent_classifier = BlocksFluentClassifier(blocks_object_detector)

    # TODO TOMORROW: handle the part of the prediates after the image trajectyory saving.
    alg(output_dir=BLOCKS_OUTPUT_DIR_PATH, problem_name='problem9.pddl', fluent_classifier=blocks_fluent_classifier)
