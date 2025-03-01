import json
from pathlib import Path
from typing import Dict

import pddlgym
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser
from pddl_plus_parser.models import (
    Observation,
    Domain, Problem
)
from pddlgym.rendering.blocks import _block_name_to_color
from sam_learning.learners.sam_learning import SAMLearner
from utilities import NegativePreconditionPolicy
from src.action_model.gym2SAM_parser import create_observation_from_trajectory
from src.fluent_classification.blocks_fluent_classifier import BlocksFluentClassifier
from src.object_detection import ColorObjectDetector
from src.types import ObjectLabel
from src.state_construction.construct_states import construct_states_from_images
from src.trajectory_handlers import ImageTrajectoryHandler
from src.utils.visualize import NormalizedRGB

BLOCKS_DOMAIN_FILE_PATH = Path("blocks.pddl")
BLOCKS_PROBLEM_DIR_PATH = Path("problems")
BLOCKS_OUTPUT_DIR_PATH = Path("blocks_images")


if __name__ == "__main__":
    blocks_domain_name = "PDDLEnvBlocks-v0"
    blocks_problem_name = "problem9.pddl"

    image_trajectory_handler = ImageTrajectoryHandler(blocks_domain_name)
    image_trajectory_handler.create_image_trajectory(problem_name=blocks_problem_name,
                                                     trajectory_output_dir=BLOCKS_OUTPUT_DIR_PATH,
                                                     num_steps=100)
    """
        extracting colors of objects from trajectory so we can detect the objects in the image
        """
    # TODO:  the mapping is generated at problem initialization, so it has to be returned from the trajectory making process
    object_name_to_color: Dict[ObjectLabel, NormalizedRGB] = {
        **{ObjectLabel(str(obj)): color for obj, color in _block_name_to_color.items()},
        ObjectLabel("robot:robot"): (0.4, 0.4, 0.4),
        ObjectLabel("table:table"): (0.5, 0.2, 0.0)
    }

    blocks_object_detector = ColorObjectDetector(object_name_to_color)
    blocks_fluent_classifier = BlocksFluentClassifier(blocks_object_detector)

    print("alg started")

    with open(f"{BLOCKS_OUTPUT_DIR_PATH}/trajectory.json", 'r') as file:
        GT_trajectory = json.load(file)

    print(f"Object name to color map: {blocks_object_detector.object_color_map}")

    pddl_plus_blocks_domain: Domain = DomainParser(BLOCKS_DOMAIN_FILE_PATH).parse_domain()
    pddl_plus_blocks_problem: Problem = ProblemParser(Path(f"{BLOCKS_PROBLEM_DIR_PATH}/{blocks_problem_name}"),
                                                      pddl_plus_blocks_domain).parse_problem()
    GT_observation: Observation = create_observation_from_trajectory(GT_trajectory, pddl_plus_blocks_domain,
                                                                     pddl_plus_blocks_problem)

    # Output the resulting Observation object
    print("printing GT observation:")
    for component in GT_observation.components:
        print(str(component))

    print("*****************************")

    grounded_actions = [step["ground_action"] for step in GT_trajectory]
    imaged_trajectory_info = construct_states_from_images(image_trajectory_handler, blocks_fluent_classifier, BLOCKS_OUTPUT_DIR_PATH, grounded_actions)
    imaged_observation: Observation = create_observation_from_trajectory(imaged_trajectory_info,
                                                                         pddl_plus_blocks_domain,
                                                                         pddl_plus_blocks_problem)

    # Output the resulting Observation object
    print("printing imaged observation:")
    for component in imaged_observation.components:
        print(str(component))

    print("*****************************")

    sam_learner = SAMLearner(pddl_plus_blocks_domain, negative_preconditions_policy=NegativePreconditionPolicy.hard)
    # sam_learner = SAMLearner(pddl_plus_blocks_domain)

    partial_domain, report = sam_learner.learn_action_model([imaged_observation])
    print(partial_domain.to_pddl())
    print(report)