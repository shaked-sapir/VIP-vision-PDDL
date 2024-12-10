import json
import os
from pathlib import Path
from typing import Tuple, Dict, List

import cv2
from PIL import Image

import pddlgym
from pddlgym.core import _select_operator, PDDLEnv
from pddlgym.rendering.blocks import _block_name_to_color

from src.action_model.GYM2SAM_parser import create_observation_from_trajectory
from src.fluent_classification.colors import NormalizedRGB
from src.fluent_classification.contours import get_image_predicates
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser
from pddl_plus_parser.models import (
    State,
    GroundedPredicate,
    Observation,
    ActionCall,
    Domain, Problem, PDDLObject, SignatureType
)

def parse_image_predicate_to_gym(predicate_str: str, is_holding_in_image: bool) -> str:
    """
    Parse a predicate extracted from an image to pddlgym format.
    :param predicate_str: the string representing the predicate, e.g. `holding(e:block)`
    :param is_holding_in_image: whether the predicate holds in the image it was extracted from
    :return: updated string representing the predicate
    """
    return predicate_str if is_holding_in_image else f"Not{predicate_str}"


def is_positive_gym_predicate(predicate_str: str) -> bool:
    return 'Not' not in predicate_str


def create_imaged_trajectory_info(ground_actions: List[str], object_name_to_color: Dict[str, str]):
    imaged_trajectory_info = []
    for i, action in enumerate(ground_actions):

        current_state_image = cv2.imread(f"blocks_images/state_{i:04d}.png")  # in BGR format
        current_state_image_predicates = get_image_predicates(current_state_image, object_name_to_color)
        current_state_image_pddl_predicates: List[str] = [parse_image_predicate_to_gym(pred, holds_in_image) for pred, holds_in_image in
                                      current_state_image_predicates.items()]

        next_state_image = cv2.imread(f"blocks_images/state_{i+1:04d}.png")  # in BGR format
        next_state_image_predicates = get_image_predicates(next_state_image, object_name_to_color)
        next_state_image_pddl_predicates: List[str] = [parse_image_predicate_to_gym(pred, holds_in_image) for
                                                          pred, holds_in_image in
                                                          next_state_image_predicates.items()]
        imaged_trajectory_info.append({
            "step": i+1,
            "current_state": {
                "literals": [pred for pred in current_state_image_pddl_predicates if is_positive_gym_predicate(pred)]
            },
            "ground_action": action,
            "next_state": {
                "literals": [pred for pred in next_state_image_pddl_predicates if is_positive_gym_predicate(pred)]
            },
        })
    return imaged_trajectory_info


BLOCKS_DOMAIN_FILE_PATH = Path("blocks.pddl")
BLOCKS_PROBLEM_DIR_PATH = Path("problems")


def set_problem_by_name(pddl_env: PDDLEnv, problem_name: str):
    # Ensure the problem name ends with '.pddl'
    if not problem_name.endswith('.pddl'):
        problem_name += '.pddl'
    # Find the index of the problem with the given name
    for idx, prob in enumerate(pddl_env.problems):
        if prob.problem_fname.endswith(problem_name):
            pddl_env.fix_problem_index(idx)
            return
    raise ValueError(f"Problem '{problem_name}' not found in the problem list of the environment you provided.")

def alg(num_steps: int, output_dir: Path, problem_name: str):
    """
    This the main workflow of predicates classification within an image in the blocks world.
    :param num_steps: the number of steps we want to have for the trajectory\
    :param output_dir: the output directory name we want to save the images to
    :param problem: the problem instance we desire to make trajectory from #TODO: find a way to make the problem injectable
    :return:
    """

    print("alg started")
    env = pddlgym.make("PDDLEnvBlocks-v0")
    set_problem_by_name(env, problem_name)

    obs, info = env.reset()

    new_obs = obs

    # Create a directory to save images if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a file to save states and actions
    log_file_path = os.path.join(output_dir, "states_and_actions.json")
    states_and_actions = []

    # Render the environment and save the image
    img = env.render(mode='rgb_array')
    img_pil = Image.fromarray(img)
    img_pil.save(os.path.join(output_dir, f"state_{0:04d}.png"))

    # Run 1000 random moves and save the images
    for i in range(1, num_steps+1):
        obs = new_obs
        # Sample a random valid action from the set of valid actions

        while new_obs == obs:
            action = env.action_space.sample(obs)
            new_obs, _, done, _, _ = env.step(action)

        # Record the state and action
        state_action_entry = {
            "step": i,
            "current_state": {
                "literals": [str(literal) for literal in obs.literals],
                "objects": [str(obj) for obj in obs.objects],
                "goal": [str(literal) for literal in obs.goal.literals]
            },
            "ground_action": str(action),

            #TODO later: the _select_operator seems to make it a "safe" action, but the blocksworld is not a domain prone to unsafety - discuss with Roni
            "operator_object_assignment": _select_operator(obs, action, env.domain)[1],
            "lifted_preconds": str(env.domain.operators['pick-up'].preconds.literals),
            "next_state": {
                "literals": [str(literal) for literal in new_obs.literals],
                "objects": [str(obj) for obj in new_obs.objects],
                "goal": [str(literal) for literal in new_obs.goal.literals]
            }
        }

        states_and_actions.append(state_action_entry)

        # Render the environment and save the image
        img = env.render()
        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(output_dir, f"state_{i:04d}.png"))

        if done:
            break

    # Save the states and actions to the log file
    with open(log_file_path, 'w') as log_file:
        json.dump(states_and_actions, log_file, indent=4)


    print(f"Images saved to the directory '{output_dir}'")
    print(f"States and actions log saved to '{log_file_path}'")

    """
    extracting colors of objects from trajectory so we can detect the objects in the image
    """
    # extract colors:  the mapping is generated at problem initialization
    object_name_to_color: Dict[str, NormalizedRGB] = {
        **{str(obj): color for obj, color in _block_name_to_color.items()},
        "robot:robot": (0.4, 0.4, 0.4),
        "table:table": (0.5, 0.2, 0.0)
    }

    print(f"Object name to color map: {object_name_to_color}")

    pddl_plus_blocks_domain: Domain = DomainParser(BLOCKS_DOMAIN_FILE_PATH).parse_domain()
    pddl_plus_blocks_problem: Problem = ProblemParser(Path(f"{BLOCKS_PROBLEM_DIR_PATH}/problem9.pddl"),
                                                      pddl_plus_blocks_domain).parse_problem()
    GT_observation: Observation = create_observation_from_trajectory(states_and_actions, pddl_plus_blocks_domain,
                                                                  pddl_plus_blocks_problem)

    # Output the resulting Observation object
    print("printing GT observation:")
    for component in GT_observation.components:
        print(str(component))

    print("*****************************")

    grounded_actions = [step["ground_action"] for step in states_and_actions]
    imaged_trajectory_info = create_imaged_trajectory_info(grounded_actions, object_name_to_color)
    imaged_observation: Observation = create_observation_from_trajectory(imaged_trajectory_info, pddl_plus_blocks_domain, pddl_plus_blocks_problem)

    # Output the resulting Observation object
    print("printing imaged observation:")
    for component in imaged_observation.components:
        print(str(component))

    print("*****************************")


if __name__ == "__main__":
    alg(num_steps=15, output_dir=Path("blocks_images"), problem_name='problem3')
