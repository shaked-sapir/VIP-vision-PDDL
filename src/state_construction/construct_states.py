from pathlib import Path
from typing import List

import cv2

from src.action_model.pddl2gym_parser import parse_image_predicate_to_gym, is_positive_gym_predicate
from src.trajectory_handlers import ImageTrajectoryHandler


def construct_states_from_images(
        image_trajectory_handler: ImageTrajectoryHandler,
        fluent_classifier, images_path: Path,
        ground_actions: List[str],
        action_model=None
):
    imaged_trajectory = []
    for i, action in enumerate(ground_actions):
        current_state_image = image_trajectory_handler.load_image(images_path, i)
        current_state_image_predicates = fluent_classifier.classify(current_state_image)
        current_state_image_pddl_predicates: List[str] = [parse_image_predicate_to_gym(pred, holds_in_image) for
                                                          pred, holds_in_image in
                                                          current_state_image_predicates.items()]

        next_state_image = image_trajectory_handler.load_image(images_path, i+1)
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
