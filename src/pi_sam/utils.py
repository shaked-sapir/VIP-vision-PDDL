from pathlib import Path
from typing import Union, Optional

from pddl_plus_parser.models import GroundedPredicate, Domain

from src.action_model.gym2SAM_parser import lift_predicate
from src.action_model.pddl2gym_parser import NEGATION_PREFIX, UNKNOWN_PREFIX, pddlplus_to_gym_predicate


def save_masking_info(experiment_path: Path, problem_name: str, trajectory_masking_info: list[set[GroundedPredicate]]) -> None:
    with open(experiment_path / f'{problem_name}.masking_info', 'w') as f:
        for state_masking in trajectory_masking_info:
            predicates_str = ', '.join([str(pred) for pred in state_masking])
            f.write(f"{predicates_str}\n")


def load_masking_info(
    masking_file_path: Union[Path, str],
    domain: Domain,
) -> list[set[GroundedPredicate]]:
    """
    Load masking info from a .masking_info file.

    Can be called in two ways:
    1. load_masking_info(experiment_dir, domain, problem_name) - constructs file path
    2. load_masking_info(masking_file_path, domain) - uses direct file path

    :param masking_file_path: Either the experiment directory or direct path to .masking_info file
    :param domain: The PDDL domain for predicate parsing
    :return: List of masking info (set of grounded predicates per state)
    """
    assert masking_file_path.suffix == '.masking_info', "The masking file must have a .masking_info extension."

    loaded_trajectory_masking_info: list[set[GroundedPredicate]] = []
    with open(masking_file_path, 'r') as f:
        for line in f:
            predicates_strs = [] if line.strip() == '' else line.strip().split(', ') # handle empty lines properly
            state_masking = set()
            for pred_str in predicates_strs:
                gym_format_pred = pddlplus_to_gym_predicate(pred_str)
                predicate_name, lifted_predicate_signature, predicate_object_mapping = lift_predicate(gym_format_pred, domain)
                grounded_pred = GroundedPredicate(
                    name=predicate_name,
                    signature=lifted_predicate_signature,
                    object_mapping=predicate_object_mapping,
                    is_positive=NEGATION_PREFIX not in pred_str,
                    is_masked=True
                )
                state_masking.add(grounded_pred)
            loaded_trajectory_masking_info.append(state_masking)

    return loaded_trajectory_masking_info
