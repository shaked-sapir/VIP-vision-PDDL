from pathlib import Path
from typing import Union

from pddl_plus_parser.lisp_parsers import TrajectoryParser
from pddl_plus_parser.models import GroundedPredicate, Domain, State, Observation

from src.action_model.gym2SAM_parser import lift_predicate
from src.action_model.pddl2gym_parser import NEGATION_PREFIX, pddlplus_to_gym_predicate
from src.utils.pddl import ground_observation_completely


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


def mask_state(state: State, masking_info: set[GroundedPredicate]) -> State:
    """
    Masks the predicates in the state according to the masking info provided.

    This utility function applies masking to a state by setting the is_masked flag
    on predicates specified in the masking_info.

    :param state: The state to mask predicates in.
    :param masking_info: A set of predicates to mask in the state.
    :return: A state with predicates masked according to the masking info provided.
    """
    for masked_pred in masking_info:
        # state.state_predicates are only positive- a positive version of the masked predicate is needed for access
        masked_pred_positive_form = masked_pred.copy(is_negated=not masked_pred.is_positive)
        all_matching_predicates = state.state_predicates[masked_pred_positive_form.lifted_untyped_representation]

        # TODO LATER: maybe refactor the pos/neg search, it is a bit clunky
        for pred in all_matching_predicates:
            if pred == masked_pred or pred.copy(is_negated=True) == masked_pred:
                pred.is_masked = True
                pred.is_positive = masked_pred.is_positive # ensure the sign is correct
                break
        # else:
            # print(f"Warning: Masked predicate {masked_pred} not found in state.")

    return state


def mask_observation(observation: Observation, masking_info: list[set[GroundedPredicate]]) -> Observation:
    """
    Masks the predicates in the observation for learning with partial information.

    This utility function applies masking to an observation using provided masking info.
    NOTE: Assumes observation is "full" - meaning that all predicates are grounded in the states.

    :param observation: The observation to mask predicates in (should be grounded).
    :param masking_info: A list of sets, where each set contains predicates to mask for each state.
                        Length should be len(observation.components) + 1.
    :return: An observation with predicates masked according to the masking info provided.
    """
    assert len(observation.components)+1 == len(masking_info), "Masking info should hold data foreach state in the Trajectory"

    observation.components[0].previous_state = mask_state(
        observation.components[0].previous_state,
        masking_info[0]
    )

    # Note that for each 2 consecutive components (c, c'), it holds that c.next_state == c'.previous_state,
    # so they should be masked in the same way.Therefore, we generate the masking info only once for each component.
    for i in range(len(observation.components) - 1):
        curr_component, next_component = observation.components[i], observation.components[i + 1]
        masked_state = mask_state(
            curr_component.next_state,
            masking_info[i + 1]
        )
        curr_component.next_state = masked_state
        next_component.previous_state = masked_state

    observation.components[-1].next_state = mask_state(observation.components[-1].next_state,
                                                             masking_info[-1])

    return observation


def mask_observations(observations: list[Observation], masking_info: list[list[set[GroundedPredicate]]]) -> list[Observation]:
    """
    Masks the predicates in multiple observations according to provided masking info.

    This utility function for batch masking of observations.

    :param observations: A list of observations to mask (should be grounded).
    :param masking_info: A list of masking info, one for each observation.
    :return: A list of masked observations.
    """
    return [mask_observation(obs, mask_info) for obs, mask_info in zip(observations, masking_info)]


# ============================================================================
# Combined Loading Utilities
# ============================================================================

def load_masked_observation(
    trajectory_path: Path,
    masking_info_path: Path,
    domain: Domain
) -> Observation:
    """
    Load a trajectory and apply masking in one unified call.

    This utility function combines the common pattern of:
    1. Loading a trajectory from a .trajectory file
    2. Loading masking info from a .masking_info file
    3. Grounding the observation completely
    4. Applying the mask to the grounded observation

    This eliminates code duplication and provides a single, well-tested entry point
    for the common workflow of loading masked observations for PI-SAM learning.

    :param trajectory_path: Path to the .trajectory file
    :param masking_info_path: Path to the .masking_info file
    :param domain: The PDDL domain for parsing predicates and actions
    :return: A fully grounded and masked observation ready for PI-SAM learning

    Example:
        >>> from pathlib import Path
        >>> from pddl_plus_parser.lisp_parsers import DomainParser
        >>>
        >>> domain = DomainParser("domain.pddl").parse_domain()
        >>> traj_path = Path("problem1.trajectory")
        >>> mask_path = Path("problem1.masking_info")
        >>>
        >>> masked_obs = load_masked_observation(traj_path, mask_path, domain)
    """

    # Step 1: Parse the trajectory file
    trajectory_parser = TrajectoryParser(partial_domain=domain)
    observation = trajectory_parser.parse_trajectory(trajectory_path)

    # Step 2: Load masking info from file
    masking_info = load_masking_info(masking_info_path, domain)

    # Step 3: Ground observation completely
    grounded_observation = ground_observation_completely(domain, observation)

    # Step 4: Apply mask to grounded observation
    masked_observation = mask_observation(grounded_observation, masking_info)

    return masked_observation
