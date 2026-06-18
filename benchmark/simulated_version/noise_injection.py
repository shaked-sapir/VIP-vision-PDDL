"""Bounded noise injection for simulated conflict-search experiments.

Pipeline
--------
Given a fully ground-truth observation, this module produces a degraded
copy by:
  1. Masking a subset of fluents (making them invisible to the learner).
  2. Flipping the polarity of a subset of the *remaining* (visible) fluents.

The exact set of injected flips is returned as ``set[FluentLevelPatch]``,
which is the same type the conflict search produces, so precision/recall
can be computed directly by set comparison.
"""

from typing import List, Set, Tuple

from pddl_plus_parser.models import Observation

from src.pi_sam.noisy_pisam.typings import FluentLevelPatch
from src.pi_sam.predicate_masking import PredicateMasker
from src.pi_sam.predicate_noising import PredicateNoiser
from src.utils.pddl_state import (
    copy_observation_linked,
    flip_fluent_in_state,
    get_state_unmasked_predicates,
)


def create_bounded_noisy_observation(
    gt_observation: Observation,
    masker: PredicateMasker,
    noiser: PredicateNoiser,
    obs_idx: int,
) -> Tuple[Observation, List[Set], Set[FluentLevelPatch]]:
    """Inject bounded masking and polarity noise into a GT observation.

    Args:
        gt_observation: A fully ground-truth observation (no masking applied).
        masker: Configured masker that decides which fluents to hide.
        noiser: Configured noiser that selects which visible fluents to flip.
        obs_idx: The observation index to embed in each ``FluentLevelPatch``
            (must match the index used when passing the observation to the
            conflict search).

    Returns:
        A three-tuple:
          - ``noisy_obs``: Deep copy of the observation with masking and
            polarity flips applied.  Consecutive components share the same
            boundary state object (shared-state invariant is preserved).
          - ``masking_info``: ``list[set[GroundedPredicate]]`` with one entry
            per state (N+1 total), matching the format of ``save_masking_info``.
          - ``ground_truth_flips``: ``set[FluentLevelPatch]`` recording every
            injected polarity flip.  Compare this against the conflict search's
            proposed patches to compute precision/recall.
    """
    noisy_obs = copy_observation_linked(gt_observation)

    # N+1 states: initial state + one next_state per component.
    states = [noisy_obs.components[0].previous_state] + [
        comp.next_state for comp in noisy_obs.components
    ]

    masking_info: List[Set] = []
    ground_truth_flips: Set[FluentLevelPatch] = set()

    for state_idx, state in enumerate(states):
        # Step 1: apply masking — sets is_masked=True on selected predicates.
        masked_predicates = masker.mask_state(state)
        masking_info.append(masked_predicates)

        # Step 2: select a subset of the *unmasked* predicates to flip.
        unmasked = get_state_unmasked_predicates(state)
        to_flip = noiser.noise(unmasked)

        # Determine the (comp_idx, state_type) coordinates for patch recording.
        if state_idx == 0:
            comp_idx, state_type = 0, "prev"
        else:
            comp_idx, state_type = state_idx - 1, "next"

        for pred in to_flip:
            fluent_str = pred.untyped_representation
            flip_fluent_in_state(state, fluent_str)
            ground_truth_flips.add(FluentLevelPatch(
                observation_index=obs_idx,
                component_index=comp_idx,
                state_type=state_type,
                fluent=fluent_str,
            ))

    return noisy_obs, masking_info, ground_truth_flips
