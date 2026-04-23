from typing import Set, Tuple

from pddl_plus_parser.models import Domain, State, ActionCall, GroundedPredicate
from sam_learning.core.matching_utils import (
    extract_discrete_effects_partial_observability,
    extract_not_effects_partial_observability,
)
from utilities import NegativePreconditionPolicy

from src.pi_sam.noisy_pisam.noisy_learner_mixin import NoisyLearnerMixin
from src.pi_sam.pi_sam_learning import PISAMLearner


class NoisyPisamLearner(NoisyLearnerMixin, PISAMLearner):
    """PI-SAM learner with conflict-driven noisy learning.

    Uses partial-observability effect extraction:
      - extract_discrete_effects_partial_observability (must-be effects)
      - extract_not_effects_partial_observability      (cannot-be effects)

    Precondition methods (_add_new_action_preconditions, _update_action_preconditions)
    are resolved via MRO to PISAMLearner, which accepts (grounded_action, previous_state).
    """

    def __init__(
        self,
        partial_domain: Domain,
        negative_preconditions_policy: NegativePreconditionPolicy = NegativePreconditionPolicy.hard,
        seed: int = 42,
    ):
        PISAMLearner.__init__(self, partial_domain, negative_preconditions_policy, seed)
        self._init_noisy_fields()

    # --- Template methods ---

    def _extract_discrete_effects(
        self, prev_preds: Set[GroundedPredicate], next_preds: Set[GroundedPredicate],
    ) -> Tuple[Set[GroundedPredicate], Set[GroundedPredicate]]:
        return extract_discrete_effects_partial_observability(prev_preds, next_preds)

    def _extract_cannot_be_effects(
        self, prev_preds: Set[GroundedPredicate], next_preds: Set[GroundedPredicate],
    ) -> Set[GroundedPredicate]:
        return extract_not_effects_partial_observability(prev_preds, next_preds)

    def _delegate_handle_effects(
        self, grounded_action: ActionCall, previous_state: State, next_state: State,
    ) -> None:
        PISAMLearner.handle_effects(self, grounded_action, previous_state, next_state)
