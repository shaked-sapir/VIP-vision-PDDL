from typing import Set, Tuple

from pddl_plus_parser.models import Domain, State, ActionCall, GroundedPredicate
from sam_learning.core.matching_utils import extract_discrete_effects, extract_not_effects
from sam_learning.learners import SAMLearner
from utilities import NegativePreconditionPolicy

from src.pi_sam.noisy_pisam.noisy_learner_mixin import NoisyLearnerMixin
from src.utils.pddl import get_state_grounded_predicates


class NoisySAMLearner(NoisyLearnerMixin, SAMLearner):
    """SAM learner with conflict-driven noisy learning.

    Uses full-observability effect extraction:
      - extract_discrete_effects  (must-be effects, prev + next)
      - extract_not_effects       (cannot-be effects, prev only)

    Overrides ``_flip_fluent_in_state`` to handle missing fluents via
    closed-world assumption (creates a new positive grounded predicate
    if neither the fluent nor its negation exists in the state).
    """

    def __init__(
        self,
        partial_domain: Domain,
        negative_preconditions_policy: NegativePreconditionPolicy = NegativePreconditionPolicy.hard,
    ):
        SAMLearner.__init__(self, partial_domain, negative_preconditions_policy)
        self._init_noisy_fields()

    # ------------------------------------------------------------------
    # Template methods
    # ------------------------------------------------------------------

    def _extract_discrete_effects(
        self, prev_preds: Set[GroundedPredicate], next_preds: Set[GroundedPredicate],
    ) -> Tuple[Set[GroundedPredicate], Set[GroundedPredicate]]:
        return extract_discrete_effects(prev_preds, next_preds)

    def _extract_cannot_be_effects(
        self, prev_preds: Set[GroundedPredicate], next_preds: Set[GroundedPredicate],
    ) -> Set[GroundedPredicate]:
        return extract_not_effects(prev_preds)

    def _delegate_handle_effects(
        self, grounded_action: ActionCall, previous_state: State, next_state: State,
    ) -> None:
        """SAM-style effect handling on the no-conflict path."""
        self.logger.debug(f"handling action {grounded_action.name} effects.")
        prev_preds = get_state_grounded_predicates(previous_state)
        next_preds = get_state_grounded_predicates(next_state)

        grounded_add, grounded_del = extract_discrete_effects(prev_preds, next_preds)
        lifted_add = self.matcher.get_possible_literal_matches(grounded_action, list(grounded_add))
        lifted_del = self.matcher.get_possible_literal_matches(grounded_action, list(grounded_del))

        observed_action = self.partial_domain.actions[grounded_action.name]
        observed_action.discrete_effects.update(set(lifted_add) | set(lifted_del))

        cannot_be = extract_not_effects(prev_preds)
        lifted_cannot = self.matcher.get_possible_literal_matches(grounded_action, list(cannot_be))
        self.cannot_be_effect[grounded_action.name].update(lifted_cannot)

        self.logger.debug(f"finished handling action- {grounded_action.name} effects.")

    # ------------------------------------------------------------------
    # Closed-world _flip_fluent_in_state override
    # ------------------------------------------------------------------

    def _flip_fluent_in_state(self, state: State, fluent_str: str) -> None:
        """Flip fluent_str in the given state.

        If neither the fluent nor its negation exists, treat it as implicitly
        negated (closed-world assumption) and add a new positive grounded
        predicate to the state.
        """
        from src.action_model.pddl2gym_parser import negate_str_predicate

        candidates = {fluent_str, negate_str_predicate(fluent_str)}

        for gp in get_state_grounded_predicates(state):
            if gp.untyped_representation not in candidates:
                continue

            base_key = (
                gp.lifted_untyped_representation
                if gp.is_positive
                else negate_str_predicate(gp.lifted_untyped_representation)
            )

            for p in state.state_predicates[base_key]:
                if p.untyped_representation in candidates:
                    p.is_positive = not p.is_positive
                    return

        # Fluent not found — closed-world: create a new positive literal
        parts = (
            fluent_str.strip()[1:-1].split()
            if "not" not in fluent_str
            else fluent_str.strip()[6:-2].split()
        )
        pred_name = parts[0]
        obj_names = parts[1:]

        if pred_name not in self.partial_domain.predicates:
            raise ValueError(f"Predicate {pred_name} not found in domain")

        domain_pred = self.partial_domain.predicates[pred_name]
        param_names = list(domain_pred.signature.keys())
        if len(param_names) != len(obj_names):
            raise ValueError(
                f"Parameter count mismatch for {fluent_str}: "
                f"expected {len(param_names)} but got {len(obj_names)}"
            )

        new_gp = GroundedPredicate(
            name=pred_name,
            signature=domain_pred.signature,
            object_mapping=dict(zip(param_names, obj_names)),
            is_positive=True,
            is_masked=False,
        )
        base_key = new_gp.lifted_untyped_representation
        state.state_predicates.setdefault(base_key, set()).add(new_gp)

    # ------------------------------------------------------------------
    # SAMLearner method overrides (different precondition signatures)
    # ------------------------------------------------------------------

    def add_new_action(
        self, grounded_action: ActionCall, previous_state: State, next_state: State,
    ) -> None:
        self.logger.info(f"Adding the action {str(grounded_action)} to the domain.")
        observed_action = self.partial_domain.actions[grounded_action.name]
        self.observed_actions.append(observed_action.name)
        self._add_new_action_preconditions(grounded_action)
        self.handle_effects(grounded_action, previous_state, next_state)
        self.logger.debug(f"Finished adding the action {grounded_action.name}.")

    def update_action(
        self, grounded_action: ActionCall, previous_state: State, next_state: State,
    ) -> None:
        self.logger.debug(f"updating action {str(grounded_action)}.")
        self._update_action_preconditions(grounded_action)
        self.handle_effects(grounded_action, previous_state, next_state)
        self.logger.debug(f"finished updating action {str(grounded_action)}.")

    def _remove_unobserved_actions_from_partial_domain(self):
        self.logger.debug("Removing unobserved actions from the partial domain")
        for action in [a for a in self.partial_domain.actions if a not in self.observed_actions]:
            self.partial_domain.actions.pop(action)
