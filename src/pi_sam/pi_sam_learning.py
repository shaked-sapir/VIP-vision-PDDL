import random
from typing import Dict, List, Tuple, Set

from pddl_plus_parser.models import GroundedPredicate, Domain, Observation, State, ActionCall, Predicate, Action
from sam_learning.core import LearnerDomain, extract_discrete_effects_partial_observability, extract_not_effects_partial_observability
from sam_learning.learners import SAMLearner

from src.utils.pddl import get_state_grounded_predicates, get_state_unmasked_predicates, get_state_masked_predicates
from utilities import NegativePreconditionPolicy


class PISAMLearner(SAMLearner):
    """
    A learner that applies the PI-SAM learning algorithm for learning from partial observations.

    This class extends the SAMLearner class and overrides methods to handle masked predicates
    in observations during learning. It does NOT provide utility methods for grounding or masking
    observations - those should be done externally using utility functions.

    Note: Observations should be grounded (using ground_observation_completely from utils.pddl)
    and masked (using mask_observation from utils.pddl) before being passed to learn_action_model().
    This class is focused solely on learning from already-prepared masked observations.

    :param partial_domain: The domain to learn from, which should be a partial domain.
    :param negative_preconditions_policy: The policy for handling negative preconditions.
    :param seed: A seed for random number generation to ensure reproducibility.
    """

    def __init__(
            self,
            partial_domain: Domain,
            negative_preconditions_policy: NegativePreconditionPolicy = NegativePreconditionPolicy.hard,
            seed: int = 42
    ):
        super().__init__(partial_domain, negative_preconditions_policy)
        self.seed = seed
        random.seed(seed)

    """
        PI-SAM rules methods
    """

    def handle_effects(self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """
        This method handles rules #2 (is_effect) & #3 (cannot_be_effect) of PI-SAM.
        :param grounded_action: the grounded action that was executed according to the trajectory.
        :param previous_state: the state that the action was executed on.
        :param next_state: the state that was created after executing the action on the previous state.
        :return:
        """
        # handle must_be_effects
        self.logger.debug(f"handling action {grounded_action.name} effects.")
        grounded_add_effects, grounded_del_effects = extract_discrete_effects_partial_observability(
            get_state_grounded_predicates(previous_state),
            get_state_grounded_predicates(next_state)
        )
        lifted_add_effects = self.matcher.get_possible_literal_matches(grounded_action, list(grounded_add_effects))
        lifted_delete_effects = self.matcher.get_possible_literal_matches(grounded_action, list(grounded_del_effects))

        self.logger.debug("Adding the effects to the action effects.")
        observed_action = self.partial_domain.actions[grounded_action.name]
        observed_action.discrete_effects.update(set(lifted_add_effects).union(lifted_delete_effects))

        # handle cannot_be_effects
        cannot_be_effects = extract_not_effects_partial_observability(
            get_state_grounded_predicates(previous_state),
            get_state_grounded_predicates(next_state)
        )

        # this is to preserve the current behaviour of Esam / future EPi-SAM, no use at the moment
        self.cannot_be_effect[grounded_action.name].update(
            {pred.untyped_representation for pred in cannot_be_effects}
        )
        self.logger.debug(f"finished handling action- {grounded_action.name} effects.")

    def add_new_action(self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """Create a new action in the domain.

        :param grounded_action: the grounded action that was executed according to the trajectory.
        :param previous_state: the state that the action was executed on.
        :param next_state: the state that was created after executing the action on the previous
        """
        self.logger.info(f"Adding the action {str(grounded_action)} to the domain.")
        # adding the preconditions each predicate is grounded in this stage.
        observed_action = self.partial_domain.actions[grounded_action.name]
        self.observed_actions.append(observed_action.name)
        self._add_new_action_preconditions(grounded_action, previous_state)
        self.handle_effects(grounded_action, previous_state, next_state)
        self.logger.debug(f"Finished adding the action {grounded_action.name}.")

    def update_action(self, grounded_action: ActionCall, previous_state: State, next_state: State) -> None:
        """updates an existing action in the domain based on a transition.

        :param grounded_action: the grounded action that was executed according to the trajectory.
        :param previous_state: the state that the action was executed on.
        :param next_state: the state that was created after executing the action on the previous
            state.
        """
        self.logger.debug(f"updating action {str(grounded_action)}.")
        self._update_action_preconditions(grounded_action, previous_state)
        self.handle_effects(grounded_action, previous_state, next_state)
        self.logger.debug(f"finished updating action {str(grounded_action)}.")

    def _add_new_action_preconditions(self, grounded_action: ActionCall, previous_state: State) -> None:
        """method to add new action's discrete preconditions.

        in the new_action case, this is equivalent to the regular sam, because we should not
        care about masking - we have to include the predicate as a potential precondition of
        the action, so we won't miss it in case we don't have any other transitions with this
        action - in this situation we might mistakenly exclude the predicate as a precondition
        leading to unsafe action model.

        :param grounded_action: the action that is currently being executed.
        """
        self.logger.debug(f"Adding the preconditions of {grounded_action.name} to the action model.")
        current_action = self.partial_domain.actions[grounded_action.name]
        previous_state_predicates = set(
            self.matcher.get_possible_literal_matches(
                # for not missing potential preconditions that were masked in the initial state.
                # that's why we don't use the masked here
                grounded_action, list(get_state_grounded_predicates(previous_state))
            )
        )

        for predicate in previous_state_predicates:
            current_action.preconditions.add_condition(predicate)

    def _update_action_preconditions(self, grounded_action: ActionCall, previous_state: State) -> None:
        """Updates the preconditions of an action after it was observed at least once.
        This method handles rule #1 (cannot_be_precondition) of PI-SAM.
        It is much similar to the SAM's update_action_preconditions method, but it is more restrictive
        to allow checking matches only for unmasked literals.

        :param grounded_action: the grounded action that is being executed in the trajectory component.
        """
        self.logger.debug(f"Updating the preconditions of {grounded_action.name} in the action model.")
        current_action = self.partial_domain.actions[grounded_action.name]
        previous_state_unmasked_predicates = set(
            self.matcher.get_possible_literal_matches(
                grounded_action, list(get_state_unmasked_predicates(previous_state))
            )
        )

        previous_state_masked_predicates = set(
            self.matcher.get_possible_literal_matches(
                grounded_action, list(get_state_masked_predicates(previous_state))
            )
        )
        conditions_to_remove = []
        for current_precondition in current_action.preconditions.root.operands:
            # assuming that the predicates in the preconditions are NOT nested.
            if (isinstance(current_precondition, Predicate)
                    and current_precondition not in previous_state_masked_predicates  # we can't decide for masked preds
                    and current_precondition not in previous_state_unmasked_predicates):
                conditions_to_remove.append(current_precondition)

        for condition in conditions_to_remove:
            current_action.preconditions.remove_condition(condition)

    def learn_action_model(self, observations: List[Observation], **kwargs) -> Tuple[LearnerDomain, Dict[str, str]]:
        """
        Learn action model from observations.

        IMPORTANT: Observations should be grounded and masked BEFORE calling this method.

        Preparation steps (do these BEFORE calling this method):
        1. Ground: ground_observation_completely(domain, obs) from utils.pddl
        2. Generate masking: PredicateMasker.mask_observation(grounded_obs)
        3. Apply masks: mask_observation(grounded_obs, masking_info) from utils.pddl
        4. Then call: learner.learn_action_model([masked_obs])

        :param observations: A list of grounded and masked observations.
        :return: A tuple of (learned domain, learning report dictionary).
        """
        return super().learn_action_model(observations)
