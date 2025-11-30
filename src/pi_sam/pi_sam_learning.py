import random
from typing import Dict, List, Tuple

from pddl_plus_parser.models import Domain, Observation, State, ActionCall, Predicate, ObservedComponent
from sam_learning.core import LearnerDomain
from sam_learning.core.matching_utils import extract_discrete_effects_partial_observability, \
    extract_not_effects_partial_observability
from sam_learning.learners import SAMLearner
from utilities import NegativePreconditionPolicy

from src.utils.pddl import get_state_grounded_predicates, get_state_unmasked_predicates, get_state_masked_predicates


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

        lifted_cannot_be_effects = self.matcher.get_possible_literal_matches(grounded_action, list(cannot_be_effects))

        self.cannot_be_effect[grounded_action.name].update(
            {pred for pred in lifted_cannot_be_effects}
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

    def handle_single_trajectory_component(self, component: ObservedComponent) -> None:
        """Handles a single trajectory component as a part of the learning process.

        :param component: the trajectory component that is being handled at the moment.
        """
        previous_state = component.previous_state
        grounded_action = component.grounded_action_call
        next_state = component.next_state

        if self._verify_parameter_duplication(grounded_action):
            self.logger.warning(
                f"{str(grounded_action)} contains duplicated parameters! Not supported in SAM."
                f"aborting learning from component"
            )
            return

        self.triplet_snapshot.create_triplet_snapshot(
            previous_state=previous_state.copy(), # MY CODE !! (copying the state to avoid modifying the original state by the hack)
            next_state=next_state.copy(), # MY CODE !! (copying the state to avoid modifying the original state by the hack)
            current_action=grounded_action,
            observation_objects=self.current_trajectory_objects,
        )
        if grounded_action.name not in self.observed_actions:
            self.add_new_action(grounded_action, previous_state, next_state)

        else:
            self.update_action(grounded_action, previous_state, next_state)

    def _remove_unobserved_actions_from_partial_domain(self):
        """Removes the actions that were not observed from the partial domain."""
        self.logger.debug("Removing unobserved actions from the partial domain")
        actions_to_remove = [action for action in self.partial_domain.actions if action not in self.observed_actions]
        for action in actions_to_remove:
            self.partial_domain.actions.pop(action)

    def learn_action_model(self, observations: List[Observation], **kwargs) -> Tuple[
        LearnerDomain, Dict[str, str]]:  # MY CODE !! (ading the **kwargs to allow future extensions)
        """Learn the SAFE action model from the input trajectories.

        :param observations: the list of trajectories that are used to learn the safe action model.
        :return: a domain containing the actions that were learned.
        """
        self.logger.info("Starting to learn the action model!")
        self.start_measure_learning_time()
        self.deduce_initial_inequality_preconditions()
        self._complete_possibly_missing_actions()
        for observation in observations:
            self.current_trajectory_objects = observation.grounded_objects
            for component in observation.components:
                if not component.is_successful:
                    self.logger.warning("Skipping the transition because it was not successful.")
                    continue

                self.handle_single_trajectory_component(component)

        self.construct_safe_actions()
        self._remove_unobserved_actions_from_partial_domain()
        self.handle_negative_preconditions_policy()
        self.end_measure_learning_time()
        learning_report = self._construct_learning_report()
        return self.partial_domain, learning_report

