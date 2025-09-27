import random
from typing import Dict, List, Tuple, Set

from pddl_plus_parser.models import GroundedPredicate, Domain, Observation, State, ActionCall, Predicate, Action
from sam_learning.core import LearnerDomain, extract_discrete_effects_partial_observability, extract_not_effects_partial_observability
from sam_learning.learners import SAMLearner

from src.pi_sam.predicate_masking import MaskingType, PredicateMasker
from src.utils.pddl import copy_observation, get_all_possible_groundings, get_state_grounded_predicates, \
    get_state_unmasked_predicates, get_state_masked_predicates
from utilities import NegativePreconditionPolicy


class PISAMLearner(SAMLearner):
    """
    A learner that applies the PI-SAM learning algorithm, which includes masking strategies for grounded predicates
    to provide partial observations during the learning process.

    This class extends the SAMLearner class and overrides methods to incorporate masking strategies for grounded predicates.

    :param partial_domain: The domain to learn from, which should be a partial domain with grounded predicates.
    :param predicate_masker: An instance of PredicateMasker to handle the masking of predicates. If None, a default
        PredicateMasker will be created with the provided seed. This is for the option to create a masking via strategy
        if masking info isn't known ahead of time.
    :param seed: A seed for random number generation to ensure reproducibility of the masking process.
    """

    def __init__(
            self,
            partial_domain: Domain,
            negative_preconditions_policy: NegativePreconditionPolicy = NegativePreconditionPolicy.hard,
            predicate_masker: PredicateMasker = None,
            seed: int = 42
    ):
        super().__init__(partial_domain, negative_preconditions_policy)

        self.predicate_masker = predicate_masker or PredicateMasker(seed=seed)
        self.seed = seed
        random.seed(seed)

    """
    important note! # TODO: go over and delete afterwards
        
    we can pass the snapshots, but it will require as to override some logics of SAMLearner in this class,
    in order to make sure the masked literals are handled correctly.
    """

    def set_masking_strategy(self, masking_strategy: MaskingType, masking_kwargs: dict = None) -> None:
        self.predicate_masker.set_masking_strategy(masking_strategy, **masking_kwargs)

    def mask_observations_by_strategy(self, observations: List[Observation]) -> List[List[set[GroundedPredicate]]]:
        """
        Masks the predicates in each observation according to the masking strategy.

        :param observations: A list of observations to mask.
        :return: A list of lists, where each inner list contains sets of predicates to mask for each state in the observation.
        """
        full_observations = [self.ground_observation_completely(obs) for obs in observations]
        return [self.predicate_masker.mask_observation(obs) for obs in full_observations]

    def _get_all_possible_groundings_for_domain(self, observation: Observation) -> Dict[str, Set[GroundedPredicate]]:
        """
        For each lifted predicate in the domain, compute all possible groundings for the given observation.
        Note: this returns all groundings as positive literals, regardless of the actual state of the observation -
        so negativity can be handled as needed in states creation.

        :param observation: The observation containing grounded objects.
        :return: A dictionary mapping lifted predicate names to their possible grounded predicates.
        """
        grounded_objects = observation.grounded_objects
        all_grounded_predicates = {}

        for lifted_predicate_name, lifted_predicate in self.partial_domain.predicates.items():
            # keys are the untyped representations of the predicates, matching the predicate dicts of states
            all_grounded_predicates[lifted_predicate.untyped_representation] = get_all_possible_groundings(
                lifted_predicate, grounded_objects)

        return all_grounded_predicates

    @staticmethod
    def _ground_all_predicates_in_state(state: State,
                                        all_domain_grounded_predicates: Dict[str, Set[GroundedPredicate]]) -> State:
        """
        for each predicate in domain predicates, check all its possible groundings against the state's grounded
        predicates: if a grounding does not exist in the state then add it to the state as a negative literal.

        :param state: the state to ground all predicates in
        :param all_domain_grounded_predicates: a dictionary mapping each predicate name to its possible grounded
        predicates in the domain
        :return: a state with all predicates grounded, either positive or negative.
        """
        new_state = state.copy()

        # Add all grounded predicates from the state
        for predicate_name, grounded_predicates in state.state_predicates.items():
            new_state.state_predicates[predicate_name] = set(grounded_predicates)

        # For each predicate in the domain, check if it exists in the state, if not - add it as a negative literal
        for predicate_name, grounded_predicates in all_domain_grounded_predicates.items():
            for grounded_predicate in grounded_predicates:
                # We have to check if the there are any predicates with the same name in the state, and handle properly
                if grounded_predicate not in new_state.state_predicates.get(predicate_name, set()):
                    (new_state.state_predicates.setdefault(predicate_name, set())
                     .add(grounded_predicate.copy(is_negated=True)))

        return new_state

    def ground_all_states_in_observation(self, observation: Observation,
                                         all_domain_grounded_predicates: Dict[str, Set[GroundedPredicate]]
                                         ) -> Observation:
        """
        for a given observation, ground all predicates in each state of the observation.

        :param observation: an observation (trajectory) to handle
        :param all_domain_grounded_predicates: a dictionary mapping each predicate name to its possible groundings in the domain,
               using the objects of the observation.
        :return: a full observation, with all possible literals for each state
        """
        new_observation = copy_observation(observation)
        for component in new_observation.components:
            component.previous_state = self._ground_all_predicates_in_state(
                component.previous_state, all_domain_grounded_predicates)
            component.next_state = self._ground_all_predicates_in_state(
                component.next_state, all_domain_grounded_predicates)

        return new_observation

    def ground_observation_completely(self, observation: Observation) -> Observation:
        """
        Ground all predicates in the states of the observation.

        :param observation: The observation to ground.
        :return: A new observation with all predicates grounded.
        """
        all_domain_grounded_predicates = self._get_all_possible_groundings_for_domain(observation)
        return self.ground_all_states_in_observation(observation, all_domain_grounded_predicates)

    @staticmethod
    def _mask_state(state: State, masking_info: Set[GroundedPredicate]) -> State:
        """
        Masks the predicates in the state according to the masking info provided.

        :param state: The state to mask predicates in.
        :param masking_info: A set of predicates to mask in the state.
        :return: A state with predicates masked according to the masking info provided.
        """
        for masked_pred in masking_info:
            # state.state_predicates are only positive- a positive version of the masked predicate is needed for access
            masked_pred_positive_form = masked_pred.copy(is_negated=not masked_pred.is_positive)
            all_matching_predicates = state.state_predicates[masked_pred_positive_form.lifted_untyped_representation]
            for pred in all_matching_predicates:
                if pred == masked_pred:
                    pred.is_masked = True
                    break
            else:
                print(f"Warning: Masked predicate {masked_pred} not found in state.")

        return state

    def _mask_observation(self, observation: Observation, masking_info: List[Set[GroundedPredicate]]) -> Observation:
        """
        This function masks the predicates in the observation so it could be used for learning with partial information.
        NOTE: we assume that observation is "full" - meaning that all predicates are grounded in the states.

        :param observation: The observation to mask predicates in.
        :param masking_info: A dictionary mapping state indices to sets of predicates to mask.
        :return: An observation with predicates masked according to the masking info provided.
        """
        assert len(observation.components)+1 == len(masking_info), "Masking info should hold data foreach state in the Trajectory"

        observation.components[0].previous_state = self._mask_state(
            observation.components[0].previous_state,
            masking_info[0]
        )

        # Note that for each 2 consecutive components (c, c'), it holds that c.next_state == c'.previous_state,
        # so they should be masked in the same way.Therefore, we generate the masking info only once for each component.
        for i in range(len(observation.components) - 1):
            curr_component, next_component = observation.components[i], observation.components[i + 1]
            masked_state = self._mask_state(
                curr_component.next_state,
                masking_info[i + 1]
            )
            curr_component.next_state = masked_state
            next_component.previous_state = masked_state

        observation.components[-1].next_state = self._mask_state(observation.components[-1].next_state,
                                                                 masking_info[-1])

        return observation

    def mask_observations(self, observations: List[Observation], masking_info: List[List[Set[GroundedPredicate]]]) -> List[Observation]:
        """
        Masks the predicates in each observation according to the masking strategy.

        :param observations: A list of observations to mask.
        :param masking_info: Optional information about which predicates to mask in each state.
        :return: A list of masked observations.
        """

        return [self._mask_observation(obs, mask_info) for obs, mask_info in zip(observations, masking_info)]

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
                grounded_action, list(get_state_grounded_predicates(previous_state)) # so we won't miss potential preconds that were masked in the initial state - this is why we don't use the masked here
                # grounded_action, list(get_state_unmasked_predicates(previous_state))
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

    def learn_action_model(self, observations: List[Observation], *, masking_info: List[List[set[GroundedPredicate]]]
                           ) -> Tuple[LearnerDomain, Dict[str, str]]:
        # each observation has a dict, each dict is {<state_index>: <set of masked predicates>} where the set's values are the untyped representations of the masked predicates.
        #  TODO: create a proper class / type for this
        full_observations = [self.ground_observation_completely(obs) for obs in observations]
        masked_observations = self.mask_observations(full_observations, masking_info)
        return super().learn_action_model(
            masked_observations)
