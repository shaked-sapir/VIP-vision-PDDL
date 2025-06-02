import random
from typing import Dict, List, Tuple, Set

from pddl_plus_parser.models import GroundedPredicate, Domain, Observation, State
from sam_learning.core import LearnerDomain
from sam_learning.learners import SAMLearner

from src.pi_sam.predicate_masking import MaskingType, PredicateMasker
from src.utils.pddl import copy_observation, get_all_possible_groundings, get_state_grounded_predicates


class PISAMLearner(SAMLearner):
    """
    A learner that applies the PI-SAM learning algorithm, which includes masking strategies for grounded predicates
    to provide partial observations during the learning process.

    This class extends the SAMLearner class and overrides methods to incorporate masking strategies for grounded predicates.

    :param partial_domain: The domain to learn from, which should be a partial domain with grounded predicates.
    :param seed: An seed for random number generation to ensure reproducibility of the masking process.
    """

    def __init__(self, partial_domain: Domain, predicate_masker: PredicateMasker = None, seed: int = 42):
        super().__init__(partial_domain)

        self.predicate_masker = predicate_masker or PredicateMasker(seed=seed)
        self.seed = seed
        random.seed(seed)

    """
    Going over SAMLearner methods to see what methods should be modified regarding the masking operations.
    The best thing, in my opinion, is to have my own thoughts about it, and then ask Argaman to hold a conversation about it,
    see what she says and that I am heading at the right direction. maybe she could also have a DR for my PI-SAM agent.
    
    
    @learn_action_model: GOOD - as this is the main function of the class, so we should change its inner methods.
    @are_states_different: GOOD - is compares states, which use the `GroundedPredicate.untyped_representation()` method to determine 
                                  equality, and we override it in the MaskableGroundedPredicate class.
    @end_measure_learning_time: GOOD - just time comparison
    @start_measure_learning_time: GOOD - just time comparison
    @construct_safe_actions: GOOD - as it should be safe by default #TODO actually we have to check this thing up with Argaman/Yarin
    @handle_negative_preconditions_policy: GOOD - uses only the relevant field which is inherited by the papa class.
    @remove_negative_preconditions: GOOD - when getting there, there should not be any masked predicates anymore
    @deduce_initial_inequality_preconditions: GOOD - there should not be any manipulations regarding the masked predicates
    @handle_single_trajectory_component: BAD - (add_new_action)/(update_action) -> `extract_effects`
    @_verify_parameter_duplication: GOOD - this is not related to predicates by any means
    @update_action: BAD - depends on the `_handle_action_effects` -> `extract_effects`
    @add_new_action: BAD - depends on the `_handle_action_effects` -> `extract_effects`
    @_construct_learning_report: GOOD - just constructing learning report from the un/safe actions
    @_add_new_action_preconditions: GOOD - for the initial preconditions definition - we define all predicates, regardless whether they are masked or not. it will only matter for applying SAM rules
    @_update_action_preconditions: BAD - needs to update the `extract_effects` code to check for masked predicates
    @_handle_action_effects: BAD - needs to update the `extract_effects` code to check for masked predicates (other than that - it should be ok though)
    @_handle_consts_in_effects:
    @_remove_unobserved_actions_from_partial_domain: GOOD - affected only by `observed_actions`
    """


    """
    important note! # TODO: go over and delete afterwards
        
    we can pass the snapshots, but it will require as to override some logics of SAMLearner in this class,
    in order to make sure the masked literals are handled correctly.
    """

    def set_masking_strategy(self, masking_strategy: MaskingType, masking_kwargs: dict = None) -> None:
        self.predicate_masker.set_masking_strategy(masking_strategy, **masking_kwargs)

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
            all_grounded_predicates[lifted_predicate_name] = get_all_possible_groundings(
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
            all_matching_predicates = state.state_predicates[masked_pred.lifted_untyped_representation]
            for pred in all_matching_predicates:
                if pred == masked_pred:
                    pred.is_masked = True
                    break
            else:
                # If the predicate is not found, we should log a warning or handle it as needed
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
        #  TODO: list[set[GroundedPredicate]] is not a good type hint (and especially if we wrap it in list for the observation list. consider expoerting to a class

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
        assert len(observations)+1 == len(masking_info), "Masking info should hold data foreach state in the Trajectory"

        #TODO: this is the same code as the "return" line, might be more comprhesive for debugging purposes - remove when works
        # masked_observations = []
        # for obs, mask_info in zip(observations, masking_info):
        #     masked_obs = self._mask_observation(obs, mask_info)
        #     masked_observations.append(masked_obs)
        # return masked_observations

        return [self._mask_observation(obs, mask_info) for obs, mask_info in zip(observations, masking_info)]

    def mask_observations_by_strategy(self, observations: List[Observation]) -> List[List[set[GroundedPredicate]]]:
        """
        Masks the predicates in each observation according to the masking strategy.

        :param observations: A list of observations to mask.
        :return: A list of lists, where each inner list contains sets of predicates to mask for each state in the observation.
        """
        full_observations = [self.ground_observation_completely(obs) for obs in observations]
        return [self.predicate_masker.mask_observation(obs) for obs in full_observations]

    def learn_action_model(self, observations: List[Observation], masking_info: List[List[Set[GroundedPredicate]]]) -> Tuple[LearnerDomain, Dict[str, str]]:
        # each observation has a dict, each dict is {<state_index>: <set of masked predicates>} where the set's values are the untyped representations of the masked predicates.
        #  TODO: create a proper class / type for this
        full_observations = [self.ground_observation_completely(obs) for obs in observations]
        masked_observations = self.mask_observations(full_observations, masking_info)
        return super().learn_action_model(
            masked_observations)  # TODO: check if we need to override this method or not (or maybe its inner methods)
