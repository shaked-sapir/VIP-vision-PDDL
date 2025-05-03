import itertools
import random
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Tuple, Set

from pddl_plus_parser.models import GroundedPredicate, SignatureType, Domain, Observation, State, PDDLObject, Predicate
from sam_learning.learners import SAMLearner
from sam_learning.core import LearnerDomain


class MaskingType(str, Enum):
    RANDOM = "random"
    PERCENTAGE = "percentage"


class MaskingStrategy(ABC):
    """
    This class serves as a base class for masking a set of predicates based on some logic/strategy.
    """
    @abstractmethod
    def mask(self, predicates: set[GroundedPredicate], *args, **kwargs) -> set[GroundedPredicate]:
        raise NotImplementedError

class RandomMasking(MaskingStrategy):
    """
    this class masks a certain predicate with probability p, and leaves it as it is with probability (1 - p).
    """
    def mask(self, predicates: set[GroundedPredicate], masking_proba: float = 0.3, *args, **kwargs) -> set[GroundedPredicate]:
        for predicate in predicates:
            if random.random() < masking_proba:
                predicate.is_masked = True
        return predicates


class PercentageMasking(MaskingStrategy):
    """
    This class masks some p percent of predicates from the set
    """
    def mask(self, predicates: set[GroundedPredicate], masking_ratio: float = 0.75, *args, **kwargs) -> set[GroundedPredicate]:
        sample_size = max(1, round(len(predicates) * masking_ratio))  # Ensure at least 1 element if p > 0
        sample = set(random.sample(list(predicates), sample_size))
        for predicate in sample:
            predicate.is_masked = True
        return predicates



class PISAMLearner(SAMLearner):
    def __init__(self, partial_domain: Domain):
        super().__init__(partial_domain)

        self.masking_strategies = {
            MaskingType.RANDOM: RandomMasking(),
            MaskingType.PERCENTAGE: PercentageMasking()
        }

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

    def mask(self, predicates: set[GroundedPredicate], masking_strategy: MaskingType = MaskingType.RANDOM, *args, **kwargs) -> set[GroundedPredicate]:
        return self.masking_strategies[masking_strategy].mask(predicates, *args, **kwargs)

        """
        Important note!
        we decided on a couple of things (summarize this and insert into the progress report with Roni:
        1. we will create the "full" state for each state in the trajectory, i.e. each literal in the domain will appear
        either with positive or negative form (in oppose to the current trajectory files, which only include predicates 
        [positive form only]). this way we'll be consistent with the definitions of PI-SAM and could apply its rules
        properly.
    
        
        3. we can pass the snapshots, but it will require as to override some logics of SAMLearner in this class,
        in order to make sure the masked literals are handled correctly.
        """

    def get_all_possible_groundings(self, predicate: Predicate, grounded_objects: Dict[str, PDDLObject]) -> Set[GroundedPredicate]:
        param_names = list(predicate.signature.keys())
        param_types = list(predicate.signature.values()) # this one also handles signature with multiple objects of the same type

        # Get all objects compatible with each parameter type
        object_domains = []
        for t in param_types:
            matches = [obj.name for obj in grounded_objects.values() if obj.type.is_sub_type(t)]
            object_domains.append(matches)

        grounded = set()

        for values in itertools.product(*object_domains):
            mapping = dict(zip(param_names, values))
            grounded.add(GroundedPredicate(
                name=predicate.name,
                signature=predicate.signature,
                object_mapping=mapping,
                is_positive=predicate.is_positive
            ))

        return grounded

    def _prepare_state(self, state: State, grounded_objects: Dict[str, PDDLObject]) -> State:
        # TODO: implement
        # return state
        for lifted_predicate in self.partial_domain.predicates.values():
            predicate_representation = lifted_predicate.untyped_representation()
            all_predicate_groundings = self.get_all_possible_groundings(lifted_predicate, grounded_objects)

            #TODO: 1. compute diff between all to state, and 2. put the diff in the state in negative form. 3. make sure that literals in a state are necessarily positives


    def _prepare_observation(self, observation: Observation) -> Observation:
        """
        for a given trajectory, create all possible literals (either positive or negative) for each state,
        where positive literals are given by the predicates of the state and negative literals are the "complementary"
        set of the predicates of the state.

        :param observation: an observation (trajectory) to handle
        :return: a full observation, with all possible literals.
        """

        observation.components[0].previous_state = self._prepare_state(observation.components[0].previous_state, observation.grounded_objects)

        # for two consecutive components c_(i) and c_(i+1), it holds that c_(i).next_state == c_(i+1).prev_state
        for i in range(len(observation.components)-1):
            curr_component, next_component = observation.components[i], observation.components[i+1]
            state_with_all_grounded_literals = self._prepare_state(curr_component.next_state, observation.grounded_objects)
            curr_component.next_state = state_with_all_grounded_literals
            next_component.previous_state = state_with_all_grounded_literals

        observation.components[-1].next_state = self._prepare_state(observation.components[-1].next_state, observation.grounded_objects)

        return observation

    def prepare_observations(self, observations: List[Observation]) -> List[Observation]:
        """
        For each grounded literal in the domain, create a proper instance in the observations' states.

        :param observations:
        :return:
        """
        return [self._prepare_observation(observation) for observation in observations]

    def learn_action_model(self, observations: List[Observation]) -> Tuple[LearnerDomain, Dict[str, str]]:
        observations = self.prepare_observations(observations)
        return super().learn_action_model(observations)
