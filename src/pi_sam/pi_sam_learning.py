import random
from pathlib import Path
from typing import Dict, List, Tuple, Set

from pddl_plus_parser.lisp_parsers import TrajectoryParser, ProblemParser
from pddl_plus_parser.models import GroundedPredicate, Domain, Observation, State, Problem
from sam_learning.core import LearnerDomain
from sam_learning.learners import SAMLearner

from src.pi_sam.predicate_masking import MaskingType, RandomMasking, PercentageMasking
from src.utils.pddl import copy_observation, observation_to_trajectory_file, get_all_possible_groundings


class PISAMLearner(SAMLearner):
    """
    A learner that applies the PI-SAM learning algorithm, which includes masking strategies for grounded predicates
    to provide partial observations during the learning process.

    This class extends the SAMLearner class and overrides methods to incorporate masking strategies for grounded predicates.

    :param partial_domain: The domain to learn from, which should be a partial domain with grounded predicates.
    :param seed: An seed for random number generation to ensure reproducibility of the masking process.
    """

    def __init__(self, partial_domain: Domain, seed: int = 42):
        super().__init__(partial_domain)

        self.masking_strategies = {
            MaskingType.RANDOM: RandomMasking(),
            MaskingType.PERCENTAGE: PercentageMasking()
        }
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

    def _ground_all_states_in_observation(self,
                                          observation: Observation,
                                          all_domain_grounded_predicates: Dict[str, Set[GroundedPredicate]]
                                          ) -> Observation:
        """
        for a given observation, ground all predicates in each state of the observation.

        :param observation: an observation (trajectory) to handle
        :return: a full observation, with all possible literals for each state
        """
        new_observation = copy_observation(observation)
        for component in new_observation.components:
            component.previous_state = self._ground_all_predicates_in_state(
                component.previous_state, all_domain_grounded_predicates)
            component.next_state = self._ground_all_predicates_in_state(
                component.next_state, all_domain_grounded_predicates)

        return new_observation

    def mask_observation(self, observation: Observation) -> Tuple[Observation, Dict[int, Set[GroundedPredicate]]]:
        # TODO: implement properly, without the all_domain_grounded_predicates parameter which i mistakenly added as a part of the previous function's impolementation
        """
        :param observation:
        :return:
        """
        observation.components[0].previous_state = self._ground_all_predicates_in_state(
            observation.components[0].previous_state,
            all_domain_grounded_predicates
        )

        # for two consecutive components c_(i) and c_(i+1), it holds that c_(i).next_state == c_(i+1).prev_state,
        # so they should be masked in the same way.
        for i in range(len(observation.components) - 1):
            curr_component, next_component = observation.components[i], observation.components[i + 1]
            state_with_all_grounded_literals = self._ground_all_predicates_in_state(curr_component.next_state,
                                                                                    all_domain_grounded_predicates)
            curr_component.next_state = state_with_all_grounded_literals
            next_component.previous_state = state_with_all_grounded_literals

        observation.components[-1].next_state = self._ground_all_predicates_in_state(
            observation.components[-1].next_state, all_domain_grounded_predicates)

        return observation

    def mask_trajectory(self, trajectory_file_path: Path, domain: Domain,
                        problem_path: Path) -> Tuple[Path, Dict[int, Set[GroundedPredicate]]]:
        """
        Masks the predicates in the trajectory file according to the PI-SAM algorithm.
        :param trajectory_file_path: The path to the trajectory file to be masked.
        :param domain: The domain of the problem, which contains the predicates to be masked.
        :param problem_path: The path to problem file related to the trajectory
        :return: A tuple containing the path to the masked trajectory file and a mapping of masked predicates.
        """
        problem: Problem = ProblemParser(problem_path, domain).parse_problem()
        observation: Observation = TrajectoryParser(domain, problem).parse_trajectory(trajectory_file_path)
        masked_observation, masked_predicates_mapping = self.mask_observation(observation)
        masked_trajectory_path = trajectory_file_path.with_suffix('.masked.trajectory')
        output_path = observation_to_trajectory_file(masked_observation, masked_trajectory_path)

        return output_path, masked_predicates_mapping

    def prepare_observations(self, observations: List[Observation]) -> List[Observation]:
        """
        For each grounded literal in the domain, create a proper instance in the observations' states.

        :param observations:
        :return:
        """
        return [self._ground_all_states_in_observation(observation) for observation in observations]

    def learn_action_model(self, observations: List[Observation]) -> Tuple[LearnerDomain, Dict[str, str]]:
        observations = self.prepare_observations(observations)
        return super().learn_action_model(
            observations)  # TODO: check if we need to override this method or not (or maybe its inner methods)
