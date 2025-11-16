"""Utility for extracting transitions from observations."""

from typing import List, Set, Tuple

from pddl_plus_parser.models import Observation, State, Domain, GroundedPredicate, Predicate

from src.plan_denoising.detectors.base_detector import Transition
from src.utils.pddl import get_state_masked_predicates, get_state_unmasked_predicates


class TransitionExtractor:
    """
    Extracts transitions from PDDL observations.

    A transition includes the previous state, action, next state, and the
    grounded effects of the action from the domain.
    """

    def __init__(self, domain: Domain):
        """
        Initialize the transition extractor.

        :param domain: PDDL domain containing action definitions
        """
        self.domain = domain

    def extract_transitions(self, observation: Observation) -> List[Transition]:
        """
        Extract transitions from an observation.

        :param observation: Observation object from trajectory parser
        :return: List of transitions
        """
        transitions = []

        for idx, component in enumerate(observation.components):
            # Extract fluent strings from states
            prev_state_fluents = self._extract_fluents_from_state(component.previous_state, masking_bit=False)
            prev_state_masked_fluents = self._extract_fluents_from_state(component.previous_state, masking_bit=True)
            next_state_fluents = self._extract_fluents_from_state(component.next_state, masking_bit=False)
            next_state_masked_fluents = self._extract_fluents_from_state(component.next_state, masking_bit=True)

            # Extract grounded action information
            grounded_action = component.grounded_action_call
            action_name = grounded_action.name
            lifted_action = self.domain.actions.get(action_name)
            parameters = {param: val for param, val in zip(lifted_action.parameter_names, grounded_action.parameters)}

            # Extract both add and delete effects from domain
            add_effects, delete_effects = self._ground_effects(
                action_name, parameters
            )

            transition = Transition(
                index=idx,
                prev_state=prev_state_fluents,
                prev_state_masked=prev_state_masked_fluents,
                action=str(grounded_action),
                next_state=next_state_fluents,
                next_state_masked=next_state_masked_fluents,
                action_name=action_name,
                parameters=parameters,
                add_effects=add_effects,
                delete_effects=delete_effects
            )
            transitions.append(transition)

        return transitions

    def _ground_effects(self, action_name: str, parameters: dict[str, str]) -> Tuple[Set[str], Set[str]]:
        """
        Ground the add and delete effects of an action using the domain.

        :param action_name: The lifted action name (e.g., "stack")
        :param parameters: Dictionary mapping parameter names to grounded objects
        :return: Tuple of (add_effects, delete_effects) as sets of fluent strings
        """
        add_effects = set()
        delete_effects = set()

        # Find the action in the domain
        action = self.domain.actions.get(action_name)
        if action is None:
            # Action not found in domain, return empty effects
            return add_effects, delete_effects

        # Ground add effects using discrete_effects
        for effect in action.discrete_effects:
            grounded_predicate = self._ground_predicate(effect, parameters)
            if effect.is_positive:
                add_effects.add(grounded_predicate)
            else:
                delete_effects.add(grounded_predicate)

        return add_effects, delete_effects

    @staticmethod
    def _ground_predicate(effect_predicate: Predicate, parameters: dict[str, str]) -> str:
        """
        Ground a predicate using the parameter mapping.

        :param effect_predicate: The effect predicate from the action
        :param parameters: Dictionary mapping parameter names to grounded objects
        :return: Grounded predicate string (e.g., "(on blue red)")
        """
        assert all(param in parameters for param in effect_predicate.signature.keys()), "Not all parameters provided for grounding."
        parameters = {param: parameters[param] for param in effect_predicate.signature.keys()}
        grounded_predicate = GroundedPredicate(name=effect_predicate.name, signature=effect_predicate.signature,
                                               object_mapping=parameters, is_positive=effect_predicate.is_positive)
        return grounded_predicate.untyped_representation

    @staticmethod
    def _extract_fluents_from_state(state: State, masking_bit: bool = False) -> Set[str]:
        """
        Extract fluent strings from a state.

        :param state: State object
        :return: Set of fluent strings (only positive, unmasked fluents)
        """
        relevant_predicates = get_state_masked_predicates(state) if masking_bit else get_state_unmasked_predicates(state)
        return {pred.untyped_representation for pred in relevant_predicates}
