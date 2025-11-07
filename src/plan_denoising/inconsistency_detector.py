"""Inconsistency detector for finding conflicts in PDDL trajectories."""

from pathlib import Path
from typing import List, Set, Tuple, Dict
from collections import defaultdict
from dataclasses import dataclass

from pddl_plus_parser.lisp_parsers import TrajectoryParser
from pddl_plus_parser.models import Observation, ObservedComponent, State, Domain, GroundedPredicate

from src.plan_denoising.conflict_tree import Inconsistency
from src.utils.pddl import get_state_grounded_predicates


@dataclass
class Transition:
    """
    Represents a single transition in a trajectory.

    Attributes:
        index: Index of this transition in the trajectory
        prev_state: Set of fluent strings in the previous state
        action: Name of the grounded action
        next_state: Set of fluent strings in the next state
    """
    index: int
    prev_state: Set[str]
    action: str
    next_state: Set[str]

    def get_prev_state_signature(self) -> frozenset:
        """Get a hashable signature of the previous state."""
        return frozenset(self.prev_state)


class InconsistencyDetector:
    """
    Detects inconsistencies in PDDL trajectories.

    An inconsistency occurs when two transitions have:
    1. The same action
    2. The same previous state (same set of fluents)
    3. Different next states (at least one fluent differs)

    This violates determinism: the same action in the same state should
    always lead to the same next state.
    """

    def __init__(self, domain: Domain):
        self.trajectory_parser = TrajectoryParser(partial_domain=domain)

    def load_trajectory(self, trajectory_path: Path) -> Observation:
        """
        Load a PDDL trajectory file.

        :param trajectory_path: Path to the .trajectory file
        :return: Observation object containing the trajectory
        """
        return self.trajectory_parser.parse_trajectory(trajectory_path)

    def extract_transitions(self, observation: Observation) -> List[Transition]:
        """
        Extract transitions from an observation.

        :param observation: Observation object from trajectory parser
        :return: List of transitions
        """
        transitions = []

        for idx, component in enumerate(observation.components):
            # Extract fluent strings from states
            prev_state_fluents = self._extract_fluents_from_state(component.previous_state)
            next_state_fluents = self._extract_fluents_from_state(component.next_state)

            # Extract action name
            action_name = str(component.grounded_action_call)

            transition = Transition(
                index=idx,
                prev_state=prev_state_fluents,
                action=action_name,
                next_state=next_state_fluents
            )
            transitions.append(transition)

        return transitions

    @staticmethod
    def _extract_fluents_from_state(state: State) -> Set[str]:
        """
        Extract fluent strings from a state.

        :param state: State object
        :return: Set of fluent strings (only positive, unmasked fluents)
        """

        grounded_predicates: Set[GroundedPredicate] = get_state_grounded_predicates(state)
        return {pred.untyped_representation for pred in grounded_predicates if pred.is_positive and not pred.is_masked}

    def find_inconsistencies(
        self,
        transitions: List[Transition]
    ) -> List[Inconsistency]:
        """
        Find all inconsistencies in a list of transitions.

        An inconsistency is a pair of transitions (t1, t2) where:
        - t1.action == t2.action
        - t1.prev_state == t2.prev_state
        - t1.next_state != t2.next_state (at least one fluent differs)

        :param transitions: List of transitions from a trajectory
        :return: List of detected inconsistencies
        """
        inconsistencies = []

        # Group transitions by (action, prev_state_signature)
        transition_groups: Dict[Tuple[str, frozenset], List[Transition]] = defaultdict(list)

        for trans in transitions:
            key = (trans.action, trans.get_prev_state_signature())
            transition_groups[key].append(trans)

        # For each group with multiple transitions, check for inconsistencies
        for (action, prev_state_sig), group_transitions in transition_groups.items():
            if len(group_transitions) < 2:
                continue  # No inconsistency possible with only one transition

            # Compare all pairs in the group
            for i in range(len(group_transitions)):
                for j in range(i + 1, len(group_transitions)):
                    trans1 = group_transitions[i]
                    trans2 = group_transitions[j]

                    # Find fluents that differ between next states
                    conflicting_fluents = self._find_conflicting_fluents(
                        trans1.next_state,
                        trans2.next_state
                    )

                    # Create an inconsistency for each conflicting fluent
                    for fluent in conflicting_fluents:
                        inconsistency = Inconsistency(
                            transition1_index=trans1.index,
                            transition2_index=trans2.index,
                            action_name=action,
                            conflicting_fluent=fluent,
                            fluent_in_trans1_next=fluent in trans1.next_state,
                            fluent_in_trans2_next=fluent in trans2.next_state
                        )
                        inconsistencies.append(inconsistency)

        return inconsistencies

    @staticmethod
    def _find_conflicting_fluents(
        state1: Set[str],
        state2: Set[str]
    ) -> Set[str]:
        """
        Find fluents that differ between two states.

        :param state1: First state (set of fluent strings)
        :param state2: Second state (set of fluent strings)
        :return: Set of fluents that are in one state but not the other
        """
        # Symmetric difference: fluents in state1 XOR state2
        return state1.symmetric_difference(state2)

    def detect_inconsistencies_from_observation(
        self,
        observation: Observation
    ) -> List[Inconsistency]:
        """
        Detect inconsistencies directly from an Observation object.

        :param observation: Observation object
        :return: List of detected inconsistencies
        """
        # Extract transitions
        transitions: List[Transition] = self.extract_transitions(observation)

        # Find inconsistencies
        inconsistencies = self.find_inconsistencies(transitions)

        return inconsistencies

    def detect_inconsistencies_in_trajectory(
        self,
        trajectory_path: Path
    ) -> List[Inconsistency]:
        """
        Main method: load a trajectory and detect all inconsistencies.

        :param trajectory_path: Path to the .trajectory file
        :return: List of detected inconsistencies
        """
        # Load trajectory
        observation: Observation = self.load_trajectory(trajectory_path)

        return self.detect_inconsistencies_from_observation(observation)

    def print_inconsistencies(self, inconsistencies: List[Inconsistency]) -> None:
        """
        Print inconsistencies in a human-readable format.

        :param inconsistencies: List of inconsistencies to print
        """
        if not inconsistencies:
            print("No inconsistencies detected!")
            return

        print(f"\nDetected {len(inconsistencies)} inconsistencies:\n")

        for idx, incons in enumerate(inconsistencies, 1):
            print(f"{idx}. {incons}")
            print(f"   Transitions: {incons.transition1_index} vs {incons.transition2_index}")
            print(f"   Action: {incons.action_name}")
            print(f"   Conflicting fluent: {incons.conflicting_fluent}")
            print(f"     - In transition {incons.transition1_index}'s next_state: {incons.fluent_in_trans1_next}")
            print(f"     - In transition {incons.transition2_index}'s next_state: {incons.fluent_in_trans2_next}")
            print()
