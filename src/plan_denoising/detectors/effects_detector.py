"""Determinism violation detector."""

from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import defaultdict

from .base_detector import BaseDetector, Transition


@dataclass
class EffectsViolation:
    """
    Represents an is-effect violation (inconsistency) between two transitions having the same ground action.

    Two transitions violate effects if they have some fluent in both their prev_state, which is being preserved
    in one transition but gets flipped in the second one.

    Attributes:
        transition1_index: Index of first transition
        transition2_index: Index of second transition
        action_name: Name of the action in both transitions
        conflicting_fluent: The fluent that differs between next_states
        fluent_in_trans1_next: Whether fluent is in transition1's next_state
        fluent_in_trans2_next: Whether fluent is in transition2's next_state
    """
    transition1_index: int
    transition2_index: int
    action_name: str
    conflicting_fluent: str
    fluent_in_trans1_next: bool
    fluent_in_trans2_next: bool

    def __str__(self):
        return (f"EffectsViolation(transitions=[{self.transition1_index}, {self.transition2_index}], "
                f"action='{self.action_name}', fluent='{self.conflicting_fluent}')")


class EffectsDetector(BaseDetector):
    """
    Detects violations of determinism in trajectories.

    A determinism violation occurs when two transitions have:
    1. The same action
    2. The same previous state (same set of fluents)
    3. Different next states (at least one fluent differs)

    This violates the assumption that actions are deterministic: the same action
    in the same state should always lead to the same next state.
    """

    def detect(self, transitions: List[Transition]) -> List[EffectsViolation]:
        """
        Detect determinism violations in the given transitions.

        :param transitions: List of transitions from a trajectory
        :return: List of detected determinism violations
        """
        violations = []

        # Group transitions by (action, prev_state_signature)
        transition_groups: Dict[Tuple[str, frozenset], List[Transition]] = defaultdict(list)

        for trans in transitions:
            key = (trans.action, trans.get_prev_state_signature())
            transition_groups[key].append(trans)

        # For each group with multiple transitions, check for violations
        for (action, prev_state_sig), group_transitions in transition_groups.items():
            if len(group_transitions) < 2:
                continue  # No violation possible with only one transition

            # Compare all pairs in the group
            for i in range(len(group_transitions)):
                for j in range(i + 1, len(group_transitions)):
                    trans1 = group_transitions[i]
                    trans2 = group_transitions[j]

                    # Find fluents that differ between next states
                    conflicting_fluents = self._find_next_state_conflicts(trans1, trans2)

                    # Create a violation for each conflicting fluent
                    for fluent in conflicting_fluents:
                        violation = EffectsViolation(
                            transition1_index=trans1.index,
                            transition2_index=trans2.index,
                            action_name=action,
                            conflicting_fluent=fluent,
                            fluent_in_trans1_next=fluent in trans1.next_state,
                            fluent_in_trans2_next=fluent in trans2.next_state
                        )
                        violations.append(violation)

        return violations

    @staticmethod
    def _find_next_state_conflicts(trans1: Transition, trans2: Transition) -> set[str]:
        """
        Find fluents that differ between the next states of two transitions.

        This method is used to find conflicting fluents when two transitions
        have the same action and same fluent in their prev_state but different in next_states.

        :param trans1: First transition
        :param trans2: Second transition
        :return: Set of fluents that are in one next_state but not the other
        """
        # Return symmetric difference: fluents that differ between the two next_states
        next_state_fluent_diffs = trans1.next_state.symmetric_difference(trans2.next_state)
        unmasked_next_state_diffs = {fluent for fluent in next_state_fluent_diffs
                                     if fluent not in trans1.next_state_masked
                                     and fluent not in trans2.next_state_masked}

        # now handles both pos and neg as observation is completely grounded, so if not both prev_state has it,
        # at least one of them is masked out and therefore we don't consider it for violation
        # TODO: consult with Roni about it ASAP!!
        return {
            fluent for fluent in unmasked_next_state_diffs
            if (fluent in trans1.prev_state and fluent in trans2.prev_state)
        }

    def print_violations(self, violations: List[EffectsViolation]) -> None:
        """
        Print determinism violations in a human-readable format.

        :param violations: List of determinism violations to print
        """
        if not violations:
            print("No determinism violations detected!")
            return

        print(f"\nDetected {len(violations)} determinism violations:\n")

        for idx, violation in enumerate(violations, 1):
            print(f"{idx}. {violation}")
            print(f"   Transitions: {violation.transition1_index} vs {violation.transition2_index}")
            print(f"   Action: {violation.action_name}")
            print(f"   Conflicting fluent: {violation.conflicting_fluent}")
            print(f"     - In transition {violation.transition1_index}'s next_state: {violation.fluent_in_trans1_next}")
            print(f"     - In transition {violation.transition2_index}'s next_state: {violation.fluent_in_trans2_next}")
            print()
