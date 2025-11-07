"""Frame axiom violation detector."""

from dataclasses import dataclass
from typing import List

from .base_detector import BaseDetector, Transition


@dataclass
class FrameAxiomViolation:
    """
    Represents a violation of the frame axiom.

    A fluent changed value without being an effect of the action:
    - False→True without being an add effect, OR
    - True→False without being a delete effect

    Attributes:
        transition_index: The index of the transition where the violation occurred
        action_name: The name of the action
        fluent: The fluent that violates the frame axiom
        violation_type: Either 'added' (false→true) or 'deleted' (true→false)
    """
    transition_index: int
    action_name: str
    fluent: str
    violation_type: str  # 'added' or 'deleted'

    def __str__(self):
        if self.violation_type == 'added':
            return (f"FrameAxiomViolation at transition {self.transition_index}: "
                    f"Fluent '{self.fluent}' became true after action '{self.action_name}' "
                    f"but is not an add effect.")
        else:
            return (f"FrameAxiomViolation at transition {self.transition_index}: "
                    f"Fluent '{self.fluent}' became false after action '{self.action_name}' "
                    f"but is not a delete effect.")


class FrameAxiomDetector(BaseDetector):
    """
    Detects violations of the frame axiom.

    Frame axiom: fluents not affected by an action should remain unchanged.
    A violation occurs when:
    1. A fluent becomes true without being an add effect, OR
    2. A fluent becomes false without being a delete effect
    """

    def detect(self, transitions: List[Transition]) -> List[FrameAxiomViolation]:
        """
        Detect frame axiom violations in the given transitions.

        :param transitions: List of transitions from a trajectory
        :return: List of detected frame axiom violations
        """
        violations = []

        for transition in transitions:
            # Check for fluents that became true (false → true)
            added_fluents = transition.next_state - transition.prev_state
            for fluent in added_fluents:
                if fluent not in transition.add_effects:
                    violation = FrameAxiomViolation(
                        transition_index=transition.index,
                        action_name=transition.action,
                        fluent=fluent,
                        violation_type='added'
                    )
                    violations.append(violation)

            # Check for fluents that became false (true → false)
            deleted_fluents = transition.prev_state - transition.next_state
            for fluent in deleted_fluents:
                if fluent not in transition.delete_effects:
                    violation = FrameAxiomViolation(
                        transition_index=transition.index,
                        action_name=transition.action,
                        fluent=fluent,
                        violation_type='deleted'
                    )
                    violations.append(violation)

        return violations

    def print_violations(self, violations: List[FrameAxiomViolation]) -> None:
        """
        Print frame axiom violations in a human-readable format.

        :param violations: List of frame axiom violations to print
        """
        if not violations:
            print("No frame axiom violations detected!")
            return

        print(f"\nDetected {len(violations)} frame axiom violations:\n")

        for idx, violation in enumerate(violations, 1):
            print(f"{idx}. {violation}")
            print()
