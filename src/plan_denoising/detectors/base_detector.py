"""Base detector interface for trajectory inconsistency detection."""

from abc import ABC, abstractmethod
from typing import List, Any
from dataclasses import dataclass
from pathlib import Path

from pddl_plus_parser.models import Domain, Observation


@dataclass
class Transition:
    """
    Represents a single transition in a trajectory.

    Attributes:
        index: Index of this transition in the trajectory
        prev_state: Set of fluent strings in the previous state
        action: Name of the grounded action
        next_state: Set of fluent strings in the next state
        action_name: Lifted action name (e.g., "stack")
        parameters: Dictionary mapping parameter names to grounded objects
        add_effects: Set of fluent strings that are add effects of the action
        delete_effects: Set of fluent strings that are delete effects of the action
    """
    index: int
    prev_state: set[str]
    action: str  # Full grounded action string
    next_state: set[str]
    action_name: str  # Lifted action name
    parameters: dict[str, str]  # Parameter mapping
    add_effects: set[str]
    delete_effects: set[str]

    def get_prev_state_signature(self) -> frozenset:
        """Get a hashable signature of the previous state."""
        return frozenset(self.prev_state)


class BaseDetector(ABC):
    """
    Base class for all inconsistency detectors.

    Each detector type should extend this class and implement the detect() method
    to check for a specific type of inconsistency or violation.
    """

    def __init__(self, domain: Domain):
        """
        Initialize the detector.

        :param domain: PDDL domain containing action definitions and effects
        """
        self.domain = domain

    @abstractmethod
    def detect(self, transitions: List[Transition]) -> List[Any]:
        """
        Detect inconsistencies in the given transitions.

        :param transitions: List of transitions from a trajectory
        :return: List of detected violations/inconsistencies
        """
        pass

    @abstractmethod
    def print_violations(self, violations: List[Any]) -> None:
        """
        Print violations in a human-readable format.

        :param violations: List of violations to print
        """
        pass
