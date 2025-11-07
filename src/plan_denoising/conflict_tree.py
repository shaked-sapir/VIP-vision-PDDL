"""Conflict tree for tracking repair decisions during plan denoising."""

from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class RepairChoice(Enum):
    """Which transition was repaired in a conflict."""
    FIRST = "first"
    SECOND = "second"


@dataclass
class RepairOperation:
    """
    Represents a single repair operation on a trajectory state.

    Attributes:
        transition_index: Index of the transition that was repaired
        state_type: Which state was repaired ('prev_state' or 'next_state')
        fluent_changed: The fluent that was modified
        old_value: Previous truth value of the fluent
        new_value: New truth value of the fluent
    """
    transition_index: int
    state_type: str  # 'prev_state' or 'next_state'
    fluent_changed: str
    old_value: bool  # True if fluent was present, False if absent
    new_value: bool

    def __str__(self):
        action = "added" if self.new_value else "removed"
        return (f"Repair(trans={self.transition_index}, state={self.state_type}, "
                f"{action} fluent '{self.fluent_changed}')")


@dataclass
class Inconsistency:
    """
    Represents an inconsistency between two transitions.

    Two transitions are inconsistent if they have the same action and same prev_state,
    but different next_states with respect to some fluent.

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
        return (f"Inconsistency(transitions=[{self.transition1_index}, {self.transition2_index}], "
                f"action='{self.action_name}', fluent='{self.conflicting_fluent}')")


class ConflictNode:
    """
    Node in the conflict resolution tree.

    Each node represents a decision point where we choose which transition to repair
    to resolve an inconsistency.

    Attributes:
        inconsistency: The inconsistency being resolved at this node
        repair_operation: The repair that was performed at this node
        repair_choice: Which transition was chosen for repair (FIRST or SECOND)
        parent: Parent node (None for root)
        children: Child nodes (subsequent repair decisions)
        pi_sam_result: Result of running PI-SAM after this repair (optional)
    """

    def __init__(
        self,
        inconsistency: Inconsistency,
        repair_operation: RepairOperation,
        repair_choice: RepairChoice,
        parent: Optional['ConflictNode'] = None
    ):
        self.inconsistency = inconsistency
        self.repair_operation = repair_operation
        self.repair_choice = repair_choice
        self.parent = parent
        self.children: List[ConflictNode] = []
        self.pi_sam_result = None  # Will store learned domain after PI-SAM

    def add_child(self, child: 'ConflictNode') -> None:
        """Add a child node representing a subsequent repair decision."""
        self.children.append(child)

    def get_path_from_root(self) -> List['ConflictNode']:
        """Get the path from the root to this node."""
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))

    def get_all_repairs(self) -> List[RepairOperation]:
        """Get all repair operations from root to this node."""
        path = self.get_path_from_root()
        return [node.repair_operation for node in path]

    def __str__(self):
        return (f"ConflictNode(inconsistency={self.inconsistency}, "
                f"choice={self.repair_choice.value}, "
                f"repair={self.repair_operation})")


class ConflictTree:
    """
    Tree structure for tracking repair decisions during plan denoising.

    The tree allows us to:
    1. Track which repairs have been attempted
    2. Backtrack when a repair path doesn't work
    3. Try alternative repairs (repairing the other transition)

    Each node represents a repair decision, and edges represent the sequence
    of repairs leading to that state.
    """

    def __init__(self):
        self.root: Optional[ConflictNode] = None
        self.current_node: Optional[ConflictNode] = None

    def add_repair(
        self,
        inconsistency: Inconsistency,
        repair_operation: RepairOperation,
        repair_choice: RepairChoice
    ) -> ConflictNode:
        """
        Add a new repair decision to the tree.

        :param inconsistency: The inconsistency being resolved
        :param repair_operation: The repair operation performed
        :param repair_choice: Which transition was repaired
        :return: The newly created node
        """
        new_node = ConflictNode(
            inconsistency=inconsistency,
            repair_operation=repair_operation,
            repair_choice=repair_choice,
            parent=self.current_node
        )

        if self.root is None:
            self.root = new_node
        else:
            self.current_node.add_child(new_node)

        self.current_node = new_node
        return new_node

    def backtrack(self) -> Optional[ConflictNode]:
        """
        Backtrack to the parent node to try an alternative repair.

        :return: The parent node, or None if already at root
        """
        if self.current_node is None:
            return None

        parent = self.current_node.parent
        self.current_node = parent
        return parent

    def get_current_repairs(self) -> List[RepairOperation]:
        """Get all repairs from root to current node."""
        if self.current_node is None:
            return []
        return self.current_node.get_all_repairs()

    def has_unexplored_alternative(self) -> bool:
        """
        Check if the current node's inconsistency has an unexplored alternative repair.

        If we repaired transition1 before, we can try repairing transition2 instead.
        """
        if self.current_node is None:
            return False

        # Check if we've already tried both repair choices at this level
        parent = self.current_node.parent
        if parent is None:
            # At root, check if we've only tried one choice
            return len(self.root.children) == 0  # If root has no siblings, we can try the other choice

        # Check if parent has tried both repair choices for the same inconsistency
        siblings = parent.children
        repair_choices_tried = {node.repair_choice for node in siblings}

        # If we've only tried one choice, we can try the other
        return len(repair_choices_tried) == 1

    def get_alternative_repair_choice(self) -> Optional[RepairChoice]:
        """
        Get the alternative repair choice that hasn't been tried yet.

        :return: The untried RepairChoice, or None if both have been tried
        """
        if not self.has_unexplored_alternative():
            return None

        current_choice = self.current_node.repair_choice
        if current_choice == RepairChoice.FIRST:
            return RepairChoice.SECOND
        else:
            return RepairChoice.FIRST

    def visualize(self, node: Optional[ConflictNode] = None, indent: int = 0) -> str:
        """
        Create a string visualization of the tree.

        :param node: Node to start from (default: root)
        :param indent: Current indentation level
        :return: String representation of the tree
        """
        if node is None:
            node = self.root

        if node is None:
            return "Empty tree"

        lines = []
        prefix = "  " * indent

        # Node info
        lines.append(f"{prefix}├─ {node}")
        lines.append(f"{prefix}│  Repair: {node.repair_operation}")

        # Children
        for child in node.children:
            lines.append(self.visualize(child, indent + 1))

        return "\n".join(lines)

    def __str__(self):
        return self.visualize()
