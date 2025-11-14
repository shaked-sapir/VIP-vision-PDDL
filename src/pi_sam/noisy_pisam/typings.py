from enum import Enum

from typing import Tuple, List

from pddl_plus_parser.models import Predicate, GroundedPredicate

from src.utils.pddl import lift_predicate


class ConflictType(Enum):
    """Type of conflict between patches and SAM rules."""
    FORBIDDEN_EFFECT = "forbidden_effect"  # Model patch forbids, but SAM wants to add
    REQUIRED_EFFECT = "required_effect"  # Model patch requires, but SAM says cannot be
    FORBIDDEN_PRECONDITION = "forbidden_precondition"  # Model patch forbids, but SAM wants to add
    REQUIRED_PRECONDITION = "required_precondition"  # Model patch requires, but SAM wants to remove


class ModelPart(Enum):
    """Part of the action model."""
    PRECONDITION = "pre"
    EFFECT = "eff"


class PatchOperation(Enum):
    """Operation in a model-level patch."""
    FORBID = "forbid"  # Don't allow this PBL in the model
    REQUIRE = "require"  # Require this PBL in the model


class ParameterBoundLiteral:
    """
    A lifted literal with parameter names (e.g., at(?x) for action move(?x, ?y)).
    """
    def __init__(self, predicate_name: str, parameters: Tuple[str, ...], is_positive: bool = True):
        self.predicate_name = predicate_name
        self.parameters = parameters  # e.g., ("?x",) or ("?x", "?y")
        self.is_positive = is_positive

    def matches(self, lifted_predicate: Predicate) -> bool:
        """Check if a lifted predicate matches this PBL."""
        if lifted_predicate.name != self.predicate_name:
            return False
        if lifted_predicate.is_positive != self.is_positive:
            return False
        # Check if parameter names match
        pred_params = tuple(param_name for param_name, param_type in lifted_predicate.signature.items())
        return pred_params == self.parameters

    def matches_grounded(self, grounded_predicate: GroundedPredicate) -> bool:
        """Check if a grounded predicate matches this PBL by ignoring specific objects."""
        return self.matches(lift_predicate(grounded_predicate))

    def get_grounded_candidate(self, grounded_predicates: Set[GroundedPredicate]) -> GroundedPredicate:
        """Get a grounded predicate from a list that matches this PBL."""
        for gp in grounded_predicates:
            if self.matches_grounded(gp):
                return gp
        raise ValueError(f"No grounded predicate matches PBL: {self}")

    def __str__(self):
        params_str = ", ".join(self.parameters)
        literal_str = f"{self.predicate_name}({params_str})"
        return literal_str if self.is_positive else f"not {literal_str}"

    def __hash__(self):
        return hash((self.predicate_name, self.parameters, self.is_positive))

    def __eq__(self, other):
        return (isinstance(other, ParameterBoundLiteral) and
                self.predicate_name == other.predicate_name and
                self.parameters == other.parameters and
                self.is_positive == other.is_positive)


class FluentLevelPatch:
    """Patch that flips a fluent value in a specific observation state."""
    def __init__(self, observation_index: int, component_index: int,
                 state_type: str, fluent: str):
        """
        :param observation_index: Index of observation in the list
        :param component_index: Index of component within the observation
        :param state_type: 'previous' or 'next'
        :param fluent: The fluent string to flip (e.g., "on(a, b)")
        """
        self.observation_index = observation_index
        self.component_index = component_index
        self.state_type = state_type  # 'previous' or 'next'
        self.fluent = fluent

    def __str__(self):
        return (f"FlipFluent(obs={self.observation_index}, comp={self.component_index}, "
                f"{self.state_type}_state, fluent={self.fluent})")

    def __hash__(self):
        return hash((self.observation_index, self.component_index,
                    self.state_type, self.fluent))

    def __eq__(self, other):
        return (isinstance(other, FluentLevelPatch) and
                self.observation_index == other.observation_index and
                self.component_index == other.component_index and
                self.state_type == other.state_type and
                self.fluent == other.fluent)


class ModelLevelPatch:
    """Patch that constrains the learned action model."""
    def __init__(self, action_name: str, model_part: ModelPart,
                 pbl: ParameterBoundLiteral, operation: PatchOperation):
        self.action_name = action_name
        self.model_part = model_part
        self.pbl = pbl
        self.operation = operation

    def __str__(self):
        return (f"ModelPatch({self.operation.value} {self.pbl} "
                f"in {self.model_part.value} of {self.action_name})")

    def __hash__(self):
        return hash((self.action_name, self.model_part, self.pbl, self.operation))

    def __eq__(self, other):
        return (isinstance(other, ModelLevelPatch) and
                self.action_name == other.action_name and
                self.model_part == other.model_part and
                self.pbl == other.pbl and
                self.operation == other.operation)


class Conflict:
    """A conflict detected during learning with patches."""
    def __init__(self, action_name: str, pbl: ParameterBoundLiteral,
                 conflict_type: ConflictType, observation_index: int,
                 component_index: int, grounded_fluent: str):
        self.action_name = action_name
        self.pbl = pbl
        self.conflict_type = conflict_type
        self.observation_index = observation_index
        self.component_index = component_index
        self.grounded_fluent = grounded_fluent

    def __str__(self):
        return (f"Conflict({self.conflict_type.value}: {self.grounded_fluent} vs {self.pbl} "
                f"in {self.action_name} at obs[{self.observation_index}][{self.component_index}])")
