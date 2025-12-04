# patches_and_conflicts.py

from enum import Enum
from typing import Tuple

from pddl_plus_parser.models import Predicate, ActionCall, GroundedPredicate


class PatchOperation(Enum):
    """Operation of a model-level patch."""
    FORBID = "forbid"
    REQUIRE = "require"


class ModelPart(Enum):
    """Which part of the model a patch refers to."""
    PRECONDITION = "pre"
    EFFECT = "eff"


class ParameterBoundLiteral:
    """
    Lifted literal with parameter names, e.g.:

      clear(?x)
      on(?x, ?y)
      not clear(?x)
    """
    def __init__(
        self,
        predicate_name: str,
        parameters: Tuple[str, ...],
        is_positive: bool = True,
    ):
        self.predicate_name = predicate_name
        self.parameters = parameters
        self.is_positive = is_positive

    def matches(self, lifted_predicate: Predicate) -> bool:
        """
        Check if a lifted Predicate from pddl_plus_parser matches this PBL.
        We expect lifted_predicate to have:
          - .name
          - .is_positive
          - .signature (list of parameters with .name)
        """
        if lifted_predicate.name != self.predicate_name:
            return False
        if getattr(lifted_predicate, "is_positive", True) != self.is_positive:
            return False

        pred_params = tuple(param_name for param_name, param_type in lifted_predicate.signature.items())
        return pred_params == self.parameters

    def __hash__(self) -> int:
        return hash((self.predicate_name, self.parameters, self.is_positive))

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, ParameterBoundLiteral)
            and self.predicate_name == other.predicate_name
            and self.parameters == other.parameters
            and self.is_positive == other.is_positive
        )

    def __str__(self) -> str:
        params = ", ".join(self.parameters)
        base = f"{self.predicate_name}({params})"
        return base if self.is_positive else f"not {base}"


# TODO: check if we should use the Action of pddl-plus-parser instead of just action name (to bound the PBL properly)
class ModelLevelPatch:
    """
    Model-level patch: constrain the learned model for a given action.

    Model-level patch for an action:

      - (EFFECT, FORBID)   : cannot be effect
      - (EFFECT, REQUIRE)  : must be effect
      - (PRECONDITION, FORBID): cannot be precondition
      - (PRECONDITION, REQUIRE): not supported (no SAM rule for 'must precondition')
    """
    def __init__(
        self,
        action_name: str,
        model_part: ModelPart,
        pbl: ParameterBoundLiteral,
        operation: PatchOperation,
    ):
        self.action_name = action_name
        self.model_part = model_part
        self.pbl = pbl
        self.operation = operation

    def __hash__(self) -> int:
        return hash((self.action_name, self.model_part, self.pbl, self.operation))

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, ModelLevelPatch)
            and self.action_name == other.action_name
            and self.model_part == other.model_part
            and self.pbl == other.pbl
            and self.operation == other.operation
        )

    def __str__(self) -> str:
        return (
            f"ModelPatch({self.operation.value} {self.pbl} "
            f"in {self.model_part.value} of {self.action_name})"
        )


class FluentLevelPatch:
    """
    Flip a fluent in a specific trajectory transition.

    observation_index: which trajectory/observation in the dataset.
    component_index:   which (s,a,s') component within that trajectory.
    state_type:        'prev' or 'next' – which state in that component to flip.
    fluent:            grounded fluent string, e.g. "holding(a)".
    """
    def __init__(
        self,
        observation_index: int,
        component_index: int,
        state_type: str,
        fluent: str,
    ):
        self.observation_index = observation_index
        self.component_index = component_index
        self.state_type = state_type
        self.fluent = fluent

    def __hash__(self) -> int:
        return hash(
            (self.observation_index, self.component_index, self.state_type, self.fluent)
        )

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, FluentLevelPatch)
            and self.observation_index == other.observation_index
            and self.component_index == other.component_index
            and self.state_type == other.state_type
            and self.fluent == other.fluent
        )

    def __str__(self) -> str:
        return (
            f"FluentPatch(obs={self.observation_index}, comp={self.component_index}, "
            f"{self.state_type}, {self.fluent})"
        )


class ConflictType(Enum):
    """
    Conflicts induced by the 3 patch types:

      - FORBID_EFFECT_VS_MUST:
            cannot-be-effect patch vs SAM's must-be-effect
            (discrete add/delete).

      - REQUIRE_EFFECT_VS_CANNOT:
            must-be-effect patch vs SAM's cannot-be-effect
            (cannot_be_effects).

      - FORBID_PRECOND_VS_IS:
            cannot-be-precondition patch vs SAM's attempt to
            treat a literal as a precondition (via PI-SAM rule).
    """
    FORBID_EFFECT_VS_MUST = "forbid_effect_vs_must"
    REQUIRE_EFFECT_VS_CANNOT = "require_effect_vs_cannot"
    FORBID_PRECOND_VS_IS = "forbid_precondition_vs_is_precondition"
    FRAME_AXIOM = "frame_axiom"


class Conflict:
    """
    Conflict between model patches and what PI-SAM wants to learn.

    All conflicts are:
      - localized to a transition via (observation_index, component_index).
      - grounded_fluent is ALWAYS a grounded predicate string (e.g. "holding(a)")
        taken from the actual transition (effects / previous state)  or from grounding the PBL with action args.
    """
    def __init__(
        self,
        action_name: str,
        pbl: ParameterBoundLiteral,
        conflict_type: ConflictType,
        observation_index: int,
        component_index: int,
        grounded_fluent: str,
    ):
        self.action_name = action_name
        self.pbl = pbl
        self.conflict_type = conflict_type
        self.observation_index = observation_index
        self.component_index = component_index
        self.grounded_fluent = grounded_fluent

    def __str__(self) -> str:
        return (
            f"Conflict({self.conflict_type.value}: {self.grounded_fluent} vs {self.pbl} "
            f"in {self.action_name} at obs[{self.observation_index}][{self.component_index}])"
        )

    def __repr__(self) -> str:
        return self.__str__()
