from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Tuple, Optional

from pddl_plus_parser.models import Predicate


class PatchOperation(Enum):
    """Operation of a model-level patch."""
    FORBID = "forbid"
    REQUIRE = "require"


class ModelPart(Enum):
    """Which part of the model a patch refers to."""
    PRECONDITION = "pre"
    EFFECT = "eff"
    OTHER = "other"  # this will be used for conflict grouping


@dataclass(frozen=True)
class ParameterBoundLiteral:
    """
    Lifted literal with parameter names, e.g.:

      clear(?x)
      on(?x, ?y)
      not clear(?x)
    """
    predicate_name: str
    parameters: Tuple[str, ...]
    is_positive: bool = True

    @classmethod
    def from_lifted_predicate(cls, pred: Predicate) -> ParameterBoundLiteral:
        """Build a PBL from a pddl_plus_parser lifted Predicate."""
        return cls(
            predicate_name=pred.name,
            parameters=tuple(pred.signature.keys()),
            is_positive=getattr(pred, "is_positive", True),
        )

    def matches(self, lifted_predicate: Predicate) -> bool:
        if lifted_predicate.name != self.predicate_name:
            return False
        if getattr(lifted_predicate, "is_positive", True) != self.is_positive:
            return False
        pred_params = tuple(lifted_predicate.signature.keys())
        return pred_params == self.parameters

    def __str__(self) -> str:
        params = ", ".join(self.parameters)
        base = f"{self.predicate_name}({params})"
        return base if self.is_positive else f"not {base}"


@dataclass(frozen=True)
class ModelLevelPatch:
    """
    Model-level patch: constrain the learned model for a given action.

      - (EFFECT, FORBID)   : cannot be effect
      - (EFFECT, REQUIRE)  : must be effect
      - (PRECONDITION, FORBID): cannot be precondition

    """
    action_name: str
    model_part: ModelPart
    pbl: ParameterBoundLiteral
    operation: PatchOperation

    def __str__(self) -> str:
        return (
            f"ModelPatch({self.operation.value} {self.pbl} "
            f"in {self.model_part.value} of {self.action_name})"
        )


@dataclass(frozen=True)
class FluentLevelPatch:
    """
    Flip a fluent in a specific trajectory transition.

    observation_index: which trajectory/observation in the dataset.
    component_index:   which (s,a,s') component within that trajectory.
    state_type:        'prev' or 'next' – which state in that component to flip.
    fluent:            grounded fluent string, e.g. "(holding a)".
    """
    observation_index: int
    component_index: int
    state_type: str
    fluent: str

    @property
    def normalized_fluent(self) -> str:
        """Canonical positive form: strips '(not ...)' wrapper if present."""
        f = self.fluent
        if f.startswith("(not ") and f.endswith(")"):
            return f[5:-1]
        return f

    def __str__(self) -> str:
        return (
            f"FluentPatch(obs={self.observation_index}, comp={self.component_index}, "
            f"{self.state_type}, {self.fluent})"
        )


class ConflictType(Enum):
    """
    Conflict types:

      - FORBID_EFFECT_VS_MUST:
            cannot-be-effect constraint vs SAM's must-be-effect (discrete add/delete).

      - REQUIRE_EFFECT_VS_CANNOT:
            must-be-effect constraint vs SAM's cannot-be-effect.

      - FORBID_PRECOND_VS_IS:
            cannot-be-precondition constraint vs SAM's precondition.

      - FRAME_AXIOM:
            predicate changes without sharing objects with the action.
    """
    FORBID_EFFECT_VS_MUST = "forbid_effect_vs_must"
    REQUIRE_EFFECT_VS_CANNOT = "require_effect_vs_cannot"
    FORBID_PRECOND_VS_IS = "forbid_precondition_vs_is_precondition"
    FRAME_AXIOM = "frame_axiom"


class ConflictPriority(IntEnum):
    """Smaller = higher priority."""
    EFFECT = 1
    FRAME_AXIOM = 0
    OTHER = 2


@dataclass(frozen=True)
class Conflict:
    """
    Conflict between model constraints and what SAM/PI-SAM wants to learn.

    Localized to a transition via (observation_index, component_index).
    grounded_fluent is always a grounded predicate string (e.g. "(holding a)").
    """
    action_name: str
    pbl: ParameterBoundLiteral
    conflict_type: ConflictType
    observation_index: int
    component_index: int
    grounded_fluent: str
    frame_is_add: Optional[bool] = None
    source_is_gt: bool = True  # Whether the source (prev) state of this transition is ground truth.

    def __str__(self) -> str:
        extra = ""
        if self.conflict_type == ConflictType.FRAME_AXIOM and self.frame_is_add is not None:
            kind = "add" if self.frame_is_add else "del"
            gt_tag = "GT" if self.source_is_gt else "non-GT"
            extra = f", frame_change={kind}, src={gt_tag}"
        return (
            f"Conflict({self.conflict_type.value}: {self.grounded_fluent} vs {self.pbl} "
            f"in {self.action_name} at obs[{self.observation_index}][{self.component_index}]{extra})"
        )

    def __repr__(self) -> str:
        return self.__str__()
