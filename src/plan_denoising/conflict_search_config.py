"""Configuration object for the Conflict-Directed Patch Search (CDPS).

A single immutable container for the conflict-search "shape" parameters, so they
travel as one object instead of being hand-threaded key-by-key through the
experiment stack and stashed on a throwaway learner instance.

String values (e.g. from run_config.yaml or the CLI) are coerced to their enum
types in __post_init__, so callers may pass either strings or enum members.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Union

from utilities import NegativePreconditionPolicy

from src.plan_denoising.frontier import (
    ConflictGroupStrategy,
    FluentBranchMode,
    NodeChoosingStrategy,
    SearchMode,
)


def resolve_search_mode(value: Union[str, SearchMode]) -> SearchMode:
    if isinstance(value, SearchMode):
        return value
    normalized = str(value).strip().lower()
    if normalized == "ucs":
        return SearchMode.UCS
    if normalized == "dfs":
        return SearchMode.ANYTIME_DFS
    raise ValueError(f"Unsupported search_mode '{value}'. Expected one of: dfs, ucs.")


def resolve_node_choosing_strategy(
    value: Union[str, NodeChoosingStrategy],
) -> NodeChoosingStrategy:
    if isinstance(value, NodeChoosingStrategy):
        return value
    normalized = str(value).strip().lower()
    mapping = {
        "model_patch_first": NodeChoosingStrategy.MODEL_PATCH_FIRST,
        "fluent_patch_first": NodeChoosingStrategy.FLUENT_PATCH_FIRST,
        "fluent_patch_first_then_model": NodeChoosingStrategy.FLUENT_PATCH_FIRST_THEN_MODEL,
        "randomized": NodeChoosingStrategy.RANDOMIZED,
    }
    if normalized in mapping:
        return mapping[normalized]
    raise ValueError(
        f"Unsupported node_choosing_strategy '{value}'. Expected one of: "
        "model_patch_first, fluent_patch_first, fluent_patch_first_then_model, randomized."
    )


def resolve_conflict_group_strategy(
    value: Union[str, ConflictGroupStrategy],
) -> ConflictGroupStrategy:
    if isinstance(value, ConflictGroupStrategy):
        return value
    normalized = str(value).strip().lower()
    mapping = {
        "first": ConflictGroupStrategy.FIRST,
        "largest": ConflictGroupStrategy.LARGEST,
        "largest_model_patchable": ConflictGroupStrategy.LARGEST_MODEL_PATCHABLE,
        "most_observations": ConflictGroupStrategy.MOST_OBSERVATIONS,
        "smallest": ConflictGroupStrategy.SMALLEST,
    }
    if normalized in mapping:
        return mapping[normalized]
    raise ValueError(
        f"Unsupported conflict_group_strategy '{value}'. Expected one of: "
        "first, largest, largest_model_patchable, most_observations, smallest."
    )


def resolve_fluent_branch_mode(value: Union[str, FluentBranchMode]) -> FluentBranchMode:
    if isinstance(value, FluentBranchMode):
        return value
    normalized = str(value).strip().lower()
    if normalized == "group":
        return FluentBranchMode.GROUP
    if normalized == "single":
        return FluentBranchMode.SINGLE
    raise ValueError(f"Unsupported fluent_branch_mode '{value}'. Expected one of: group, single.")


@dataclass(frozen=True)
class CDPSConfig:
    """Immutable set of conflict-search shape parameters for CDPS.

    Note: this holds only the search *shape*. The runtime budget
    (``timeout_seconds``) is passed separately to ``ConflictDrivenPatchSearch.run``.
    """

    fluent_patch_cost: float = 1.0
    fluent_patch_weight: float = 1.0
    model_patch_cost: float = 1.0
    model_constraint_weight: float = 0.0
    max_search_nodes: Optional[int] = None
    search_mode: SearchMode = SearchMode.ANYTIME_DFS
    node_choosing_strategy: NodeChoosingStrategy = NodeChoosingStrategy.MODEL_PATCH_FIRST
    conflict_group_strategy: ConflictGroupStrategy = ConflictGroupStrategy.MOST_OBSERVATIONS
    fluent_branch_mode: FluentBranchMode = FluentBranchMode.GROUP
    negative_precondition_policy: NegativePreconditionPolicy = NegativePreconditionPolicy.hard
    seed: int = 42
    # NOTE: GT awareness is no longer a config switch. The search always
    # respects the GT states passed to ConflictDrivenPatchSearch.run() via
    # gt_states_by_obs (init state is always GT). The anchored variant differs
    # only in its DATA PREP: it injects the final state as GT before learning.

    def __post_init__(self) -> None:
        # frozen=True → assign through object.__setattr__ to coerce strings → enums.
        object.__setattr__(self, "search_mode", resolve_search_mode(self.search_mode))
        object.__setattr__(
            self, "node_choosing_strategy",
            resolve_node_choosing_strategy(self.node_choosing_strategy),
        )
        object.__setattr__(
            self, "conflict_group_strategy",
            resolve_conflict_group_strategy(self.conflict_group_strategy),
        )
        object.__setattr__(
            self, "fluent_branch_mode",
            resolve_fluent_branch_mode(self.fluent_branch_mode),
        )

    def to_dict(self) -> dict:
        """JSON-safe dict (enum members → their string values) for run_params/reports."""
        data = asdict(self)
        data["search_mode"] = self.search_mode.value
        data["node_choosing_strategy"] = self.node_choosing_strategy.value
        data["conflict_group_strategy"] = self.conflict_group_strategy.value
        data["fluent_branch_mode"] = self.fluent_branch_mode.value
        data["negative_precondition_policy"] = getattr(
            self.negative_precondition_policy, "name", str(self.negative_precondition_policy)
        )
        return data
