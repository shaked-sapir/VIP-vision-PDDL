"""Search frontiers and node definition for conflict-driven patch search."""

from __future__ import annotations

import heapq
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Tuple

from src.pi_sam.noisy_pisam.typings import (
    FluentLevelPatch,
    ModelPart,
    ParameterBoundLiteral,
    PatchOperation,
)

Key = Tuple[str, ModelPart, ParameterBoundLiteral]  # (action_name, part, pbl)


class SearchMode(Enum):
    """Available search strategies for conflict-driven patch search."""
    ANYTIME_DFS = "anytime_dfs"
    UCS = "ucs"


class NodeChoosingStrategy(Enum):
    """Order policy for inserting model/fluent branch children."""

    MODEL_PATCH_FIRST = "model_patch_first"
    FLUENT_PATCH_FIRST = "fluent_patch_first"
    FLUENT_PATCH_FIRST_THEN_MODEL = "fluent_patch_first_then_model"
    RANDOMIZED = "randomized"


class ConflictGroupStrategy(Enum):
    """Policy for choosing which conflict group to resolve at each node expansion.

    FIRST:
        Original behavior — pick the group whose representative conflict has
        the lowest (priority, observation_index, component_index). Deterministic
        but ignores group size.

    LARGEST:
        Within the same priority tier, prefer the group with the most conflicts.
        Resolving a large group via model patch eliminates many conflicts at once,
        reducing tree depth.

    LARGEST_MODEL_PATCHABLE:
        Like LARGEST, but only counts non-FRAME_AXIOM conflicts in the size
        (since FRAME_AXIOM has no model-patch branch). Falls back to LARGEST
        ordering for groups that are entirely FRAME_AXIOM.

    MOST_OBSERVATIONS:
        Prefer groups whose conflicts span the most distinct observations.
        Resolving a constraint that conflicts with many different trajectories
        is more likely to be a genuine model error (vs. a single noisy obs).
    """

    FIRST = "first"
    LARGEST = "largest"
    LARGEST_MODEL_PATCHABLE = "largest_model_patchable"
    MOST_OBSERVATIONS = "most_observations"


@dataclass(order=True)
class SearchNode:
    """Search node ordered by (cost, depth).

    cost = weighted sum of fluent patches and model constraints (lower is better).
    """
    cost: float
    depth: int
    model_constraints: Dict[Key, PatchOperation] = field(compare=False)
    fluent_patches: Set[FluentLevelPatch] = field(compare=False)


class Frontier(ABC):
    """Abstract frontier — controls which node is expanded next."""

    @abstractmethod
    def push(self, node: SearchNode) -> None: ...

    @abstractmethod
    def pop(self) -> SearchNode: ...

    @abstractmethod
    def __bool__(self) -> bool: ...


class HeapFrontier(Frontier):
    """Min-heap frontier (UCS): always expands the lowest-cost node."""

    def __init__(self) -> None:
        self._heap: List[Tuple[float, int, int, SearchNode]] = []
        self._counter = itertools.count()

    def push(self, node: SearchNode) -> None:
        heapq.heappush(
            self._heap,
            (node.cost, node.depth, next(self._counter), node),
        )

    def pop(self) -> SearchNode:
        return heapq.heappop(self._heap)[3]

    def __bool__(self) -> bool:
        return len(self._heap) > 0


class StackFrontier(Frontier):
    """LIFO stack frontier (DFS): always expands the most recently added node."""

    def __init__(self) -> None:
        self._stack: List[SearchNode] = []

    def push(self, node: SearchNode) -> None:
        self._stack.append(node)

    def pop(self) -> SearchNode:
        return self._stack.pop()

    def __bool__(self) -> bool:
        return len(self._stack) > 0


_FRONTIER_REGISTRY: Dict[SearchMode, type] = {
    SearchMode.UCS: HeapFrontier,
    SearchMode.ANYTIME_DFS: StackFrontier,
}


def make_frontier(search_mode: SearchMode) -> Frontier:
    """Factory: create the right frontier for a given search mode."""
    cls = _FRONTIER_REGISTRY.get(search_mode)
    if cls is None:
        raise ValueError(
            f"Unknown search_mode: {search_mode!r}. "
            f"Valid options: {[m.value for m in SearchMode]}"
        )
    return cls()
