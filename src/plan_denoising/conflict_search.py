# conflict_search.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, List, Sequence, Optional
from copy import deepcopy
import heapq

from pddl_plus_parser.models import Domain, Observation
from sam_learning.core import LearnerDomain
from utilities import NegativePreconditionPolicy

from src.pi_sam.noisy_pisam.simpler_version.typings import (
    Conflict,
    ConflictType,
    FluentLevelPatch,
    ModelLevelPatch,
    ModelPart,
    PatchOperation,
    ParameterBoundLiteral,
)

# TODO: when working - refactor this to the same dir as SimpleNoisyPisamLearner
from src.pi_sam.noisy_pisam.simpler_version.simple_noisy_pisam_learning import SimpleNoisyPisamLearner


Key = Tuple[str, ModelPart, ParameterBoundLiteral]   # (action_name, part, pbl)


@dataclass(order=True)
class SearchNode:
    """
    Internal search node.

    cost:
        Simple heuristic = number of patches on this node:
            len(model_constraints) + len(fluent_patches)

    model_constraints:
        Mapping (action, part, pbl) -> PatchOperation (FORBID / REQUIRE)

    fluent_patches:
        Set of fluent-level patches (where to flip data).
    """
    cost: int
    model_constraints: Dict[Key, PatchOperation] = field(compare=False)
    fluent_patches: Set[FluentLevelPatch] = field(compare=False)


class ConflictDrivenPatchSearch:
    """
    Conflict-driven patch search with state-based pruning.

    State = (model_constraints, fluent_patches)

      model_constraints:
          Dict[(action_name, ModelPart, PBL) -> PatchOperation]
          Absent key => UNCONSTRAINED (no global constraint).

      fluent_patches:
          Set[FluentLevelPatch], describing data repairs.

    We maintain a visited set of these states (encoded) to avoid re-expanding
    an identical configuration of constraints + data repairs.
    """

    def __init__(
        self,
        partial_domain_template: Domain,
        negative_preconditions_policy: NegativePreconditionPolicy = NegativePreconditionPolicy.hard,
        seed: int = 42,
    ):
        """
        :param partial_domain_template:
            Partial Domain template that learners refine.
            It is deep-copied at each node so branches are independent.

        :param negative_preconditions_policy:
            Policy forwarded to SimpleNoisyPisamLearner.

        :param seed:
            Random seed forwarded to the learner.
        """
        self.partial_domain_template = partial_domain_template
        self.negative_preconditions_policy = negative_preconditions_policy
        self.seed = seed

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def run(
        self,
        observations: Sequence[Observation],
        max_nodes: Optional[int] = None,
    ) -> Tuple[LearnerDomain, List[Conflict], Dict[Key, PatchOperation], Set[FluentLevelPatch]]:
        """
        Run the conflict-driven patch search on trajectories T (= observations).

        BFS/A* hybrid: we use a min-heap with cost = #constraints + #flips.

        :param observations:
            Grounded & masked observations (trajectories).

        :param max_nodes:
            Optional limit on number of expanded nodes.

        :return:
            (learned_domain, conflicts, model_constraints, fluent_patches)

            - If a conflict-free model is found:
                  conflicts = []
                  model_constraints / fluent_patches = those of the solution state.

            - If search ends without conflict-free model:
                  returns last evaluated model and its conflicts.
        """
        # initial state: no patches
        root_constraints: Dict[Key, PatchOperation] = {}
        root_fluent_patches: Set[FluentLevelPatch] = set()

        root = SearchNode(
            cost=0,
            model_constraints=root_constraints,
            fluent_patches=root_fluent_patches,
        )

        open_heap: List[SearchNode] = [root]
        visited: Set[Tuple] = set()

        last_domain: Optional[LearnerDomain] = None
        last_conflicts: List[Conflict] = []
        last_state = (root_constraints, root_fluent_patches)

        nodes_expanded = 0

        while open_heap:
            if max_nodes is not None and nodes_expanded >= max_nodes:
                break

            node = heapq.heappop(open_heap)

            state_key = self._encode_state(node.model_constraints, node.fluent_patches)
            if state_key in visited:
                continue
            visited.add(state_key)

            nodes_expanded += 1

            domain, conflicts = self._learn_with_state(
                observations,
                node.model_constraints,
                node.fluent_patches,
            )
            last_domain, last_conflicts, last_state = domain, conflicts, (
                node.model_constraints,
                node.fluent_patches,
            )

            if not conflicts:
                # Found conflict-free model
                return domain, [], node.model_constraints, node.fluent_patches

            conflict = self._choose_conflict(conflicts)

            # Two branches: data-fix and model-fix
            data_patch = self._build_data_patch(conflict)
            key = self._conflict_key(conflict)

            # Branch 1: data-fix (keep constraints, add fluent patch)
            child1_constraints = dict(node.model_constraints)
            child1_fluent_patches = set(node.fluent_patches)
            child1_fluent_patches.add(data_patch)
            child1_cost = len(child1_constraints) + len(child1_fluent_patches)
            heapq.heappush(
                open_heap,
                SearchNode(
                    cost=child1_cost,
                    model_constraints=child1_constraints,
                    fluent_patches=child1_fluent_patches,
                ),
            )

            # Branch 2: model-fix (drop constraint for this PBL)
            child2_constraints = dict(node.model_constraints)
            child2_constraints.pop(key, None)  # UNCONSTRAINED
            child2_fluent_patches = set(node.fluent_patches)
            child2_cost = len(child2_constraints) + len(child2_fluent_patches)
            heapq.heappush(
                open_heap,
                SearchNode(
                    cost=child2_cost,
                    model_constraints=child2_constraints,
                    fluent_patches=child2_fluent_patches,
                ),
            )

        # No conflict-free model found within limits; return last evaluated
        last_constraints, last_fluent_patches = last_state
        return last_domain, last_conflicts, last_constraints, last_fluent_patches

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------

    def _encode_state(
        self,
        model_constraints: Dict[Key, PatchOperation],
        fluent_patches: Set[FluentLevelPatch],
    ) -> Tuple:
        """
        Encode state into a hashable form for the visited set.

        We sort entries to make encoding order-invariant.
        """
        constraints_tuple = tuple(sorted(
            (
                (action_name, part.value, str(pbl), op.value)
                for (action_name, part, pbl), op in model_constraints.items()
            )
        ))
        fluent_tuple = tuple(sorted(
            (
                fp.observation_index,
                fp.component_index,
                fp.state_type,
                fp.fluent,
            )
            for fp in fluent_patches
        ))
        return constraints_tuple, fluent_tuple

    def _choose_conflict(self, conflicts: List[Conflict]) -> Conflict:
        """Current policy: simply pick the first conflict."""
        return conflicts[0]

    def _conflict_key(self, conflict: Conflict) -> Key:
        """Map a Conflict to its (action, part, pbl) key."""
        if conflict.conflict_type == ConflictType.PRE_REQUIRE_VS_CANNOT:
            part = ModelPart.PRECONDITION
        else:
            part = ModelPart.EFFECT
        return conflict.action_name, part, conflict.pbl

    def _build_data_patch(self, conflict: Conflict) -> FluentLevelPatch:
        """
        Build the FluentLevelPatch for the data-fix branch.

        EFFECT conflicts:
            flip in "next" state.

        PRECONDITION conflicts:
            flip in "prev" state.
        """
        if conflict.conflict_type == ConflictType.PRE_REQUIRE_VS_CANNOT:
            state_type = "prev"
        else:  # effect conflicts
            state_type = "next"

        return FluentLevelPatch(
            observation_index=conflict.observation_index,
            component_index=conflict.component_index,
            state_type=state_type,
            fluent=conflict.grounded_fluent,
        )

    def _learn_with_state(
        self,
        observations: Sequence[Observation],
        model_constraints: Dict[Key, PatchOperation],
        fluent_patches: Set[FluentLevelPatch],
    ) -> Tuple[LearnerDomain, List[Conflict]]:
        """
        Construct a fresh SimpleNoisyPisamLearner and run
        learn_action_model_with_conflicts(T, P) on it, given the current state.
        """
        # Convert model_constraints -> set of ModelLevelPatch for the learner
        model_patches: Set[ModelLevelPatch] = set()
        for (action_name, part, pbl), op in model_constraints.items():
            model_patches.add(
                ModelLevelPatch(
                    action_name=action_name,
                    model_part=part,
                    pbl=pbl,
                    operation=op,
                )
            )

        # Fresh deep copy of the partial domain template for this branch
        domain_copy: Domain = deepcopy(self.partial_domain_template)

        learner = SimpleNoisyPisamLearner(
            partial_domain=domain_copy,
            negative_preconditions_policy=self.negative_preconditions_policy,
            seed=self.seed,
        )

        learned_domain, conflicts = learner.learn_action_model_with_conflicts(
            observations=list(observations),
            fluent_patches=fluent_patches,
            model_patches=model_patches,
        )

        return learned_domain, conflicts