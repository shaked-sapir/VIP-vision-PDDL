from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, List, Sequence, Optional
from copy import deepcopy, copy
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
from src.pi_sam.noisy_pisam.simpler_version.simple_noisy_pisam_learning import NoisyPisamLearner


Key = Tuple[str, ModelPart, ParameterBoundLiteral]   # (action_name, part, pbl)


@dataclass(order=True)
class SearchNode:
    """
    Internal search node.

    cost:
        A* cost = patch_operations_cost + heuristic (here: #conflicts).

    patch_operations_cost:
        Cumulative cost of patch operations applied to reach this node.

    model_constraints:
        Mapping (action, part, pbl) -> PatchOperation (FORBID / REQUIRE).

    fluent_patches:
        Set of fluent-level patches (where to flip data).
    """
    sort_key: Tuple[int, int] = field(init=False, repr=False)
    cost: int
    patch_operations_cost: int
    model_constraints: Dict[Key, PatchOperation] = field(compare=False)
    fluent_patches: Set[FluentLevelPatch] = field(compare=False)

    def __post_init__(self):
        # tie-breaker: prefer lower patch_operations_cost when costs equal
        self.sort_key = (self.cost, self.patch_operations_cost)


class ConflictDrivenPatchSearch:
    """
    Conflict-driven patch search with state-based pruning.

    State = (model_constraints, fluent_patches)

      model_constraints:
          Dict[(action_name, ModelPart, PBL) -> PatchOperation]
          Absent key => UNCONSTRAINED.

      fluent_patches:
          Set[FluentLevelPatch], describing data repairs.

    We maintain a visited set over encoded states to avoid re-expanding
    identical configurations of constraints + data repairs.
    """

    def __init__(
        self,
        partial_domain_template: Domain,
        negative_preconditions_policy: NegativePreconditionPolicy = NegativePreconditionPolicy.hard,
        fluent_patch_cost: int = 1,
        model_patch_cost: int = 1,
        seed: int = 42,
        logger: Optional[object] = None,
    ):
        self.partial_domain_template = partial_domain_template
        self.negative_preconditions_policy = negative_preconditions_policy
        self.fluent_patch_cost = fluent_patch_cost
        self.model_patch_cost = model_patch_cost
        self.seed = seed
        self.logger = logger

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def run(
        self,
        observations: Sequence[Observation],
        max_nodes: Optional[int] = None,
        initial_model_constraints: Optional[Dict[Key, PatchOperation]] = None,
        initial_fluent_patches: Optional[Set[FluentLevelPatch]] = None,
    ) -> Tuple[
        LearnerDomain,
        List[Conflict],
        Dict[Key, PatchOperation],
        Set[FluentLevelPatch],
        int,
        Dict[str, str],
    ]:
        """
        Run the conflict-driven patch search on trajectories T (= observations).

        A* search with:
          - g = patch_operations_cost
          - h = current number of conflicts

        :return:
            (learned_domain, conflicts, model_constraints,
             fluent_patches, node_cost, learning_report)
        """
        root_constraints: Dict[Key, PatchOperation] = initial_model_constraints or {}
        root_fluent_patches: Set[FluentLevelPatch] = initial_fluent_patches or set()

        root_state = self._encode_state(root_constraints, root_fluent_patches)
        root_node = SearchNode(
            cost=0,
            patch_operations_cost=0,
            model_constraints=root_constraints,
            fluent_patches=root_fluent_patches,
        )

        open_heap: List[Tuple[int, int, SearchNode]] = []
        heapq.heappush(open_heap, (root_node.cost, root_node.patch_operations_cost, root_node))

        visited: Set[Tuple] = set()
        visited.add(root_state)

        last_domain: Optional[LearnerDomain] = None
        last_conflicts: List[Conflict] = []
        last_state = (root_constraints, root_fluent_patches)
        last_report: Dict[str, str] = {}
        nodes_expanded = 0

        depth_tracker: Dict[Tuple, int] = {root_state: 0}

        while open_heap:
            if max_nodes is not None and nodes_expanded >= max_nodes:
                break

            _, _, node = heapq.heappop(open_heap)

            state_key = self._encode_state(node.model_constraints, node.fluent_patches)
            # state_key must be in visited because we only push unseen; but keep this for safety
            if state_key not in visited:
                visited.add(state_key)

            nodes_expanded += 1
            current_depth = depth_tracker.get(state_key, 0)

            domain, conflicts, report = self._learn_with_state(
                observations,
                node.model_constraints,
                node.fluent_patches,
            )

            last_domain, last_conflicts, last_state, last_report = domain, conflicts, (
                node.model_constraints,
                node.fluent_patches,
            ), report

            if self.logger is not None:
                self.logger.log_node(
                    node_id=nodes_expanded,
                    depth=current_depth,
                    cost=node.cost,
                    model_constraints=node.model_constraints,
                    fluent_patches=node.fluent_patches,
                    conflicts=conflicts,
                    is_solution=(len(conflicts) == 0),
                )

            # Solution: no conflicts
            if not conflicts:
                patch_diff = self._compute_patch_diff(
                    initial_constraints=root_constraints,
                    final_constraints=node.model_constraints,
                    initial_fluent_patches=root_fluent_patches,
                    final_fluent_patches=node.fluent_patches,
                )
                enriched_report = dict(report)
                enriched_report["patch_diff"] = patch_diff

                return domain, [], node.model_constraints, node.fluent_patches, node.cost, enriched_report

            # Choose which conflict to branch on
            conflict = self._choose_conflict(conflicts)

            children_generated = False

            # ------------------------------------------------------------------
            # Branch 1: data-fix (fluent-level patch)
            # ------------------------------------------------------------------
            conflict_fluent_patch = self._build_fluent_patch(conflict)
            child1_model_constraints = dict(node.model_constraints)
            child1_fluent_patches = set(node.fluent_patches)

            if conflict_fluent_patch not in child1_fluent_patches:
                child1_fluent_patches.add(conflict_fluent_patch)
                child1_fluent_patches = self._dedup_patches(child1_fluent_patches)

                child1_patch_operations_cost = node.patch_operations_cost + self.fluent_patch_cost
                child1_cost = child1_patch_operations_cost + len(conflicts)

                child1_state = self._encode_state(child1_model_constraints, child1_fluent_patches)
                if child1_state not in visited:
                    visited.add(child1_state)
                    depth_tracker[child1_state] = current_depth + 1
                    child1_node = SearchNode(
                        cost=child1_cost,
                        patch_operations_cost=child1_patch_operations_cost,
                        model_constraints=child1_model_constraints,
                        fluent_patches=child1_fluent_patches,
                    )
                    heapq.heappush(open_heap, (child1_node.cost, child1_node.patch_operations_cost, child1_node))
                    children_generated = True

            # ------------------------------------------------------------------
            # Branch 2: model-fix (add/adjust model-level patch)
            #          (Skipped for precondition-only conflicts)
            # ------------------------------------------------------------------
            if conflict.conflict_type != ConflictType.FORBID_PRECOND_VS_IS:
                new_constraints = self._build_model_patch(conflict, node.model_constraints)
                if new_constraints is not None:
                    child2_model_constraints = new_constraints
                    child2_fluent_patches = set(node.fluent_patches)

                    child2_patch_operations_cost = node.patch_operations_cost + self.model_patch_cost
                    child2_cost = child2_patch_operations_cost + len(conflicts)

                    child2_state = self._encode_state(child2_model_constraints, child2_fluent_patches)
                    if child2_state not in visited:
                        visited.add(child2_state)
                        depth_tracker[child2_state] = current_depth + 1
                        child2_node = SearchNode(
                            cost=child2_cost,
                            patch_operations_cost=child2_patch_operations_cost,
                            model_constraints=child2_model_constraints,
                            fluent_patches=child2_fluent_patches,
                        )
                        heapq.heappush(open_heap, (child2_node.cost, child2_node.patch_operations_cost, child2_node))
                        children_generated = True

            # If we could not generate any new children from this node,
            # and there are no more nodes in the heap, the search is stuck.
            if not children_generated and not open_heap:
                break

        # No conflict-free model found within limits; return last evaluated
        last_constraints, last_fluent_patches = last_state

        patch_diff = self._compute_patch_diff(
            initial_constraints=root_constraints,
            final_constraints=last_constraints,
            initial_fluent_patches=root_fluent_patches,
            final_fluent_patches=last_fluent_patches,
        )

        enriched_report = dict(last_report)
        enriched_report["patch_diff"] = patch_diff

        # cost = patch_operations_cost + heuristic; here we reuse the last node's cost
        last_cost = 0 if last_domain is None else root_node.cost
        return last_domain, last_conflicts, last_constraints, last_fluent_patches, last_cost, enriched_report

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------

    @staticmethod
    def _encode_state(
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

    @staticmethod
    def _choose_conflict(conflicts: List[Conflict]) -> Conflict:
        """
        Prefer effect conflicts over precondition conflicts.
        If none, fall back to the first one.
        """
        for c in conflicts:
            if c.conflict_type != ConflictType.FORBID_PRECOND_VS_IS:
                return c
        return conflicts[0]

    @staticmethod
    def _build_fluent_patch(conflict: Conflict) -> FluentLevelPatch:
        """
        Build the FluentLevelPatch for the data-fix branch.

        EFFECT conflicts:
            flip in "next" state.

        PRECONDITION conflicts:
            flip in "prev" state.
        """
        state_type = "prev" if conflict.conflict_type == ConflictType.FORBID_PRECOND_VS_IS else "next"
        return FluentLevelPatch(
            observation_index=conflict.observation_index,
            component_index=conflict.component_index,
            state_type=state_type,
            fluent=conflict.grounded_fluent,
        )

    @staticmethod
    def _conflict_to_key(conflict: Conflict) -> Key:
        model_part = ModelPart.PRECONDITION if conflict.conflict_type == ConflictType.FORBID_PRECOND_VS_IS else ModelPart.EFFECT
        return conflict.action_name, model_part, conflict.pbl

    @staticmethod
    def _dedup_patches(patches: Set[FluentLevelPatch]) -> Set[FluentLevelPatch]:
        """
        Cancel pairs:
            next @ (obs=X, comp=Y, fluent=F)
        with
            prev @ (obs=X, comp=Y+1, fluent=F)

        Both patches in such a pair are removed.
        """
        next_p = {
            (p.observation_index, p.component_index, p.fluent): p
            for p in patches if p.state_type == "next"
        }
        prev_p = {
            (p.observation_index, p.component_index, p.fluent): p
            for p in patches if p.state_type == "prev"
        }

        remove: Set[FluentLevelPatch] = set()

        for (obs, comp, fluent), p_next in next_p.items():
            key_prev = (obs, comp + 1, fluent)
            p_prev = prev_p.get(key_prev)
            if p_prev:
                remove.add(p_next)
                remove.add(p_prev)

        return {p for p in patches if p not in remove}

    def _build_model_patch(
        self,
        conflict: Conflict,
        model_patches: Dict[Key, PatchOperation],
    ) -> Optional[Dict[Key, PatchOperation]]:
        """
        Model-fix branch:

        Monotonic semantics:
          - If there isn't a model patch with this (action, part, pbl):
                * REQUIRE_EFFECT_VS_CANNOT -> add REQUIRE
                * FORBID_EFFECT_VS_MUST    -> add FORBID
          - If there *is* a model patch for this Key:
                * we DO NOT toggle/remove; that would recreate an old state.
                * return None to signify "no new state".
        """
        key: Key = self._conflict_to_key(conflict)

        old = model_patches
        existing_op: Optional[PatchOperation] = old.get(key)

        if existing_op is not None:
            # Already constrained in some way; toggling/removing would
            # just recreate a previous state, so we skip this branch.
            return None

        new = copy(old)

        if conflict.conflict_type == ConflictType.REQUIRE_EFFECT_VS_CANNOT:
            new[key] = PatchOperation.REQUIRE
        elif conflict.conflict_type == ConflictType.FORBID_EFFECT_VS_MUST:
            new[key] = PatchOperation.FORBID
        else:
            # We don't introduce PRECONDITION patches here, only EFFECT ones.
            return None

        return new

    def _learn_with_state(
        self,
        observations: Sequence[Observation],
        model_constraints: Dict[Key, PatchOperation],
        fluent_patches: Set[FluentLevelPatch],
    ) -> Tuple[LearnerDomain, List[Conflict], Dict[str, str]]:
        """
        Construct a fresh NoisyPisamLearner and run
        learn_action_model_with_conflicts(T, P) on it, given the current state.
        """
        model_patches: Set[ModelLevelPatch] = {
            ModelLevelPatch(
                action_name=action_name,
                model_part=part,
                pbl=pbl,
                operation=op,
            )
            for (action_name, part, pbl), op in model_constraints.items()
        }

        domain_copy: Domain = deepcopy(self.partial_domain_template)

        learner = NoisyPisamLearner(
            partial_domain=domain_copy,
            negative_preconditions_policy=self.negative_preconditions_policy,
            seed=self.seed,
        )

        learned_domain, conflicts, report = learner.learn_action_model_with_conflicts(
            observations=list(observations),
            fluent_patches=fluent_patches,
            model_patches=model_patches,
        )

        return learned_domain, conflicts, report

    @staticmethod
    def _compute_patch_diff(
        initial_constraints: Dict[Key, PatchOperation],
        final_constraints: Dict[Key, PatchOperation],
        initial_fluent_patches: Set[FluentLevelPatch],
        final_fluent_patches: Set[FluentLevelPatch],
    ) -> Dict[str, object]:
        """
        Compute what changed between initial_* and final_*.

        This is just for reporting / explanation, not for learning.
        """
        initial_keys = set(initial_constraints.keys())
        final_keys = set(final_constraints.keys())

        added_keys = final_keys - initial_keys
        removed_keys = initial_keys - final_keys
        common_keys = initial_keys & final_keys

        changed_ops = {
            key: (initial_constraints[key], final_constraints[key])
            for key in common_keys
            if initial_constraints[key] != final_constraints[key]
        }

        model_added = {key: final_constraints[key] for key in added_keys}
        model_removed = {key: initial_constraints[key] for key in removed_keys}

        fluent_added = final_fluent_patches - initial_fluent_patches
        fluent_removed = initial_fluent_patches - final_fluent_patches

        return {
            "model_patches_added": model_added,
            "model_patches_removed": model_removed,
            "model_patches_changed": changed_ops,
            "fluent_patches_added": fluent_added,
            "fluent_patches_removed": fluent_removed,
        }
