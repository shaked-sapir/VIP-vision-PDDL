from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, List, Sequence, Optional
from copy import deepcopy, copy
import heapq
import time

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
        cost = number of fluent patches, as we want to find action model which is consistent with
        training data with minimal data repairs.

    patch_operations_cost:
        Cumulative cost of patch operations applied to reach this node.

    model_constraints:
        Mapping (action, part, pbl) -> PatchOperation (FORBID / REQUIRE).

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
        seed: int = 42,
        logger: Optional[object] = None,
    ):
        self.partial_domain_template = partial_domain_template
        self.negative_preconditions_policy = negative_preconditions_policy
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
        List[Observation]
    ]:
        """
        Run the conflict-driven search on trajectories T (= observations).

        :return:
            (learned_domain, conflicts, model_constraints,
             fluent_patches, node_cost, learning_report, patched_final_observations)
        """
        root_constraints: Dict[Key, PatchOperation] = initial_model_constraints or {}
        root_fluent_patches: Set[FluentLevelPatch] = initial_fluent_patches or set()

        root_node = SearchNode(
            cost=0,
            model_constraints=root_constraints,
            fluent_patches=root_fluent_patches,
        )

        open_heap: List[SearchNode] = [root_node]
        visited: Set[Tuple] = set()

        last_domain: Optional[LearnerDomain] = None
        last_conflicts: List[Conflict] = []
        last_state: Tuple[Dict[Key, PatchOperation], Set[FluentLevelPatch]] = (
            root_constraints,
            root_fluent_patches,
        )
        last_report: Dict[str, str] = {}
        last_cost: int = root_node.cost

        nodes_expanded = 0
        max_depth = 0
        depth_tracker: Dict[Tuple, int] = {self._encode_state(root_constraints, root_fluent_patches): 0}
        start_time = time.time()

        while open_heap:
            if max_nodes is not None and nodes_expanded >= max_nodes:
                break

            node = heapq.heappop(open_heap)

            state_key = self._encode_state(node.model_constraints, node.fluent_patches)

            if state_key in visited:
                continue
            visited.add(state_key)

            nodes_expanded += 1
            current_depth = depth_tracker.get(state_key, 0)
            max_depth = max(max_depth, current_depth)

            domain, conflicts, report = self._learn_with_state(
                observations,
                node.model_constraints,
                node.fluent_patches,
            )

            last_domain, last_conflicts, last_state, last_report, last_cost = (
                domain,
                conflicts,
                (node.model_constraints,node.fluent_patches),
                report,
                node.cost
            )

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

            if not conflicts:
                # Found conflict-free model
                total_time = time.time() - start_time
                patch_diff = self._compute_patch_diff(
                    initial_constraints=root_constraints,
                    final_constraints=node.model_constraints,
                    initial_fluent_patches=root_fluent_patches,
                    final_fluent_patches=node.fluent_patches,
                )
                enriched_report = dict(report)
                enriched_report["patch_diff"] = patch_diff
                enriched_report["nodes_expanded"] = nodes_expanded
                enriched_report["max_depth"] = max_depth
                enriched_report["total_time_seconds"] = total_time

                patched_obs = self._apply_patches_to_observations(
                    observations,
                    node.fluent_patches,
                )
                return domain, [], node.model_constraints, node.fluent_patches, node.cost, enriched_report, patched_obs

            # Conflicts found - choose which conflict to branch on
            conflict = self._choose_conflict(conflicts)

            # ------------------------------------------------------------------
            # Branch 1: data-fix (fluent-level patch)
            # ------------------------------------------------------------------
            for patch in self._build_fluent_patches_for_conflict(conflict):
                child_constraints = dict(node.model_constraints)
                child_fluent_patches = set(node.fluent_patches)

                if patch in child_fluent_patches:
                    # Applying the same patch again would revert it (no net change),
                    # but we want states to represent a set of patches; skip.
                    continue

                child_fluent_patches.add(patch)
                child_fluent_patches = self._dedup_patches(child_fluent_patches)

                child_cost = len(child_fluent_patches)
                child_state = self._encode_state(child_constraints, child_fluent_patches)

                if child_state in visited:
                    continue

                depth_tracker[child_state] = current_depth + 1
                heapq.heappush(
                    open_heap,
                    SearchNode(
                        cost=child_cost,
                        model_constraints=child_constraints,
                        fluent_patches=child_fluent_patches,
                    ),
                )

            # ------------------------------------------------------------------
            # Branch 2: model-fix (add/adjust model-level patch)
            #          (Skipped for precondition-only conflicts)
            # ------------------------------------------------------------------
            if conflict.conflict_type != ConflictType.FORBID_PRECOND_VS_IS and \
                    conflict.conflict_type != ConflictType.FRAME_AXIOM:
                child2_constraints: Dict[Key, PatchOperation] = dict(node.model_constraints)
                child2_constraints = self._build_model_patch(conflict, child2_constraints)
                child2_fluent_patches = set(node.fluent_patches)

                child2_cost = len(child2_fluent_patches)
                child2_state = self._encode_state(child2_constraints, child2_fluent_patches)

                if child2_state not in visited:
                    depth_tracker[child2_state] = current_depth + 1
                    heapq.heappush(
                        open_heap,
                        SearchNode(
                            cost=child2_cost,
                            model_constraints=child2_constraints,
                            fluent_patches=child2_fluent_patches,
                        ),
                    )

        # No conflict-free model found within limits; return last evaluated
        total_time = time.time() - start_time
        last_constraints, last_fluent_patches = last_state

        patch_diff = self._compute_patch_diff(
            initial_constraints=root_constraints,
            final_constraints=last_constraints,
            initial_fluent_patches=root_fluent_patches,
            final_fluent_patches=last_fluent_patches,
        )

        enriched_report = dict(last_report)
        enriched_report["patch_diff"] = patch_diff
        enriched_report["nodes_expanded"] = nodes_expanded
        enriched_report["max_depth"] = max_depth
        enriched_report["total_time_seconds"] = total_time

        patched_obs = self._apply_patches_to_observations(
            observations,
            last_fluent_patches,
        )

        return last_domain, last_conflicts, last_constraints, last_fluent_patches, last_cost, enriched_report, patched_obs

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
            (fp.observation_index, fp.component_index, fp.state_type, fp.fluent)
            for fp in fluent_patches
        ))
        return constraints_tuple, fluent_tuple

    @staticmethod
    def _choose_conflict(conflicts: List[Conflict]) -> Conflict:
        """
        Prefer fixing frame-axioms before effects.
        If none, fall back to the first one.
        """
        for c in conflicts:
            # if c.conflict_type != ConflictType.FORBID_PRECOND_VS_IS:
            if c.conflict_type == ConflictType.FRAME_AXIOM:
                return c
        return conflicts[0]

    @staticmethod
    def _build_fluent_patch(conflict: Conflict, state_type: str) -> FluentLevelPatch:
        """Build a FluentLevelPatch for a given conflict and state_type ('prev'/'next')."""
        return FluentLevelPatch(
            observation_index=conflict.observation_index,
            component_index=conflict.component_index,
            state_type=state_type,
            fluent=conflict.grounded_fluent,
        )

    def _build_fluent_patches_for_conflict(self, conflict: Conflict) -> List[FluentLevelPatch]:
        """
        Determine which fluent patches correspond to the data-fix branches
        for a given conflict.

        - FORBID_PRECOND_VS_IS:
            * Only prev-state patches (cannot-be-precondition).
        - FRAME_AXIOM:
            * Normally both prev and next patches.
            * BUT if component_index == 0, prev_state is an initial state
              and assumed ground truth → only allow a 'next' patch.
        - Effect conflicts (FORBID/REQUIRE vs must/cannot):
            * Only next-state patches (effects).
        """
        if conflict.conflict_type == ConflictType.FORBID_PRECOND_VS_IS:
            return [self._build_fluent_patch(conflict, "prev")]

        if conflict.conflict_type == ConflictType.FRAME_AXIOM:
            # Always allow "next" as a potential noisy side
            patches: List[FluentLevelPatch] = [self._build_fluent_patch(conflict, "next")]

            # For initial transitions (component_index == 0), prev_state is ground truth:
            # we are NOT allowed to treat the initial prev_state as noisy.
            if conflict.component_index > 0:
                patches.append(self._build_fluent_patch(conflict, "prev"))

            return patches

        # Effect conflicts: data repair is in 'next' state
        return [self._build_fluent_patch(conflict, "next")]

    @staticmethod
    def _conflict_to_key(conflict: Conflict) -> Key:
        model_part = ModelPart.PRECONDITION if conflict.conflict_type == ConflictType.FORBID_PRECOND_VS_IS else ModelPart.EFFECT
        return conflict.action_name, model_part, conflict.pbl

    @staticmethod
    def _dedup_patches(patches: Set[FluentLevelPatch]) -> Set[FluentLevelPatch]:
        """
       Remove (next at (obs,comp), prev at (obs,comp+1)) pairs with same fluent.
       These correspond to a "flip + flip back" across adjacent transitions.
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
        Model-fix logic for effect conflicts:

        - If no existing model patch:
            REQUIRE_EFFECT_VS_CANNOT -> REQUIRE
            FORBID_EFFECT_VS_MUST    -> FORBID
        - If there's an existing patch for same (action,part,pbl):
            remove it (toggle back to unconstrained).
        """
        key: Key = self._conflict_to_key(conflict)
        old = model_patches
        new = copy(old)

        existing_op: Optional[PatchOperation] = old.get(key)

        if existing_op is None:
            if conflict.conflict_type == ConflictType.REQUIRE_EFFECT_VS_CANNOT:
                new[key] = PatchOperation.FORBID
            elif conflict.conflict_type == ConflictType.FORBID_EFFECT_VS_MUST:
                new[key] = PatchOperation.REQUIRE
        else:
            del new[key]

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

    def _apply_patches_to_observations(
            self,
            observations: Sequence[Observation],
            fluent_patches: Set[FluentLevelPatch],
    ) -> List[Observation]:
        """
        Apply the final fluent_patches to the original observations using
        NoisyPisamLearner's apply_fluent_patches, and return the patched copy.
        """
        domain_copy: Domain = deepcopy(self.partial_domain_template)
        learner = NoisyPisamLearner(
            partial_domain=domain_copy,
            negative_preconditions_policy=self.negative_preconditions_policy,
            seed=self.seed,
        )
        learner.set_patches(fluent_patches=fluent_patches, model_patches=set())
        return learner.apply_fluent_patches(list(observations))

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
