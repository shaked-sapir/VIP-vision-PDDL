from __future__ import annotations

import heapq
import logging
import time
from copy import deepcopy, copy
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, List, Sequence, Optional

from pddl_plus_parser.models import Domain, Observation
from sam_learning.core import LearnerDomain
from utilities import NegativePreconditionPolicy

from src.pi_sam.noisy_pisam.simpler_version.noisy_sam_learning import NoisySAMLearner
from src.pi_sam.noisy_pisam.simpler_version.typings import (
    Conflict,
    ConflictType,
    FluentLevelPatch,
    ModelLevelPatch,
    ModelPart,
    PatchOperation,
    ParameterBoundLiteral, ConflictPriority,
)

Key = Tuple[str, ModelPart, ParameterBoundLiteral]   # (action_name, part, pbl)


class DefaultSearchLogger(logging.Logger):
    """A logger that also implements log_node(), used when user does not supply a logger."""

    def log_node(self, **kwargs):
        # Very basic structured debug log
        msg = "NODE | " + ", ".join(f"{k}={v}" for k, v in kwargs.items())
        self.debug(msg)


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
    conflicts_count: int           # secondary: #conflicts in this node
    depth: int
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
        if logger is None:
            # Create default structured logger
            logger = DefaultSearchLogger(f"{__name__}.ConflictDrivenPatchSearch")
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
                handler.setFormatter(formatter)
                logger.addHandler(handler)

            logger.setLevel(logging.DEBUG)  # ensure node logs are printed

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
        timeout_seconds: int = 300,
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
        Run the conflict-driven patch search on trajectories T (= observations).

        Strategy:
          - Uniform-cost search on (#fluent patches).
          - Node ordering = (fluent_patch_count, depth).
          - Optional limits:
              * max_nodes      – max expanded nodes
              * timeout_seconds – max wall-clock time (default: 5 minutes)

        :param observations:
            Grounded & masked observations (trajectories).

        :param max_nodes:
            Optional limit on number of expanded nodes.

        :param initial_model_constraints:
            Optional initial model constraints to start from.

        :param initial_fluent_patches:
            Optional initial fluent-level patches to start from.

        :param timeout_seconds:
            Maximum allowed wall-clock time in seconds for the whole run.
            Default is 300 seconds (5 minutes).

        :return:
            (learned_domain,
             conflicts,
             model_constraints,
             fluent_patches,
             best_fluent_patch_count,
             learning_report_with_patch_diff)
        """
        root_constraints: Dict[Key, PatchOperation] = initial_model_constraints or {}
        root_fluent_patches: Set[FluentLevelPatch] = initial_fluent_patches or set()

        root_node = SearchNode(
            cost=0,
            conflicts_count=0,
            depth=0,
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
        terminated_by = ""
        while open_heap:
            if max_nodes is not None and nodes_expanded >= max_nodes:
                terminated_by = "max_nodes_exceeded"
                if self.logger is not None:
                    self.logger.info(f"Stopping search: reached max_nodes={max_nodes}")
                break

            if timeout_seconds is not None and (time.time() - start_time) >= timeout_seconds:
                terminated_by = "timeout_exceeded"
                if self.logger is not None:
                    self.logger.info(
                        f"Stopping search: timeout of {timeout_seconds:.1f}s exceeded"
                    )
                break

            node: SearchNode = heapq.heappop(open_heap)

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

            # if hasattr(self.logger, "log_node"):
            #     self.logger.log_node(
            #         node_id=nodes_expanded,
            #         depth=current_depth,
            #         cost=node.cost,
            #         model_constraints=node.model_constraints,
            #         fluent_patches=node.fluent_patches,
            #         conflicts=conflicts,
            #         is_solution=(len(conflicts) == 0),
            #     )

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
                enriched_report["terminated_by"] = "solution_found"
                enriched_report["total_time_seconds"] = total_time

                patched_obs = self._apply_patches_to_observations(
                    observations,
                    node.fluent_patches,
                )
                return domain, [], node.model_constraints, node.fluent_patches, node.cost, enriched_report, patched_obs

            # Conflicts found - choose which conflict to branch on
            # --- GROUP conflicts and pick best group according to priority ---
            conflict_groups = self._group_conflicts(conflicts)
            group = self._choose_conflict_group(conflict_groups)

            # ------------------------------------------------------------------
            # Branch 1: DATA-FIX (fluent patches for all conflicts in the group)
            # ------------------------------------------------------------------
            child1_constraints = dict(node.model_constraints)
            child1_fluent_patches = set(node.fluent_patches)

            new_patches = [self._build_fluent_patch(c) for c in group]
            changed = False
            for fp in new_patches:
                if fp not in child1_fluent_patches:
                    child1_fluent_patches.add(fp)
                    changed = True

            if changed:
                child1_fluent_patches = self._dedup_patches(child1_fluent_patches)
                child1_fluent_count = len(child1_fluent_patches)
                depth_tracker[self._encode_state(child1_constraints, child1_fluent_patches)] = current_depth + 1

                heapq.heappush(
                    open_heap,
                    SearchNode(
                        cost=child1_fluent_count,
                        conflicts_count=len(group),
                        depth=node.depth + 1,
                        model_constraints=child1_constraints,
                        fluent_patches=child1_fluent_patches,
                    ),
                )

            # ------------------------------------------------------------------
            # Branch 2: model-fix (add/adjust model-level effect patch)
            # ------------------------------------------------------------------
            rep_conflict = group[0]
            if rep_conflict.conflict_type not in {ConflictType.FRAME_AXIOM}:
                child2_constraints: Dict[Key, PatchOperation] = dict(node.model_constraints)
                child2_constraints = self._build_model_patch(rep_conflict, child2_constraints)
                child2_fluent_patches = set(node.fluent_patches)
                child2_fluent_count = len(child2_fluent_patches)

                # If model_constraints actually changed, push new node
                if child2_constraints != node.model_constraints:
                    depth_tracker[self._encode_state(child2_constraints, child2_fluent_patches)] = current_depth + 1

                    heapq.heappush(
                        open_heap,
                        SearchNode(
                            cost=child2_fluent_count,
                            conflicts_count=len(group),
                            depth=node.depth + 1,
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
        enriched_report["v"] = terminated_by
        enriched_report["total_time_seconds"] = total_time

        patched_obs = self._apply_patches_to_observations(
            observations,
            last_fluent_patches,
        )

        return last_domain, last_conflicts, last_constraints, last_fluent_patches, last_cost, enriched_report, patched_obs

    # ----------------------------------------------------------------------
    # Internal helpers: encoding + priorities + grouping
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
    def _conflict_priority(conflict: Conflict) -> ConflictPriority:
        """
        Map a Conflict to its high-level priority class.

        Current policy:
          EFFECT (FORBID/REQUIRE)   >   FRAME_AXIOM   >   OTHER
        """
        if conflict.conflict_type in {
            ConflictType.FORBID_EFFECT_VS_MUST,
            ConflictType.REQUIRE_EFFECT_VS_CANNOT,
        }:
            return ConflictPriority.EFFECT
        if conflict.conflict_type == ConflictType.FRAME_AXIOM:
            return ConflictPriority.FRAME_AXIOM
        return ConflictPriority.OTHER

    def _group_key(
            self,
            conflict: Conflict,
    ) -> Tuple[ConflictPriority, str, ModelPart, ParameterBoundLiteral, ConflictType]:
        """
        Group key: conflicts share a group if they correspond to the same
        (priority, action, model-part, PBL, conflict-type).
        """
        part = ModelPart.EFFECT
        if conflict.conflict_type == ConflictType.FRAME_AXIOM:
            part = ModelPart.OTHER

        return (
            self._conflict_priority(conflict),
            conflict.action_name,
            part,
            conflict.pbl,
            conflict.conflict_type,
        )

    def _group_conflicts(self, conflicts: List[Conflict]) -> List[List[Conflict]]:
        """
        Group conflicts by (priority, action, part, pbl, type).

        Each group generates:
          - One data-branch: multiple FluentLevelPatch (one per conflict).
          - At most one model-branch (for effect conflicts only).
        """
        groups: Dict[Tuple, List[Conflict]] = {}
        for c in conflicts:
            k = self._group_key(c)
            groups.setdefault(k, []).append(c)
        return list(groups.values())

    def _choose_conflict_group(self, groups: List[List[Conflict]]) -> List[Conflict]:
        """
        Choose the best group:

          1) Prefer groups whose conflicts have EFFECT priority over FRAME_AXIOM.
          2) Within the same priority, prefer the earliest (obs_idx, comp_idx).
          3) If still tied, just pick the first.

        Returns the chosen group (list[Conflict]).
        """

        def group_score(g: List[Conflict]) -> Tuple[int, int, int]:
            rep = min(
                g,
                key=lambda c: (
                    int(self._conflict_priority(c)),
                    c.observation_index,
                    c.component_index,
                ),
            )
            prio = int(self._conflict_priority(rep))
            return prio, rep.observation_index, rep.component_index

        return min(groups, key=group_score)

    # ----------------------------------------------------------------------
    # Fluent patches & model patches
    # ----------------------------------------------------------------------

    @staticmethod
    def _build_fluent_patch(conflict: Conflict) -> FluentLevelPatch:
        """
        Build the FluentLevelPatch for a single conflict.

        EFFECT conflicts:
            flip in "next" state.
        FRAME_AXIOM conflicts:
            also handled as "next" (direction encoded in grounded_fluent itself).
        """
        if conflict.conflict_type == ConflictType.FORBID_PRECOND_VS_IS:
            state_type = "prev"
        else:
            state_type = "next"

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

        learner = NoisySAMLearner(
            partial_domain=domain_copy,
            negative_preconditions_policy=self.negative_preconditions_policy
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
        learner = NoisySAMLearner(
            partial_domain=domain_copy,
            negative_preconditions_policy=self.negative_preconditions_policy
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
