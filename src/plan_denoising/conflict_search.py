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

# TODO: when working - refactor this to the same dir as SimpleNoisyPisamLearner
from src.pi_sam.noisy_pisam.simpler_version.simple_noisy_pisam_learning import NoisyPisamLearner


Key = Tuple[str, ModelPart, ParameterBoundLiteral]   # (action_name, part, pbl)


@dataclass(order=True)
class SearchNode:
    """
    Internal search node.
    cost:
        heuristic = A* based (patch operation cost + #conflicts from learner)

    patch_operations_cost:
        Cumulative cost of patch operations applied to reach this node.

    model_constraints:
        Mapping (action, part, pbl) -> PatchOperation (FORBID / REQUIRE)

    fluent_patches:
        Set of fluent-level patches (where to flip data).
    """
    sort_key: Tuple[int, int] = field(init=False, repr=False)
    cost: int
    patch_operations_cost: int
    model_constraints: Dict[Key, PatchOperation] = field(compare=False)
    fluent_patches: Set[FluentLevelPatch] = field(compare=False)

    def __post_init__(self):
        self.sort_key = (self.cost, self.patch_operations_cost)


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
        fluent_patch_cost: int = 1,
        model_patch_cost: int = 1,
        seed: int = 42,
        logger: Optional[object] = None,
    ):
        """
        :param partial_domain_template:
            Partial Domain template that learners refine.
            It is deep-copied at each node so branches are independent.

        :param negative_preconditions_policy:
            Policy forwarded to SimpleNoisyPisamLearner.

        :param fluent_patch_cost:
            Cost of each fluent-level patch (data-fix).

        :param model_patch_cost:
            Cost of each model-level patch (model-fix).

        :param seed:
            Random seed forwarded to the learner.

        :param logger:
            Optional logger object with a log_node() method for tracking search tree traversal.
        """
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
    ) -> Tuple[LearnerDomain, List[Conflict], Dict[Key, PatchOperation], Set[FluentLevelPatch], int, Dict[str, str]]:
        """
        Run the conflict-driven patch search on trajectories T (= observations).

        BFS/A* hybrid: we use a min-heap with cost = #constraints + #flips.

        :param observations:
            Grounded & masked observations (trajectories).

        :param max_nodes:
            Optional limit on number of expanded nodes.

        :param initial_model_constraints:
            Optional initial model constraints to start from.

        :param initial_fluent_patches:
            Optional initial fluent-level patches to start from.

        :return:
            (learned_domain, conflicts, model_constraints, fluent_patches, node_cost, learning_report)

            - If a conflict-free model is found:
                  conflicts = []
                  model_constraints / fluent_patches = those of the solution state.

            - If search ends without conflict-free model:
                  returns last evaluated model and its conflicts.
        """
        # initial state: no patches
        root_constraints: Dict[Key, PatchOperation] = initial_model_constraints or {}
        root_fluent_patches: Set[FluentLevelPatch] = initial_fluent_patches or set()

        root = SearchNode(
            cost=0,
            patch_operations_cost=0,
            model_constraints=root_constraints,
            fluent_patches=root_fluent_patches,
        )

        open_heap: List[SearchNode] = [root]
        visited: Set[Tuple] = set()

        last_domain: Optional[LearnerDomain] = None
        last_conflicts: List[Conflict] = []
        last_state: Tuple[Dict[Key, PatchOperation], Set[FluentLevelPatch]] = (root_constraints, root_fluent_patches)
        last_report: Dict[str, str] = {}

        nodes_expanded = 0
        depth_tracker: Dict[Tuple, int] = {}  # Track depth of each state
        depth_tracker[self._encode_state({}, set())] = 0

        while open_heap:
            if max_nodes is not None and nodes_expanded >= max_nodes:
                break

            node: SearchNode = heapq.heappop(open_heap)

            state_key = self._encode_state(node.model_constraints, node.fluent_patches)
            if state_key in visited:
                continue
            visited.add(state_key)

            nodes_expanded += 1
            current_depth = depth_tracker.get(state_key, nodes_expanded)

            domain, conflicts, report = self._learn_with_state(
                observations,
                node.model_constraints,
                node.fluent_patches,
            )
            last_domain, last_conflicts, last_state, last_report = domain, conflicts, (
                node.model_constraints,
                node.fluent_patches,
            ), report

            # Log node expansion if logger is provided
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
                patch_diff = self._compute_patch_diff(
                    initial_constraints=root_constraints,
                    final_constraints=node.model_constraints,
                    initial_fluent_patches=root_fluent_patches,
                    final_fluent_patches=node.fluent_patches,
                )

                # augment learner report with patch-diff info
                enriched_report = dict(report)
                enriched_report["patch_diff"] = patch_diff

                return domain, [], node.model_constraints, node.fluent_patches, node.cost, enriched_report

            conflict = self._choose_conflict(conflicts)

            # Two child branches: branches: fluent-fix and model-fix
            conflict_fluent_patch = self._build_fluent_patch(conflict)

            # Branch 1: data-fix (keep constraints, add fluent patch)
            child1_model_constraints = dict(node.model_constraints)
            child1_fluent_patches = set(node.fluent_patches)
            if conflict_fluent_patch not in child1_fluent_patches:
                child1_fluent_patches.add(conflict_fluent_patch)
            else:
                child1_fluent_patches.remove(conflict_fluent_patch)  # flip again to avoid duplicates

            child1_patch_operations_cost = node.patch_operations_cost + self.fluent_patch_cost
            child1_cost = child1_patch_operations_cost + len(conflicts)  # A* style cost with conflicts as heuristic

            child1_state = self._encode_state(child1_model_constraints, child1_fluent_patches)
            depth_tracker[child1_state] = current_depth + 1
            heapq.heappush(
                open_heap,
                SearchNode(
                    cost=child1_cost,
                    patch_operations_cost=child1_patch_operations_cost,
                    model_constraints=child1_model_constraints,
                    fluent_patches=child1_fluent_patches,
                ),
            )

            # Branch 2: model-fix (drop constraint for this PBL) - only for effect conflicts
            if conflict.conflict_type != ConflictType.FORBID_PRECOND_VS_IS:
                child2_model_constraints: Dict[Key, PatchOperation] = dict(node.model_constraints)
                child2_model_constraints = self._build_model_patch(conflict, child2_model_constraints)
                child2_fluent_patches = set(node.fluent_patches)

                child2_patch_operations_cost = node.patch_operations_cost + self.model_patch_cost
                child2_cost = child2_patch_operations_cost + len(conflicts)

                child2_state = self._encode_state(child2_model_constraints, child2_fluent_patches)
                depth_tracker[child2_state] = current_depth + 1
                heapq.heappush(
                    open_heap,
                    SearchNode(
                        cost=child2_cost,
                        patch_operations_cost=child2_patch_operations_cost,
                        model_constraints=child2_model_constraints,
                        fluent_patches=child2_fluent_patches,
                    ),
                )

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

        return last_domain, last_conflicts, last_constraints, last_fluent_patches, node.cost, enriched_report

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
        """Current policy: simply pick the first conflict."""
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
        if conflict.conflict_type == ConflictType.FORBID_PRECOND_VS_IS:
            state_type = "prev"
        else:  # effect conflicts
            state_type = "next"

        return FluentLevelPatch(
            observation_index=conflict.observation_index,
            component_index=conflict.component_index,
            state_type=state_type,
            fluent=conflict.grounded_fluent,
        )

    @staticmethod
    def _conflict_to_key(conflict: Conflict) -> Key:
        model_part = ModelPart.EFFECT if conflict.conflict_type != ConflictType.FORBID_PRECOND_VS_IS else ModelPart.PRECONDITION
        return conflict.action_name, model_part, conflict.pbl

    def _build_model_patch(self, conflict: Conflict, model_patches: Dict[Key, PatchOperation]) -> Dict[Key, PatchOperation]:
        """
        branch_2 logic:

        1. If there isn't a model patch with this (action, part, pbl):
           1.1 conflict.type == REQUIRE_EFFECT_VS_CANNOT -> add REQUIRED_EFFECT
           1.2 conflict.type == FORBID_EFFECT_VS_MUST    -> add FORBIDDEN_EFFECT
           1.3 conflict.type == FORBID_PRECOND_VS_IS     -> add FORBIDDEN_PRECOND

        2. If there *is* a model patch with this (action, part, pbl):
           2.1 if it's an EFFECT patch (REQUIRED/ FORBIDDEN_EFFECT) -> flip it
           2.2 if it's a PRECOND patch (FORBIDDEN_PRECOND)          -> remove it
        """

        key: Key = self._conflict_to_key(conflict)

        old = model_patches
        new = copy(old)  # shallow copy is enough (Key & enums are immutable)

        existing_op: PatchOperation | None = old.get(key)

        # 1) No existing model patch for this Key
        if existing_op is None:
            if conflict.conflict_type == ConflictType.REQUIRE_EFFECT_VS_CANNOT:
                new[key] = PatchOperation.REQUIRE

            elif conflict.conflict_type == ConflictType.FORBID_EFFECT_VS_MUST:
                new[key] = PatchOperation.FORBID

            elif conflict.conflict_type == ConflictType.FORBID_PRECOND_VS_IS:
                new[key] = PatchOperation.FORBID

        # 2) There *is* a model patch for this Key
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
        Construct a fresh SimpleNoisyPisamLearner and run
        learn_action_model_with_conflicts(T, P) on it, given the current state.
        """
        # Convert model_constraints -> set of ModelLevelPatch for the learner
        model_patches: Set[ModelLevelPatch] = {
            ModelLevelPatch(
                action_name=action_name,
                model_part=part,
                pbl=pbl,
                operation=op,
            )
            for (action_name, part, pbl), op in model_constraints.items()
        }

        # Fresh deep copy of the partial domain template for this branch
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

        # ----- model constraints diff -----
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

        model_added = {
            key: final_constraints[key]
            for key in added_keys
        }
        model_removed = {
            key: initial_constraints[key]
            for key in removed_keys
        }

        # ----- fluent patches diff -----
        fluent_added = final_fluent_patches - initial_fluent_patches
        fluent_removed = initial_fluent_patches - final_fluent_patches

        return {
            "model_patches_added": model_added,
            "model_patches_removed": model_removed,
            "model_patches_changed": changed_ops,
            "fluent_patches_added": fluent_added,
            "fluent_patches_removed": fluent_removed,
        }
