from __future__ import annotations

import json
import logging
import statistics
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, Set, Tuple, List, Sequence, Optional

from pddl_plus_parser.models import Domain, Observation
from sam_learning.core import LearnerDomain
from utilities import NegativePreconditionPolicy

from src.pi_sam.noisy_pisam.noisy_learner_mixin import NoisyLearnerMixin
from src.pi_sam.noisy_pisam.noisy_pisam_learning import NoisyPisamLearner
from src.pi_sam.noisy_pisam.noisy_sam_learning import NoisySAMLearner
from src.pi_sam.noisy_pisam.typings import (
    Conflict,
    ConflictType,
    FluentLevelPatch,
    ModelLevelPatch,
    ModelPart,
    PatchOperation,
    ParameterBoundLiteral,
    ConflictPriority,
)
from src.pi_sam.plan_denoising.frontier import (
    Key,
    SearchMode,
    SearchNode,
    Frontier,
    make_frontier,
)


class DefaultSearchLogger(logging.Logger):
    """A logger that also implements log_node(), used when user does not supply a logger."""

    def log_node(self, **kwargs):
        msg = "NODE | " + ", ".join(f"{k}={v}" for k, v in kwargs.items())
        self.debug(msg)


class ConflictDrivenPatchSearchBase(ABC):
    """
    Conflict-driven patch search with state-based pruning.

    State = (model_constraints, fluent_patches)

      model_constraints:
          Dict[(action_name, ModelPart, PBL) -> PatchOperation]
          Absent key => UNCONSTRAINED.

      fluent_patches:
          Set[FluentLevelPatch], describing data repairs.

    Subclasses implement ``_create_learner`` to select the SAM-family
    learner (NoisyPisamLearner, NoisySAMLearner, etc.).
    """

    def __init__(
        self,
        partial_domain_template: Domain,
        negative_preconditions_policy: NegativePreconditionPolicy = NegativePreconditionPolicy.hard,
        seed: int = 42,
        logger: Optional[object] = None,
        search_mode: SearchMode = SearchMode.ANYTIME_DFS,
        conflict_free_models_dir: Optional[Path] = None,
        save_t_prime_fn: Optional[Callable[[List[Observation], Path], None]] = None,
        fluent_patch_cost: float = 1.0,
        fluent_patch_weight: float = 1.0,
        model_patch_cost: float = 1.0,
        model_constraint_weight: float = 0.0,
        **kwargs,
    ):
        if kwargs:
            unexpected_kwargs = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected_kwargs}")

        self.partial_domain_template = partial_domain_template
        self.negative_preconditions_policy = negative_preconditions_policy
        self.seed = seed
        self.search_mode = search_mode
        self.conflict_free_models_dir = conflict_free_models_dir
        self.save_t_prime_fn = save_t_prime_fn
        self.fluent_patch_cost = fluent_patch_cost
        self.fluent_patch_weight = fluent_patch_weight
        self.model_patch_cost = model_patch_cost
        self.model_constraint_weight = model_constraint_weight
        self._conflict_free_model_counter = 0

        if logger is None:
            logger = DefaultSearchLogger(f"{__name__}.ConflictDrivenPatchSearch")
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)

        self.logger = logger

    # ------------------------------------------------------------------
    # Abstract — subclasses provide the learner
    # ------------------------------------------------------------------

    @abstractmethod
    def _create_learner(self, domain_copy: Domain) -> NoisyLearnerMixin:
        """Create a fresh noisy learner for one expansion."""
        ...

    # ------------------------------------------------------------------
    # Cost
    # ------------------------------------------------------------------

    def _compute_cost(
        self,
        fluent_patches: Set[FluentLevelPatch],
        model_constraints: Dict[Key, PatchOperation],
    ) -> float:
        """Weighted cost of a search node.

        cost = fluent_patch_weight * fluent_patch_cost * |fluent_patches|
             + model_constraint_weight * model_patch_cost * |model_constraints|

        With default weights (fluent_patch_weight=1.0, model_constraint_weight=0.0) this reduces to
        the original cost function: fluent_patch_cost * |fluent_patches|.
        """
        return (self.fluent_patch_weight * self.fluent_patch_cost * len(fluent_patches)
                + self.model_constraint_weight * self.model_patch_cost * len(model_constraints))

    # ------------------------------------------------------------------
    # Model / timing persistence
    # ------------------------------------------------------------------

    def _save_conflict_free_model(
        self, domain: LearnerDomain, patched_observations: Optional[List[Observation]] = None,
    ) -> None:
        if self.conflict_free_models_dir is None:
            return
        self.conflict_free_models_dir.mkdir(parents=True, exist_ok=True)
        idx = self._conflict_free_model_counter
        self._conflict_free_model_counter += 1
        model_dir = self.conflict_free_models_dir / f"conflict_free_model_{idx}"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model.pddl").write_text(domain.to_pddl())
        if patched_observations and self.save_t_prime_fn:
            self.save_t_prime_fn(patched_observations, model_dir / "final_observations")
        self.logger.info(f"Saved conflict-free model {idx} to {model_dir}")

    def _save_final_model(
        self, domain: LearnerDomain, patched_observations: Optional[List[Observation]] = None,
    ) -> None:
        if self.conflict_free_models_dir is None:
            return
        self.conflict_free_models_dir.mkdir(parents=True, exist_ok=True)
        final_dir = self.conflict_free_models_dir / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        (final_dir / "model.pddl").write_text(domain.to_pddl())
        if patched_observations and self.save_t_prime_fn:
            self.save_t_prime_fn(patched_observations, final_dir / "final_observations")
        self.logger.info(f"Saved final model (no conflict-free) to {final_dir}")

    def _write_node_expansion_times(
        self, node_expansion_times: List[Dict], search_mode: SearchMode,
    ) -> None:
        if self.conflict_free_models_dir is None:
            return
        out_dir = self.conflict_free_models_dir.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "node_expansion_times.json"

        wall_times = [e["wall_time_seconds"] for e in node_expansion_times]
        summary: Dict = {}
        if wall_times:
            summary["min_wall_time_seconds"] = round(min(wall_times), 4)
            summary["max_wall_time_seconds"] = round(max(wall_times), 4)
            summary["avg_wall_time_seconds"] = round(statistics.mean(wall_times), 4)
            if len(wall_times) >= 2:
                quantiles = statistics.quantiles(wall_times, n=100)
                summary["percentile_10_wall_time_seconds"] = round(quantiles[9], 4)
                summary["percentile_25_wall_time_seconds"] = round(quantiles[24], 4)
                summary["percentile_50_wall_time_seconds"] = round(quantiles[49], 4)
                summary["percentile_90_wall_time_seconds"] = round(quantiles[89], 4)
                summary["percentile_95_wall_time_seconds"] = round(quantiles[94], 4)
                summary["percentile_99_wall_time_seconds"] = round(quantiles[98], 4)

        payload = {
            "search_mode": search_mode.value,
            "summary": summary,
            "node_expansion_times": node_expansion_times,
        }
        out_path.write_text(json.dumps(payload, indent=2))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        observations: Sequence[Observation],
        max_nodes: Optional[int] = None,
        initial_model_constraints: Optional[Dict[Key, PatchOperation]] = None,
        initial_fluent_patches: Optional[Set[FluentLevelPatch]] = None,
        timeout_seconds: int = 60,
    ) -> Tuple[
        LearnerDomain,
        List[Conflict],
        Dict[Key, PatchOperation],
        Set[FluentLevelPatch],
        int,
        Dict[str, str],
        List[Observation],
    ]:
        """
        Run the conflict-driven patch search on trajectories T (= observations).

        Returns:
            (learned_domain, conflicts, model_constraints, fluent_patches,
             patch_count, learning_report, patched_observations)
        """
        root_constraints: Dict[Key, PatchOperation] = initial_model_constraints or {}
        root_fluent_patches: Set[FluentLevelPatch] = initial_fluent_patches or set()

        frontier: Frontier = make_frontier(self.search_mode)
        frontier.push(SearchNode(
            cost=self._compute_cost(root_fluent_patches, root_constraints),
            depth=0,
            model_constraints=root_constraints,
            fluent_patches=root_fluent_patches,
        ))

        visited: Set[Tuple] = set()
        nodes_expanded = 0
        max_depth = 0
        start_time = time.time()
        terminated_by = "exhausted"

        node_expansion_times: List[Dict] = []

        # Fallback state (last expanded node, used when no solution found)
        last_domain: Optional[LearnerDomain] = None
        last_conflicts: List[Conflict] = []
        last_constraints = root_constraints
        last_fluent_patches = root_fluent_patches
        last_report: Dict[str, str] = {}
        last_cost: float = self._compute_cost(root_fluent_patches, root_constraints)

        # Best conflict-free solution found so far (anytime tracking)
        best_domain: Optional[LearnerDomain] = None
        best_constraints: Dict[Key, PatchOperation] = {}
        best_fluent_patches: Set[FluentLevelPatch] = set()
        best_report: Dict[str, str] = {}
        best_cost: float = float("inf")
        conflict_free_count: int = 0

        while frontier:
            # --- Termination checks ---
            if max_nodes is not None and nodes_expanded >= max_nodes:
                terminated_by = "max_nodes_exceeded"
                self.logger.info(f"Stopping search: reached max_nodes={max_nodes}")
                break

            if timeout_seconds is not None and (time.time() - start_time) >= timeout_seconds:
                terminated_by = "timeout_exceeded"
                self.logger.info(f"Stopping search: timeout of {timeout_seconds:.1f}s exceeded")
                break

            node: SearchNode = frontier.pop()

            state_key = self._encode_state(node.model_constraints, node.fluent_patches)
            if state_key in visited:
                continue
            visited.add(state_key)

            nodes_expanded += 1
            max_depth = max(max_depth, node.depth)
            current_cost = self._compute_cost(node.fluent_patches, node.model_constraints)
            node.cost = current_cost

            # Branch-and-bound: prune if we already have a better or equal solution
            if best_domain is not None and current_cost >= best_cost:
                continue

            # --- Expand node ---
            t0 = time.time()
            domain, conflicts, report = self._learn_with_state(
                observations, node.model_constraints, node.fluent_patches,
            )
            wall_time = time.time() - t0

            node_expansion_times.append({
                "node_index": len(node_expansion_times),
                "wall_time_seconds": round(wall_time, 4),
                "depth": node.depth,
                "cost": current_cost,
                "fluent_patch_count": len(node.fluent_patches),
                "model_constraint_count": len(node.model_constraints),
                "nodes_expanded_so_far": nodes_expanded,
            })

            last_domain = domain
            last_conflicts = conflicts
            last_constraints = node.model_constraints
            last_fluent_patches = node.fluent_patches
            last_report = report
            last_cost = current_cost

            # Post-expansion timeout check
            if timeout_seconds is not None and (time.time() - start_time) >= timeout_seconds:
                terminated_by = "timeout_exceeded"
                self.logger.info(
                    f"Stopping search: timeout of {timeout_seconds:.1f}s exceeded after expansion"
                )
                break

            # --- Conflict-free solution ---
            if not conflicts:
                conflict_free_count += 1
                patched_obs = self._apply_patches_to_observations(
                    observations, node.fluent_patches,
                )
                self._save_conflict_free_model(domain, patched_obs)

                if current_cost < best_cost:
                    best_domain = domain
                    best_constraints = node.model_constraints
                    best_fluent_patches = node.fluent_patches
                    best_report = report
                    best_cost = current_cost
                    self.logger.info(
                        f"New best solution: cost={best_cost:.2f} "
                        f"(fluent_patches={len(node.fluent_patches)}, "
                        f"model_constraints={len(node.model_constraints)}), "
                        f"nodes_expanded={nodes_expanded}, depth={node.depth}"
                    )

                if self.search_mode == SearchMode.UCS:
                    terminated_by = "solution_found_ucs"
                    break
                continue  # DFS: don't branch from solution, keep searching

            # --- Branch on conflicts ---
            conflict_groups = self._group_conflicts(conflicts)
            group = self._choose_conflict_group(conflict_groups)

            # Child A: DATA-FIX (fluent patches for all conflicts in the group)
            child_a_model_constraints = dict(node.model_constraints)
            child_a_fluent_patches = set(node.fluent_patches)
            changed = False
            for c in group:
                fp = self._build_fluent_patch(c)
                if fp not in child_a_fluent_patches:
                    child_a_fluent_patches.add(fp)
                    changed = True

            if changed:
                child_a_fluent_patches = self._dedup_patches(child_a_fluent_patches)
                frontier.push(SearchNode(
                    cost=self._compute_cost(child_a_fluent_patches, child_a_model_constraints),
                    depth=node.depth + 1,
                    model_constraints=child_a_model_constraints,
                    fluent_patches=child_a_fluent_patches,
                ))

            # Child B: MODEL-FIX (only for non-frame-axiom conflicts)
            rep_conflict = group[0]
            if rep_conflict.conflict_type not in {ConflictType.FRAME_AXIOM}:
                new_constraints = self._build_model_patch(
                    rep_conflict, dict(node.model_constraints),
                )
                if new_constraints != node.model_constraints:
                    child_b_fluent_patches = set(node.fluent_patches)
                    frontier.push(SearchNode(
                        cost=self._compute_cost(child_b_fluent_patches, new_constraints),
                        depth=node.depth + 1,
                        model_constraints=new_constraints,
                        fluent_patches=child_b_fluent_patches,
                    ))

        # ------------------------------------------------------------------
        # Build final result
        # ------------------------------------------------------------------
        total_time = time.time() - start_time
        if terminated_by == "timeout_exceeded" and timeout_seconds is not None:
            total_time = min(total_time, float(timeout_seconds))

        if best_domain is not None:
            final_domain = best_domain
            final_conflicts: List[Conflict] = []
            final_constraints = best_constraints
            final_fluent_patches = best_fluent_patches
            final_cost = best_cost
            base_report = best_report
        else:
            final_domain = last_domain
            final_conflicts = last_conflicts
            final_constraints = last_constraints
            final_fluent_patches = last_fluent_patches
            final_cost = last_cost
            base_report = last_report

        patch_diff = self._compute_patch_diff(
            initial_constraints=root_constraints,
            final_constraints=final_constraints,
            initial_fluent_patches=root_fluent_patches,
            final_fluent_patches=final_fluent_patches,
        )
        enriched_report = dict(base_report)
        enriched_report.update(
            dict(
                patch_diff=patch_diff,
                nodes_expanded=nodes_expanded,
                max_depth=max_depth,
                terminated_by=terminated_by,
                total_time_seconds=total_time,
                best_cost=(best_cost if best_domain is not None else None),
                best_fluent_patch_count=(len(best_fluent_patches) if best_domain is not None else None),
                best_model_constraint_count=(len(best_constraints) if best_domain is not None else None),
                fluent_patch_cost=self.fluent_patch_cost,
                fluent_patch_weight=self.fluent_patch_weight,
                model_patch_cost=self.model_patch_cost,
                model_constraint_weight=self.model_constraint_weight,
                conflict_free_model_count=conflict_free_count,
            )
        )

        patched_obs = self._apply_patches_to_observations(
            observations, final_fluent_patches,
        )
        if conflict_free_count == 0:
            self._save_final_model(final_domain, patched_obs)
        self._write_node_expansion_times(node_expansion_times, self.search_mode)

        return (
            final_domain,
            final_conflicts,
            final_constraints,
            final_fluent_patches,
            final_cost,
            enriched_report,
            patched_obs,
        )

    # ------------------------------------------------------------------
    # Encoding + priorities + grouping
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_state(
        model_constraints: Dict[Key, PatchOperation],
        fluent_patches: Set[FluentLevelPatch],
    ) -> Tuple:
        constraints_tuple = tuple(sorted(
            (action_name, part.value, str(pbl), op.value)
            for (action_name, part, pbl), op in model_constraints.items()
        ))
        fluent_tuple = tuple(sorted(
            (fp.observation_index, fp.component_index, fp.state_type, fp.normalized_fluent)
            for fp in fluent_patches
        ))
        return constraints_tuple, fluent_tuple

    @staticmethod
    def _conflict_priority(conflict: Conflict) -> ConflictPriority:
        if conflict.conflict_type in {
            ConflictType.FORBID_EFFECT_VS_MUST,
            ConflictType.REQUIRE_EFFECT_VS_CANNOT,
        }:
            return ConflictPriority.EFFECT
        if conflict.conflict_type == ConflictType.FRAME_AXIOM:
            return ConflictPriority.FRAME_AXIOM
        return ConflictPriority.OTHER

    def _group_key(
        self, conflict: Conflict,
    ) -> Tuple[ConflictPriority, str, ModelPart, ParameterBoundLiteral, ConflictType]:
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
        groups: Dict[Tuple, List[Conflict]] = {}
        for c in conflicts:
            groups.setdefault(self._group_key(c), []).append(c)
        return list(groups.values())

    def _choose_conflict_group(self, groups: List[List[Conflict]]) -> List[Conflict]:
        def group_score(g: List[Conflict]) -> Tuple[int, int, int]:
            rep = min(
                g,
                key=lambda c: (
                    int(self._conflict_priority(c)),
                    c.observation_index,
                    c.component_index,
                ),
            )
            return int(self._conflict_priority(rep)), rep.observation_index, rep.component_index

        return min(groups, key=group_score)

    # ------------------------------------------------------------------
    # Fluent patches & model patches
    # ------------------------------------------------------------------

    @staticmethod
    def _build_fluent_patch(conflict: Conflict) -> FluentLevelPatch:
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
        model_part = (
            ModelPart.PRECONDITION
            if conflict.conflict_type == ConflictType.FORBID_PRECOND_VS_IS
            else ModelPart.EFFECT
        )
        return conflict.action_name, model_part, conflict.pbl

    @staticmethod
    def _dedup_patches(patches: Set[FluentLevelPatch]) -> Set[FluentLevelPatch]:
        """Remove (next at (obs,comp), prev at (obs,comp+1)) pairs with same fluent."""
        next_p = {
            (p.observation_index, p.component_index, p.normalized_fluent): p
            for p in patches if p.state_type == "next"
        }
        prev_p = {
            (p.observation_index, p.component_index, p.normalized_fluent): p
            for p in patches if p.state_type == "prev"
        }
        remove: Set[FluentLevelPatch] = set()
        for (obs, comp, fluent), p_next in next_p.items():
            p_prev = prev_p.get((obs, comp + 1, fluent))
            if p_prev:
                remove.add(p_next)
                remove.add(p_prev)
        return {p for p in patches if p not in remove}

    def _build_model_patch(
        self,
        conflict: Conflict,
        model_patches: Dict[Key, PatchOperation],
    ) -> Dict[Key, PatchOperation]:
        """
        Build/update model constraints based on a representative conflict.

        Rules:
          - map conflict type to a desired operation
          - no-op if key already has the desired operation
          - otherwise set/replace key with desired operation
        """
        key: Key = self._conflict_to_key(conflict)
        desired_op = {
            ConflictType.REQUIRE_EFFECT_VS_CANNOT: PatchOperation.FORBID,
            ConflictType.FORBID_EFFECT_VS_MUST: PatchOperation.REQUIRE,
        }.get(conflict.conflict_type)
        if desired_op is None:
            return dict(model_patches)

        # No-op if already at desired op; otherwise set/replace to desired op.
        if model_patches.get(key) == desired_op:
            return dict(model_patches)
        return {**model_patches, key: desired_op}

    # ------------------------------------------------------------------
    # Learner interaction
    # ------------------------------------------------------------------

    def _learn_with_state(
        self,
        observations: Sequence[Observation],
        model_constraints: Dict[Key, PatchOperation],
        fluent_patches: Set[FluentLevelPatch],
    ) -> Tuple[LearnerDomain, List[Conflict], Dict[str, str]]:
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
        learner = self._create_learner(domain_copy)
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
        domain_copy: Domain = deepcopy(self.partial_domain_template)
        learner = self._create_learner(domain_copy)
        learner.set_patches(fluent_patches=fluent_patches, model_patches=set())
        return learner.apply_fluent_patches(list(observations))

    @staticmethod
    def _compute_patch_diff(
        initial_constraints: Dict[Key, PatchOperation],
        final_constraints: Dict[Key, PatchOperation],
        initial_fluent_patches: Set[FluentLevelPatch],
        final_fluent_patches: Set[FluentLevelPatch],
    ) -> Dict[str, object]:
        initial_keys = set(initial_constraints.keys())
        final_keys = set(final_constraints.keys())
        added_keys = final_keys - initial_keys
        removed_keys = initial_keys - final_keys
        common_keys = initial_keys & final_keys

        return {
            "model_patches_added": {k: final_constraints[k] for k in added_keys},
            "model_patches_removed": {k: initial_constraints[k] for k in removed_keys},
            "model_patches_changed": {
                k: (initial_constraints[k], final_constraints[k])
                for k in common_keys
                if initial_constraints[k] != final_constraints[k]
            },
            "fluent_patches_added": final_fluent_patches - initial_fluent_patches,
            "fluent_patches_removed": initial_fluent_patches - final_fluent_patches,
        }


# ======================================================================
# Concrete subclasses
# ======================================================================

class ConflictDrivenPatchSearchPISAM(ConflictDrivenPatchSearchBase):
    """Conflict-driven patch search using PI-SAM (partial observability)."""

    def _create_learner(self, domain_copy: Domain) -> NoisyPisamLearner:
        return NoisyPisamLearner(
            partial_domain=domain_copy,
            negative_preconditions_policy=self.negative_preconditions_policy,
            seed=self.seed,
        )


class ConflictDrivenPatchSearchSAM(ConflictDrivenPatchSearchBase):
    """Conflict-driven patch search using SAM (full observability)."""

    def _create_learner(self, domain_copy: Domain) -> NoisySAMLearner:
        return NoisySAMLearner(
            partial_domain=domain_copy,
            negative_preconditions_policy=self.negative_preconditions_policy,
        )


# Backward-compatible alias — callers that imported the old name get the PI-SAM variant
ConflictDrivenPatchSearch = ConflictDrivenPatchSearchPISAM
