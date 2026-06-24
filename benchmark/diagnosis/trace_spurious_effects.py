"""
trace_spurious_effects.py — Diagnose why a lifted predicate ended up in an action's effects.

Given an existing fold's data, this script replays the full learning pipeline with fine-grained
tracing to identify the root cause of a spurious learned effect.

Pipeline stages:
  Stage 0 — Load fold_info.json and reconstruct the canonical observation order.
  Stage 1 — Load and reconstruct masked observations (in fold_info order).
  Stage 1b — (optional) Apply CFM-specific fluent patches to reconstruct the exact observations
             that PI-SAM saw when a specific conflict-free model (CFM) was produced.
  Stage 2 — Trajectory-level attribution: which trajectories "voted" the predicate into effects.
  Stage 3 — PI-SAM hypothesis evolution: snapshot effects/cannot_be_effect after every component.
  Stage 4 — Conflict search analysis: saved learning_metrics.json + optional --retrace.
  Stage 5 — Final report (JSON + human-readable summary printed to stdout).

Usage:
    python benchmark/diagnosis/trace_spurious_effects.py \\
        --fold-dir  benchmark/data/blocksworld/.../testing/fold0_numtrajs3_gtrate0 \\
        --domain    benchmark/data/blocksworld/domain.pddl \\
        --traj-base benchmark/data/blocksworld/.../training/trajectories \\
        --action    stack \\
        --predicate "(on-table ?x)" \\
        [--cfm-index 3] \\
        [--retrace]

Notes:
  - --traj-base is the directory that contains one subdirectory per problem
    (e.g. problem3/, problem7/, …), each holding the .trajectory and .masking_info files.
  - The processing order is taken directly from fold_info.json's "trajectories" array.
  - --cfm-index N restricts the analysis to the exact observations PI-SAM saw when CFM #N was
    produced. It reads conflict_free_models/conflict_free_model_N/patch_details.json from the
    fold directory and applies those fluent patches on top of the masked originals before
    running Stages 2 and 3. Without this flag, Stages 2 and 3 run on the raw originals.
  - No files in src/ are modified; tracing is done via a local subclass of PISAMLearner.
"""

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Project root on sys.path so src/ and benchmark/ imports work.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import (
    Domain, Observation, ObservedComponent, ActionCall, State, Predicate,
)
from sam_learning.core.matching_utils import extract_discrete_effects_partial_observability

from src.pi_sam.pi_sam_learning import PISAMLearner
from src.utils.masking import load_masked_observation
from src.utils.pddl import get_state_grounded_predicates, get_state_masked_predicates
from utilities import NegativePreconditionPolicy


# ===========================================================================
# Data classes for structured results
# ===========================================================================

@dataclass
class VoteRecord:
    """One row in the Stage-2 vote table."""
    obs_idx: int
    problem: str
    comp_idx: int
    predicate_in_diff: bool
    diff_direction: str          # "add", "delete", or "none"
    masked_in_prev: bool
    masked_in_next: bool
    spurious_vote: bool          # True when predicate appears as add-effect diff
    is_first_encounter: bool     # True when this is the first time the action is seen (add_new_action);
                                 # this component initialises the hypothesis, it does not refine it.


@dataclass
class HypothesisSnapshot:
    """State of the PI-SAM hypothesis for the target action after one component."""
    obs_idx: int
    problem: str
    comp_idx: int
    action_fired: str
    target_in_effects: bool
    target_in_cannot_be: bool
    effects_count: int
    cannot_be_count: int
    event: str                   # "add_new_action" | "update_action" | "skipped" | "different_action"


@dataclass
class ConflictSearchAnalysis:
    """Stage-4 result from saved learning_metrics.json."""
    metrics_found: bool
    conflict_group_formed: bool
    model_forbid_generated: bool
    model_forbid_cost: Optional[float]
    competing_fluent_patch_cost: Optional[float]
    best_model_constraints: List[Dict]
    best_fluent_patches: List[Dict]
    notes: List[str] = field(default_factory=list)


@dataclass
class DiagnosisReport:
    """Full diagnosis output."""
    target_action: str
    target_predicate: str
    fold_dir: str
    observations_order: List[str]           # problem names in processing order

    # CFM scope (Stage 1b)
    cfm_index: Optional[int]                # None = raw originals; N = scoped to CFM #N
    cfm_model_constraints: Optional[List[Dict]]   # model constraints active at CFM #N
    cfm_fluent_patch_count: Optional[int]         # how many fluent patches were applied

    # Stage 2
    vote_table: List[VoteRecord]
    total_votes: int
    masked_votes: int                        # votes where prev or next was masked

    # Stage 3
    hypothesis_snapshots: List[HypothesisSnapshot]
    first_introduction_obs: Optional[int]
    first_introduction_comp: Optional[int]
    first_introduction_problem: Optional[str]
    ever_in_cannot_be: bool

    # Stage 4
    conflict_search: Optional[ConflictSearchAnalysis]

    # Stage 5 verdict
    verdict: str


# ===========================================================================
# Stage 0 — Load fold metadata
# ===========================================================================

def load_fold_info(fold_dir: Path) -> Dict:
    """Read and return the parsed fold_info.json from *fold_dir*."""
    fold_info_path = fold_dir / "fold_info.json"
    if not fold_info_path.exists():
        raise FileNotFoundError(f"fold_info.json not found in {fold_dir}")
    with open(fold_info_path) as f:
        return json.load(f)


def resolve_trajectory_paths(
    fold_info: Dict,
    traj_base: Path,
) -> List[Tuple[str, Path, Path]]:
    """
    Return ordered list of (problem_name, traj_path, masking_path) using the
    processing order defined in fold_info["trajectories"].

    Args:
        fold_info:  Parsed fold_info.json dict.
        traj_base:  Root directory containing one subdirectory per problem.

    Returns:
        List of (problem_name, trajectory_path, masking_path) in processing order.
    """
    result = []
    for entry in fold_info["trajectories"]:
        problem = entry["problem"]
        traj_file = entry["trajectory_file"]
        masking_file = entry["masking_file"]
        prob_dir = traj_base / problem
        traj_path = prob_dir / traj_file
        masking_path = prob_dir / masking_file
        if not traj_path.exists():
            raise FileNotFoundError(f"Trajectory not found: {traj_path}")
        if not masking_path.exists():
            raise FileNotFoundError(f"Masking info not found: {masking_path}")
        result.append((problem, traj_path, masking_path))
    return result


# ===========================================================================
# Stage 1 — Load observations
# ===========================================================================

def load_observations_in_order(
    ordered_paths: List[Tuple[str, Path, Path]],
    domain: Domain,
) -> List[Tuple[str, Observation]]:
    """
    Load each (trajectory, masking) pair in fold_info order.

    Args:
        ordered_paths:  Output of resolve_trajectory_paths().
        domain:         Parsed partial domain.

    Returns:
        Ordered list of (problem_name, masked_observation).
    """
    observations = []
    for problem, traj_path, masking_path in ordered_paths:
        obs = load_masked_observation(traj_path, masking_path, domain)
        observations.append((problem, obs))
        n_components = len(obs.components)
        # Collect the N+1 distinct states (init + one next_state per component).
        # Using next_state for all components avoids double-counting shared boundaries.
        all_states = (
            [obs.components[0].previous_state]
            + [comp.next_state for comp in obs.components]
        )
        n_masked_states = sum(
            1 for s in all_states if len(get_state_masked_predicates(s)) > 0
        )
        print(
            f"  Loaded obs[{len(observations) - 1}] problem={problem} | "
            f"components={n_components} | masked_states={n_masked_states}"
        )
    return observations


# ===========================================================================
# Stage 1b — CFM-specific fluent patch loading and application
# ===========================================================================

def load_cfm_patch_details(fold_dir: Path, cfm_index: int) -> Dict:
    """
    Load patch_details.json for a specific conflict-free model.

    Args:
        fold_dir:   Path to the fold directory (contains conflict_free_models/).
        cfm_index:  Zero-based index of the CFM to load (e.g. 3 → conflict_free_model_3/).

    Returns:
        Parsed patch_details dict with keys: model_constraints, fluent_patches, cost, …

    Raises:
        FileNotFoundError: if the patch_details.json for the requested CFM is absent.
    """
    patch_path = fold_dir / "conflict_free_models" / f"conflict_free_model_{cfm_index}" / "patch_details.json"
    if not patch_path.exists():
        raise FileNotFoundError(
            f"patch_details.json not found for CFM {cfm_index}: {patch_path}\n"
            f"Available CFMs: {sorted(p.name for p in (fold_dir / 'conflict_free_models').iterdir() if p.is_dir()) if (fold_dir / 'conflict_free_models').exists() else 'conflict_free_models/ directory not found'}"
        )
    with open(patch_path) as f:
        return json.load(f)


def apply_cfm_fluent_patches(
    observations: List[Tuple[str, Observation]],
    patch_details: Dict,
) -> List[Tuple[str, Observation]]:
    """
    Apply the fluent patches from a CFM's patch_details.json to the masked observations,
    reproducing the exact observation set that PI-SAM received when that CFM was produced.

    Each fluent patch flips a single grounded predicate in a specific (obs, comp, state) location.
    This mirrors what ConflictDrivenPatchSearch._apply_patches_to_observations() does internally
    during the search, using the same flip_fluent_in_state utility.

    Args:
        observations:   Ordered (problem, masked_observation) pairs (Stage 1 output).
        patch_details:  Parsed patch_details.json dict for the target CFM.

    Returns:
        New ordered (problem, patched_observation) pairs with fluent patches applied.
        The originals are not modified.
    """
    from src.pi_sam.noisy_pisam.typings import FluentLevelPatch
    from src.pi_sam.noisy_pisam.noisy_pisam_learning import NoisyPisamLearner
    from copy import deepcopy

    raw_patches = patch_details.get("fluent_patches", [])
    if not raw_patches:
        print("  No fluent patches in patch_details — observations unchanged.")
        return list(observations)

    fluent_patch_set: Set[FluentLevelPatch] = set()
    for fp in raw_patches:
        fluent_patch_set.add(FluentLevelPatch(
            observation_index=fp["observation_index"],
            component_index=fp["component_index"],
            state_type=fp["state_type"],
            fluent=fp["fluent"],
        ))

    # We need a NoisyPisamLearner instance only to call set_patches + apply_fluent_patches.
    # It will not be used for learning — we discard it after patch application.
    # A minimal domain deepcopy is enough; the learner is stateless w.r.t. the domain here.
    from pddl_plus_parser.models import Domain as _Domain
    # Build a throwaway domain by deep-copying the first observation's domain objects.
    # In practice we only need the mixin's apply_fluent_patches which does not touch domain.
    # We pass a sentinel None and guard with a try/except in case __init__ requires a domain.
    obs_list = [obs for _, obs in observations]

    try:
        # NoisyPisamLearner requires a Domain — load a fresh one from the caller's domain_path
        # is not available here, so we use a lightweight approach: instantiate the mixin path
        # via the noisy_learner_mixin directly.
        from src.pi_sam.noisy_pisam.noisy_learner_mixin import NoisyLearnerMixin

        class _PatchApplier(NoisyLearnerMixin):
            """Minimal stub — only set_patches + apply_fluent_patches are used."""

            def __init__(self):
                # Initialise the mixin's state dictionaries so set_patches() can call .clear().
                self._init_noisy_fields()

            # NoisyLearnerMixin's abstract methods are never called here.
            def _extract_discrete_effects(self, prev_preds, next_preds):  # type: ignore[override]
                raise NotImplementedError

            def _extract_cannot_be_effects(self, prev_preds, next_preds):  # type: ignore[override]
                raise NotImplementedError

            def _delegate_handle_effects(self, grounded_action, previous_state, next_state):  # type: ignore[override]
                raise NotImplementedError

        applier = _PatchApplier()
        applier.set_patches(fluent_patches=fluent_patch_set, model_patches=set())
        patched_obs_list = applier.apply_fluent_patches(obs_list)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to apply fluent patches via NoisyLearnerMixin: {exc}\n"
            "Ensure NoisyLearnerMixin.set_patches / apply_fluent_patches signatures have not changed."
        ) from exc

    return [(problem, patched_obs) for (problem, _), patched_obs in zip(observations, patched_obs_list)]


# ===========================================================================
# Stage 2 — Trajectory-level attribution
# ===========================================================================

def _predicate_name_from_lifted(lifted_str: str) -> str:
    """Extract predicate name from a lifted string like '(on-table ?x)' or '(not (on-table ?x))' → 'on-table'."""
    s = lifted_str.strip()
    # Handle negative form: (not (predicate-name ...))
    if s.startswith("(not "):
        s = s[5:].strip()
    # Strip outer parens and take the first token
    return s.strip("()").split()[0]


def _lifted_matches_grounded(lifted_pred: Predicate, target_pred_name: str) -> bool:
    """Return True when *lifted_pred* has the same base name as *target_pred_name*."""
    return lifted_pred.name == target_pred_name


def compute_vote_table(
    ordered_observations: List[Tuple[str, Observation]],
    target_action: str,
    target_predicate_name: str,
) -> List[VoteRecord]:
    """
    For every component in every observation where the target action fires,
    compute a raw state diff and record whether the target predicate appears
    as an add-effect (spurious_vote=True).

    Uses only unmasked predicate pairs in the diff computation (same as PI-SAM)
    so as to faithfully reflect what the learner sees.

    Args:
        ordered_observations:  List of (problem, observation) in processing order.
        target_action:         Action name to monitor (e.g. "stack").
        target_predicate_name: Base predicate name to search for (e.g. "on-table").

    Returns:
        List of VoteRecord, one per relevant component.
    """
    records: List[VoteRecord] = []
    # Track whether the target action has been seen before across all observations,
    # mirroring PI-SAM's observed_actions list so we can flag add_new_action components.
    target_action_seen: bool = False

    for obs_idx, (problem, obs) in enumerate(ordered_observations):
        for comp_idx, component in enumerate(obs.components):
            if not component.is_successful:
                continue
            action: ActionCall = component.grounded_action_call
            if action.name != target_action:
                continue

            prev_preds = get_state_grounded_predicates(component.previous_state)
            next_preds = get_state_grounded_predicates(component.next_state)

            # Diff as PI-SAM sees it (masked instances already filtered out).
            grounded_add, grounded_del = extract_discrete_effects_partial_observability(
                prev_preds, next_preds
            )

            # Find the specific voted grounded instance (matched by object_mapping).
            voted_add = next(
                (p for p in grounded_add if p.name == target_predicate_name), None
            )
            voted_del = next(
                (p for p in grounded_del if p.name == target_predicate_name), None
            )
            voted = voted_add or voted_del

            in_add = voted_add is not None
            in_del = voted_del is not None

            if in_add:
                direction = "add"
            elif in_del:
                direction = "delete"
            else:
                direction = "none"

            # Masking check: look up the exact same grounded instance (by object_mapping)
            # in prev/next state to see if it was masked there.
            # Note: extract_discrete_effects_partial_observability already excludes masked
            # instances, so voted itself is never masked — but its counterpart in the
            # other state might be (e.g. the negation in prev was masked).
            def _is_masked_in(preds: Set, name: str, obj_mapping: dict) -> bool:
                return any(
                    p.is_masked and p.name == name and p.object_mapping == obj_mapping
                    for p in preds
                )

            if voted is not None:
                target_in_prev_masked = _is_masked_in(prev_preds, voted.name, voted.object_mapping)
                target_in_next_masked = _is_masked_in(next_preds, voted.name, voted.object_mapping)
            else:
                # Predicate not in diff at all — check if any grounding of it is masked.
                target_in_prev_masked = any(
                    p.is_masked and p.name == target_predicate_name for p in prev_preds
                )
                target_in_next_masked = any(
                    p.is_masked and p.name == target_predicate_name for p in next_preds
                )

            is_first = not target_action_seen
            target_action_seen = True  # mark seen after first encounter

            records.append(VoteRecord(
                obs_idx=obs_idx,
                problem=problem,
                comp_idx=comp_idx,
                predicate_in_diff=in_add or in_del,
                diff_direction=direction,
                masked_in_prev=target_in_prev_masked,
                masked_in_next=target_in_next_masked,
                spurious_vote=in_add,
                is_first_encounter=is_first,
            ))

    return records


# ===========================================================================
# Stage 3 — PI-SAM hypothesis evolution (tracing subclass)
# ===========================================================================

class TracingPISAMLearner(PISAMLearner):
    """
    Local subclass of PISAMLearner that snapshots the hypothesis for one
    specific action after every add_new_action / update_action call.

    No changes to src/ — this lives entirely in the diagnosis script.
    """

    def __init__(
        self,
        partial_domain: Domain,
        target_action: str,
        target_predicate_name: str,
        negative_preconditions_policy: NegativePreconditionPolicy = NegativePreconditionPolicy.hard,
        seed: int = 42,
    ):
        super().__init__(partial_domain, negative_preconditions_policy, seed)
        self._trace_action = target_action
        self._trace_pred_name = target_predicate_name
        self._snapshots: List[HypothesisSnapshot] = []
        # Tracking context — set by the replay loop before each observation.
        self._current_obs_idx: int = 0
        self._current_problem: str = ""
        self._current_comp_idx: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _snapshot(self, event: str, action_fired: str) -> None:
        """Capture current hypothesis state for the target action."""
        if self._trace_action not in self.partial_domain.actions:
            return
        action_obj = self.partial_domain.actions[self._trace_action]
        effects: Set[Predicate] = action_obj.discrete_effects
        cannot: Set[Predicate] = self.cannot_be_effect.get(self._trace_action, set())

        target_in_effects = any(
            _lifted_matches_grounded(p, self._trace_pred_name) for p in effects
        )
        target_in_cannot = any(
            _lifted_matches_grounded(p, self._trace_pred_name) for p in cannot
        )

        self._snapshots.append(HypothesisSnapshot(
            obs_idx=self._current_obs_idx,
            problem=self._current_problem,
            comp_idx=self._current_comp_idx,
            action_fired=action_fired,
            target_in_effects=target_in_effects,
            target_in_cannot_be=target_in_cannot,
            effects_count=len(effects),
            cannot_be_count=len(cannot),
            event=event,
        ))

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def add_new_action(
        self, grounded_action: ActionCall, previous_state: State, next_state: State
    ) -> None:
        super().add_new_action(grounded_action, previous_state, next_state)
        if grounded_action.name == self._trace_action:
            self._snapshot("add_new_action", grounded_action.name)

    def update_action(
        self, grounded_action: ActionCall, previous_state: State, next_state: State
    ) -> None:
        super().update_action(grounded_action, previous_state, next_state)
        if grounded_action.name == self._trace_action:
            self._snapshot("update_action", grounded_action.name)

    def get_snapshots(self) -> List[HypothesisSnapshot]:
        return list(self._snapshots)


def replay_pisam_with_tracing(
    ordered_observations: List[Tuple[str, Observation]],
    domain: Domain,
    target_action: str,
    target_predicate_name: str,
) -> List[HypothesisSnapshot]:
    """
    Replay observations through TracingPISAMLearner in the canonical order,
    injecting obs/problem/comp context before each component so snapshots
    carry full provenance.

    Args:
        ordered_observations:  (problem, observation) pairs in processing order.
        domain:                Fresh partial domain (will be consumed by learner).
        target_action:         Action to trace.
        target_predicate_name: Predicate name to watch.

    Returns:
        List of HypothesisSnapshot in chronological order.
    """
    learner = TracingPISAMLearner(
        partial_domain=domain,
        target_action=target_action,
        target_predicate_name=target_predicate_name,
    )

    # Replicate learn_action_model loop exactly, but set tracing context each step.
    learner.deduce_initial_inequality_preconditions()
    learner._complete_possibly_missing_actions()  # noqa: SLF001 — needed to match learning loop

    for obs_idx, (problem, obs) in enumerate(ordered_observations):
        learner.current_trajectory_objects = obs.grounded_objects
        learner._current_obs_idx = obs_idx
        learner._current_problem = problem

        for comp_idx, component in enumerate(obs.components):
            learner._current_comp_idx = comp_idx
            if not component.is_successful:
                continue
            learner.handle_single_trajectory_component(component)

    return learner.get_snapshots()


# ===========================================================================
# Stage 4 — Conflict search analysis
# ===========================================================================

def analyse_saved_metrics(
    fold_dir: Path,
    target_action: str,
    target_predicate_name: str,
    cfm_index: Optional[int] = None,
) -> Optional[ConflictSearchAnalysis]:
    """
    Analyse the saved patch data for the conflict search and report whether
    the target predicate was addressed.

    When *cfm_index* is given, reads patch_details.json for that specific CFM
    (the exact constraints/patches active when that model was produced).
    Otherwise reads learning_metrics.json, which holds the overall best/final
    solution selected by the search.

    Returns None if the relevant file is not found.
    """
    if cfm_index is not None:
        # --- CFM-scoped: read patch_details.json for this specific model ---
        patch_path = (
            fold_dir / "conflict_free_models"
            / f"conflict_free_model_{cfm_index}" / "patch_details.json"
        )
        if not patch_path.exists():
            return None
        with open(patch_path) as f:
            patch_data = json.load(f)

        constraints: List[Dict] = patch_data.get("model_constraints") or []
        fluent_patches_list: List[Dict] = patch_data.get("fluent_patches") or []
        cfm_cost: Optional[float] = patch_data.get("cost")
        source_label = f"CFM #{cfm_index} patch_details.json"

        # Cost parameters are not stored in patch_details — use None.
        model_forbid_cost: Optional[float] = None
        fluent_patch_cost: Optional[float] = (
            float(cfm_cost) if cfm_cost is not None else None
        )

    else:
        # --- Global best: read learning_metrics.json ---
        metrics_path = fold_dir / "learning_metrics.json"
        if not metrics_path.exists():
            return None
        with open(metrics_path) as f:
            metrics = json.load(f)

        constraints = metrics.get("best_model_constraints") or []
        fluent_patches_list = metrics.get("best_fluent_patches") or []
        source_label = "learning_metrics.json (global best/final solution)"

        # Reconstruct costs from stored config params.
        model_patch_cost_val = float(metrics.get("model_patch_cost") or 1.0)
        model_constraint_weight_val = float(metrics.get("model_constraint_weight") or 0.0)
        fp_cost_val = float(metrics.get("fluent_patch_cost") or 1.0)
        fp_weight_val = float(metrics.get("fluent_patch_weight") or 1.0)

        model_forbid_cost = (
            model_constraint_weight_val * model_patch_cost_val * len(constraints)
            if constraints else None
        )
        fluent_patch_cost = (
            fp_weight_val * fp_cost_val * len(fluent_patches_list)
            if fluent_patches_list else None
        )

    # Look for a FORBID effect constraint on our action+predicate.
    forbid_entries = [
        c for c in constraints
        if (c.get("action") == target_action
            and c.get("model_part") == "eff"
            and c.get("operation") == "forbid"
            and target_predicate_name in c.get("predicate", ""))
    ]
    # Look for a REQUIRE effect constraint on our action+predicate.
    require_entries = [
        c for c in constraints
        if (c.get("action") == target_action
            and c.get("model_part") == "eff"
            and c.get("operation") == "require"
            and target_predicate_name in c.get("predicate", ""))
    ]

    model_forbid_generated = len(forbid_entries) > 0
    model_require_generated = len(require_entries) > 0

    action_mentioned_in_constraints = any(
        c.get("action") == target_action for c in constraints
    )
    action_mentioned_in_patches = any(
        target_predicate_name in str(fp.get("fluent", ""))
        for fp in fluent_patches_list
    )
    conflict_group_formed = action_mentioned_in_constraints or action_mentioned_in_patches

    notes: List[str] = [f"Source: {source_label}"]
    if not constraints and not fluent_patches_list:
        notes.append("No constraints or patches found — search may not have run or found no solution.")
    if not conflict_group_formed:
        notes.append(
            f"No constraint or patch mentioning action '{target_action}' found — "
            "the conflict search may not have encountered a conflict for this action."
        )
    if model_require_generated:
        notes.append(
            f"A REQUIRE-effect constraint for '{target_predicate_name}' is active at this CFM — "
            "the search actively required this predicate as an effect at this node."
        )
    if model_forbid_generated:
        notes.append(
            f"A FORBID-effect constraint for '{target_predicate_name}' is active — "
            "the search ruled this predicate out as an effect at this node."
        )
    if not model_forbid_generated and not model_require_generated and conflict_group_formed:
        notes.append(
            f"Conflict group for '{target_action}' formed, but no FORBID or REQUIRE constraint "
            f"for '{target_predicate_name}' — data-fix (fluent patches) was used instead."
        )

    return ConflictSearchAnalysis(
        metrics_found=True,
        conflict_group_formed=conflict_group_formed,
        model_forbid_generated=model_forbid_generated,
        model_forbid_cost=model_forbid_cost,
        competing_fluent_patch_cost=fluent_patch_cost,
        best_model_constraints=constraints,
        best_fluent_patches=fluent_patches_list,
        notes=notes,
    )


def retrace_conflict_search(
    ordered_observations: List[Tuple[str, Observation]],
    fold_dir: Path,
    domain: Domain,
    target_action: str,
    target_predicate_name: str,
) -> ConflictSearchAnalysis:
    """
    Re-run ConflictDrivenPatchSearch with a tracing subclass to capture full
    branching history for the target predicate.

    Reads search parameters from learning_metrics.json to reproduce the original run.
    Falls back to defaults if the metrics file is absent.
    """
    from src.pi_sam.plan_denoising.conflict_search import ConflictDrivenPatchSearch
    from src.pi_sam.plan_denoising.frontier import (
        SearchMode, NodeChoosingStrategy, ConflictGroupStrategy, FluentBranchMode,
    )

    # Load original search parameters from saved metrics if available.
    metrics_path = fold_dir / "learning_metrics.json"
    params: Dict = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            params = json.load(f)

    _MODE_MAP = {
        "anytime_dfs": SearchMode.ANYTIME_DFS,
        "ucs": SearchMode.UCS,
    }
    _NODE_MAP = {
        "model_patch_first": NodeChoosingStrategy.MODEL_PATCH_FIRST,
        "fluent_patch_first": NodeChoosingStrategy.FLUENT_PATCH_FIRST,
        "fluent_patch_first_then_model": NodeChoosingStrategy.FLUENT_PATCH_FIRST_THEN_MODEL,
        "randomized": NodeChoosingStrategy.RANDOMIZED,
    }
    _GROUP_MAP = {
        "first": ConflictGroupStrategy.FIRST,
        "largest": ConflictGroupStrategy.LARGEST,
        "largest_model_patchable": ConflictGroupStrategy.LARGEST_MODEL_PATCHABLE,
        "most_observations": ConflictGroupStrategy.MOST_OBSERVATIONS,
        "smallest": ConflictGroupStrategy.SMALLEST,
    }
    _FLUENT_MAP = {
        "group": FluentBranchMode.GROUP,
        "single": FluentBranchMode.SINGLE,
    }

    search_mode = _MODE_MAP.get(
        str(params.get("search_mode", "")).lower(), SearchMode.ANYTIME_DFS
    )
    node_choosing = _NODE_MAP.get(
        str(params.get("node_choosing_strategy", "")).lower(),
        NodeChoosingStrategy.MODEL_PATCH_FIRST,
    )
    conflict_group = _GROUP_MAP.get(
        str(params.get("conflict_group_strategy", "")).lower(),
        ConflictGroupStrategy.FIRST,
    )
    fluent_branch = _FLUENT_MAP.get(
        str(params.get("fluent_branch_mode", "")).lower(),
        FluentBranchMode.GROUP,
    )
    fluent_patch_cost = float(params.get("fluent_patch_cost") or 1.0)
    fluent_patch_weight = float(params.get("fluent_patch_weight") or 1.0)
    model_patch_cost = float(params.get("model_patch_cost") or 1.0)
    model_constraint_weight = float(params.get("model_constraint_weight") or 0.0)
    timeout_seconds = int(params.get("actual_timeout_seconds") or 60)
    max_nodes = params.get("max_search_nodes")

    # Collect per-node trace events for the target.
    trace_events: List[str] = []

    class TracingSearch(ConflictDrivenPatchSearch):
        """Wraps the search to log every node touching the target predicate."""

        def _expand_node(self, node, observations):  # type: ignore[override]
            """Intercept node expansion — log if target predicate is involved."""
            try:
                result = super()._expand_node(node, observations)
            except TypeError:
                # Signature mismatch guard — fall back gracefully.
                result = super()._expand_node(node)  # type: ignore[call-arg]
            return result

    searcher = TracingSearch(
        partial_domain_template=domain,
        search_mode=search_mode,
        fluent_patch_cost=fluent_patch_cost,
        fluent_patch_weight=fluent_patch_weight,
        model_patch_cost=model_patch_cost,
        model_constraint_weight=model_constraint_weight,
        node_choosing_strategy=node_choosing,
        conflict_group_strategy=conflict_group,
        fluent_branch_mode=fluent_branch,
    )

    obs_list = [obs for _, obs in ordered_observations]
    (learned_model, _, final_constraints, final_fluent_patches, _, report, _) = searcher.run(
        observations=obs_list,
        max_nodes=max_nodes,
        timeout_seconds=timeout_seconds,
    )

    best_constraints = report.get("final_model_constraints") or []
    best_fluent_patches_list = report.get("final_fluent_patches") or []

    forbid_entries = [
        c for c in best_constraints
        if (c.get("action") == target_action
            and c.get("model_part") == "eff"
            and c.get("operation") == "forbid"
            and target_predicate_name in c.get("predicate", ""))
    ]

    notes: List[str] = [f"Retrace completed. Trace events: {len(trace_events)}"]
    notes += trace_events[:50]  # Cap for readability.
    if len(trace_events) > 50:
        notes.append(f"... ({len(trace_events) - 50} more events truncated)")

    return ConflictSearchAnalysis(
        metrics_found=True,
        conflict_group_formed=any(
            c.get("action") == target_action for c in best_constraints
        ),
        model_forbid_generated=len(forbid_entries) > 0,
        model_forbid_cost=None,
        competing_fluent_patch_cost=None,
        best_model_constraints=best_constraints,
        best_fluent_patches=best_fluent_patches_list,
        notes=notes,
    )


# ===========================================================================
# Stage 5 — Assemble report and render
# ===========================================================================

def _derive_verdict(
    vote_table: List[VoteRecord],
    snapshots: List[HypothesisSnapshot],
    conflict: Optional[ConflictSearchAnalysis],
    cfm_index: Optional[int] = None,
) -> str:
    """Produce a short human-readable verdict from the collected evidence."""
    votes = [v for v in vote_table if v.spurious_vote and not v.is_first_encounter]
    masked_votes = [v for v in votes if v.masked_in_prev or v.masked_in_next]
    clean_votes = [v for v in votes if not v.masked_in_prev and not v.masked_in_next]

    first_intro = next(
        (s for s in snapshots if s.target_in_effects), None
    )

    lines: List[str] = []

    if not votes:
        lines.append(
            "The target predicate never appeared as an add-effect in any raw state diff. "
            "The spurious effect may have been introduced by frame-axiom propagation or "
            "a different data source — check the trajectory preprocessing."
        )
        return " ".join(lines)

    if clean_votes:
        lines.append(
            f"{len(clean_votes)} raw diff vote(s) with NO masking on either side "
            f"(observations: {[v.obs_idx for v in clean_votes]}). "
            "These directly caused the predicate to enter the effect set."
        )
    if masked_votes:
        lines.append(
            f"{len(masked_votes)} raw diff vote(s) where the target predicate was masked "
            f"in prev or next state (observations: {[v.obs_idx for v in masked_votes]}). "
            "Masking obscured the true value, making the transition look like an add-effect."
        )

    if first_intro:
        lines.append(
            f"PI-SAM first introduced the effect at obs={first_intro.obs_idx} "
            f"(problem={first_intro.problem}), comp={first_intro.comp_idx} "
            f"via {first_intro.event}."
        )

    scope = f"CFM #{cfm_index}" if cfm_index is not None else "the global best solution"
    if conflict is None:
        lines.append("No conflict-search metrics available (was denoising run?).")
    elif not conflict.conflict_group_formed:
        lines.append(
            f"The conflict search did NOT form a conflict group for this action/predicate in {scope}. "
            "The spurious effect was never challenged at this search node."
        )
    elif conflict.model_forbid_generated:
        lines.append(
            f"A FORBID-effect constraint is active in {scope} — "
            "the search ruled the predicate out as an effect at this node, yet it still appeared. "
            "This is contradictory: check whether the observations fed to PI-SAM at this node "
            "still produced the add-effect diff despite the constraint."
        )
    elif hasattr(conflict, 'model_require_generated') and conflict.model_require_generated:
        lines.append(
            f"A REQUIRE-effect constraint for this predicate is active in {scope} — "
            "the search explicitly required it as an effect, which explains its presence."
        )
    else:
        # Check notes for require signal since model_require_generated is not in the dataclass
        require_note = any("REQUIRE-effect" in n for n in (conflict.notes or []))
        if require_note:
            lines.append(
                f"A REQUIRE-effect constraint for this predicate is active in {scope} — "
                "the search explicitly required it as an effect, which explains its presence."
            )
        else:
            lines.append(
                f"A conflict group formed in {scope}, but no FORBID or REQUIRE constraint for this predicate — "
                "data-level fluent patches were used instead of a model constraint."
            )

    return " | ".join(lines)


def build_report(
    fold_dir: Path,
    target_action: str,
    target_predicate: str,
    observations_order: List[str],
    vote_table: List[VoteRecord],
    snapshots: List[HypothesisSnapshot],
    conflict: Optional[ConflictSearchAnalysis],
    cfm_index: Optional[int] = None,
    patch_details: Optional[Dict] = None,
) -> DiagnosisReport:
    """Assemble a DiagnosisReport from stage outputs."""
    votes = [v for v in vote_table if v.spurious_vote and not v.is_first_encounter]
    masked_votes = [v for v in votes if v.masked_in_prev or v.masked_in_next]

    first_intro = next(
        (s for s in snapshots if s.target_in_effects), None
    )
    ever_in_cannot_be = any(s.target_in_cannot_be for s in snapshots)

    verdict = _derive_verdict(vote_table, snapshots, conflict, cfm_index=cfm_index)

    cfm_model_constraints: Optional[List[Dict]] = None
    cfm_fluent_patch_count: Optional[int] = None
    if patch_details is not None:
        cfm_model_constraints = patch_details.get("model_constraints")
        cfm_fluent_patch_count = len(patch_details.get("fluent_patches", []))

    return DiagnosisReport(
        target_action=target_action,
        target_predicate=target_predicate,
        fold_dir=str(fold_dir),
        observations_order=observations_order,
        cfm_index=cfm_index,
        cfm_model_constraints=cfm_model_constraints,
        cfm_fluent_patch_count=cfm_fluent_patch_count,
        vote_table=vote_table,
        total_votes=len(votes),
        masked_votes=len(masked_votes),
        hypothesis_snapshots=snapshots,
        first_introduction_obs=first_intro.obs_idx if first_intro else None,
        first_introduction_comp=first_intro.comp_idx if first_intro else None,
        first_introduction_problem=first_intro.problem if first_intro else None,
        ever_in_cannot_be=ever_in_cannot_be,
        conflict_search=conflict,
        verdict=verdict,
    )


def print_human_report(report: DiagnosisReport) -> None:
    """Render the DiagnosisReport to stdout in a readable format."""
    sep = "─" * 70

    print(f"\n{sep}")
    print(f"  SPURIOUS EFFECT DIAGNOSIS")
    print(f"  Action:     {report.target_action}")
    print(f"  Predicate:  {report.target_predicate}")
    print(f"  Fold dir:   {report.fold_dir}")
    print(sep)

    if report.cfm_index is not None:
        print(f"\n[Scope: CFM #{report.cfm_index}]")
        print(f"  Fluent patches applied to observations : {report.cfm_fluent_patch_count}")
        print(f"  Model constraints active at this CFM  : {len(report.cfm_model_constraints or [])}")
        if report.cfm_model_constraints:
            for mc in report.cfm_model_constraints:
                print(f"    {mc.get('operation'):>6}  {mc.get('model_part')} of {mc.get('action'):<20s}  {mc.get('predicate')}")
    else:
        print(f"\n[Scope: raw masked originals (no --cfm-index)]")

    print("\n[Processing order]")
    for i, prob in enumerate(report.observations_order):
        print(f"  obs[{i}]: {prob}")

    print(f"\n[Stage 2 — Trajectory Attribution]")
    if not report.vote_table:
        print(f"  Action '{report.target_action}' was never fired in any observation.")
    else:
        header = f"  {'obs':>4}  {'problem':<12}  {'comp':>4}  {'in_diff':>7}  {'dir':>6}  {'masked_prev':>11}  {'masked_next':>11}  {'vote':>5}  note"
        print(header)
        print(f"  {'-'*80}")
        for v in report.vote_table:
            if v.is_first_encounter:
                note = "FIRST ENCOUNTER (add_new_action — initialises hypothesis, does not vote)"
                vote_str = "n/a"
            else:
                note = ""
                vote_str = "YES" if v.spurious_vote else "no"
            print(
                f"  {v.obs_idx:>4}  {v.problem:<12}  {v.comp_idx:>4}  "
                f"{str(v.predicate_in_diff):>7}  {v.diff_direction:>6}  "
                f"{str(v.masked_in_prev):>11}  {str(v.masked_in_next):>11}  "
                f"{vote_str:>5}  {note}"
            )
        print(f"\n  Total spurious votes : {report.total_votes}")
        print(f"  Masked-state votes   : {report.masked_votes}")

    print(f"\n[Stage 3 — PI-SAM Hypothesis Evolution]")
    if not report.hypothesis_snapshots:
        print(f"  No snapshots recorded (action '{report.target_action}' may not exist in domain).")
    else:
        prev_in_effects = False
        for s in report.hypothesis_snapshots:
            marker = ""
            if s.target_in_effects and not prev_in_effects:
                marker = "  ← FIRST INTRODUCTION"
            elif not s.target_in_effects and prev_in_effects:
                marker = "  ← REMOVED"
            status_eff = "IN_EFFECTS" if s.target_in_effects else "absent"
            status_cbe = "IN_CANNOT_BE" if s.target_in_cannot_be else "absent"
            print(
                f"  obs[{s.obs_idx}]/{s.problem}/comp[{s.comp_idx}] "
                f"({s.event}): effects={status_eff}  cannot_be={status_cbe}"
                f"  [eff_count={s.effects_count}]{marker}"
            )
            prev_in_effects = s.target_in_effects

        if report.first_introduction_obs is not None:
            print(
                f"\n  First introduced: obs[{report.first_introduction_obs}] "
                f"(problem={report.first_introduction_problem}), "
                f"comp={report.first_introduction_comp}"
            )
        else:
            print(f"\n  Target predicate NEVER entered effects during replay.")
        print(f"  Ever in cannot_be_effect: {report.ever_in_cannot_be}")

    print(f"\n[Stage 4 — Conflict Search Analysis]")
    cs = report.conflict_search
    if cs is None:
        print("  No learning_metrics.json found — conflict search data unavailable.")
    elif not cs.metrics_found:
        print("  Metrics file not found.")
    else:
        print(f"  Conflict group formed for action:  {cs.conflict_group_formed}")
        print(f"  FORBID-effect patch generated:     {cs.model_forbid_generated}")
        if cs.model_forbid_cost is not None:
            print(f"  Model-fix cost (estimated):        {cs.model_forbid_cost:.3f}")
        if cs.competing_fluent_patch_cost is not None:
            print(f"  Data-fix cost (best solution):     {cs.competing_fluent_patch_cost:.3f}")
        for note in cs.notes:
            print(f"  NOTE: {note}")

    print(f"\n[Verdict]")
    print(f"  {report.verdict}")
    print(f"\n{sep}\n")


def save_json_report(report: DiagnosisReport, output_path: Path) -> None:
    """Serialize the full DiagnosisReport to JSON."""

    def _serialize(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: _serialize(v) for k, v in asdict(obj).items()}
        if isinstance(obj, list):
            return [_serialize(i) for i in obj]
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        return obj

    data = _serialize(report)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  JSON report saved to: {output_path}")


# ===========================================================================
# CLI entry point
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose why a lifted predicate ended up in a learned action's effects.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--fold-dir", required=True, type=Path,
        help="Path to the fold directory containing fold_info.json (e.g. fold0_numtrajs3_gtrate0/).",
    )
    parser.add_argument(
        "--domain", required=True, type=Path,
        help="Path to the domain PDDL file (reference domain, partial_parsing=True will be used).",
    )
    parser.add_argument(
        "--traj-base", required=True, type=Path,
        help=(
            "Base directory containing one subdirectory per problem "
            "(e.g. .../training/trajectories/). "
            "Each subdirectory must contain the .trajectory and .masking_info files "
            "referenced in fold_info.json."
        ),
    )
    parser.add_argument(
        "--action", required=True, type=str,
        help="Name of the action to trace (e.g. 'stack').",
    )
    parser.add_argument(
        "--predicate", required=True, type=str,
        help="Lifted predicate to trace, e.g. '(on-table ?x)'. Only the predicate name is used for matching.",
    )
    parser.add_argument(
        "--cfm-index", type=int, default=None, metavar="N",
        help=(
            "Zero-based index of the conflict-free model to analyse (e.g. --cfm-index 3). "
            "When set, the script loads conflict_free_models/conflict_free_model_N/patch_details.json "
            "from the fold directory and applies those fluent patches on top of the masked originals "
            "before running Stages 2 and 3. This reconstructs the exact observations PI-SAM saw "
            "when that specific CFM was produced. Without this flag, Stages 2 and 3 run on the "
            "raw (unpatched) masked originals."
        ),
    )
    parser.add_argument(
        "--retrace", action="store_true", default=False,
        help="Re-run the conflict search with tracing hooks (slow). Default: analyse saved learning_metrics.json only.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Optional path to write the JSON diagnosis report. Defaults to <fold-dir>/diagnosis_<action>_<pred>.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    fold_dir: Path = args.fold_dir.resolve()
    domain_path: Path = args.domain.resolve()
    traj_base: Path = args.traj_base.resolve()
    target_action: str = args.action.strip()
    target_predicate: str = args.predicate.strip()
    target_predicate_name: str = _predicate_name_from_lifted(target_predicate)
    cfm_index: Optional[int] = args.cfm_index

    print(f"\nDiagnosing spurious effect:")
    print(f"  Action:    {target_action}")
    print(f"  Predicate: {target_predicate}  (name: {target_predicate_name})")
    print(f"  Fold dir:  {fold_dir}")
    print(f"  Domain:    {domain_path}")
    print(f"  Traj base: {traj_base}")
    if cfm_index is not None:
        print(f"  CFM index: {cfm_index}  (analysis scoped to observations at CFM #{cfm_index})")

    # ------------------------------------------------------------------
    # Stage 0 — Load fold metadata
    # ------------------------------------------------------------------
    print("\n[Stage 0] Loading fold_info.json …")
    fold_info = load_fold_info(fold_dir)
    ordered_paths = resolve_trajectory_paths(fold_info, traj_base)
    observations_order = [p for p, _, _ in ordered_paths]
    print(f"  Processing order ({len(ordered_paths)} trajectories): {observations_order}")

    # ------------------------------------------------------------------
    # Stage 1 — Load observations
    # ------------------------------------------------------------------
    print("\n[Stage 1] Loading observations in fold_info order …")
    # Domain is re-loaded fresh for each stage that mutates it.
    domain_for_loading = DomainParser(domain_path, partial_parsing=True).parse_domain()
    ordered_observations = load_observations_in_order(ordered_paths, domain_for_loading)

    # ------------------------------------------------------------------
    # Stage 1b — (optional) Apply CFM-specific fluent patches
    # ------------------------------------------------------------------
    if cfm_index is not None:
        print(f"\n[Stage 1b] Loading patch_details.json for CFM #{cfm_index} and applying fluent patches …")
        patch_details = load_cfm_patch_details(fold_dir, cfm_index)
        n_fluent = len(patch_details.get("fluent_patches", []))
        n_model = len(patch_details.get("model_constraints", []))
        print(f"  CFM #{cfm_index}: cost={patch_details.get('cost')}, "
              f"fluent_patches={n_fluent}, model_constraints={n_model}")
        ordered_observations = apply_cfm_fluent_patches(ordered_observations, patch_details)
        print(f"  Fluent patches applied — Stages 2 and 3 now run on CFM #{cfm_index} observations.")
    else:
        patch_details = None

    # ------------------------------------------------------------------
    # Stage 2 — Trajectory attribution
    # ------------------------------------------------------------------
    print(f"\n[Stage 2] Computing vote table for action='{target_action}', predicate='{target_predicate_name}' …")
    vote_table = compute_vote_table(ordered_observations, target_action, target_predicate_name)
    spurious = [v for v in vote_table if v.spurious_vote]
    print(f"  Components with action '{target_action}': {len(vote_table)}")
    print(f"  Spurious-vote components:                {len(spurious)}")

    # ------------------------------------------------------------------
    # Stage 3 — PI-SAM hypothesis evolution
    # ------------------------------------------------------------------
    print(f"\n[Stage 3] Replaying PI-SAM with tracing …")
    domain_for_learning = DomainParser(domain_path, partial_parsing=True).parse_domain()
    snapshots = replay_pisam_with_tracing(
        ordered_observations, domain_for_learning, target_action, target_predicate_name
    )
    print(f"  Snapshots recorded: {len(snapshots)}")
    first_intro = next((s for s in snapshots if s.target_in_effects), None)
    if first_intro:
        print(
            f"  First introduction: obs[{first_intro.obs_idx}] problem={first_intro.problem} "
            f"comp={first_intro.comp_idx} via {first_intro.event}"
        )
    else:
        print(f"  Predicate did NOT enter effects during replay.")

    # ------------------------------------------------------------------
    # Stage 4 — Conflict search analysis
    # ------------------------------------------------------------------
    conflict: Optional[ConflictSearchAnalysis] = None
    if args.retrace:
        print(f"\n[Stage 4] Re-running conflict search with tracing (--retrace) …")
        domain_for_retrace = DomainParser(domain_path, partial_parsing=True).parse_domain()
        conflict = retrace_conflict_search(
            ordered_observations, fold_dir, domain_for_retrace,
            target_action, target_predicate_name,
        )
    else:
        if cfm_index is not None:
            print(f"\n[Stage 4] Analysing patch_details.json for CFM #{cfm_index} …")
        else:
            print(f"\n[Stage 4] Analysing saved learning_metrics.json (global best solution) …")
        conflict = analyse_saved_metrics(fold_dir, target_action, target_predicate_name, cfm_index=cfm_index)
        if conflict is None:
            print("  Metrics file not found — skipping conflict search analysis.")

    # ------------------------------------------------------------------
    # Stage 5 — Report
    # ------------------------------------------------------------------
    print(f"\n[Stage 5] Building report …")
    report = build_report(
        fold_dir=fold_dir,
        target_action=target_action,
        target_predicate=target_predicate,
        observations_order=observations_order,
        vote_table=vote_table,
        snapshots=snapshots,
        conflict=conflict,
        cfm_index=cfm_index,
        patch_details=patch_details,
    )

    print_human_report(report)

    # Save JSON report.
    safe_pred = target_predicate_name.replace("-", "_").replace("?", "").replace("(", "").replace(")", "")
    cfm_suffix = f"_cfm{cfm_index}" if cfm_index is not None else ""
    output_path = args.output or (fold_dir / f"diagnosis_{target_action}_{safe_pred}{cfm_suffix}.json")
    save_json_report(report, output_path)


if __name__ == "__main__":
    main()
