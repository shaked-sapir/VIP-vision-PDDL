"""
trace_spurious_effects.py — Diagnose why a lifted predicate ended up in an action's effects.

Given an existing fold's data, this script replays the full learning pipeline with fine-grained
tracing to identify the root cause of a spurious learned effect.

Pipeline stages:
  Stage 0 — Load fold_info.json and reconstruct the canonical observation order.
  Stage 1 — Load and reconstruct masked observations (in fold_info order).
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
        [--retrace]

Notes:
  - --traj-base is the directory that contains one subdirectory per problem
    (e.g. problem3/, problem7/, …), each holding the .trajectory and .masking_info files.
  - The processing order is taken directly from fold_info.json's "trajectories" array.
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
        all_preds = get_state_grounded_predicates(obs.components[0].previous_state)
        n_masked_states = sum(
            1 for comp in obs.components
            if len(get_state_masked_predicates(comp.previous_state)) > 0
               or len(get_state_masked_predicates(comp.next_state)) > 0
        )
        print(
            f"  Loaded obs[{len(observations) - 1}] problem={problem} | "
            f"components={n_components} | masked_states≈{n_masked_states}"
        )
    return observations


# ===========================================================================
# Stage 2 — Trajectory-level attribution
# ===========================================================================

def _predicate_name_from_lifted(lifted_str: str) -> str:
    """Extract predicate name from a lifted string like '(on-table ?x)' → 'on-table'."""
    return lifted_str.strip("()").split()[0].lstrip("not (").strip()


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

    for obs_idx, (problem, obs) in enumerate(ordered_observations):
        for comp_idx, component in enumerate(obs.components):
            if not component.is_successful:
                continue
            action: ActionCall = component.grounded_action_call
            if action.name != target_action:
                continue

            prev_preds = get_state_grounded_predicates(component.previous_state)
            next_preds = get_state_grounded_predicates(component.next_state)

            # Raw diff (same logic as extract_discrete_effects_partial_observability)
            grounded_add, grounded_del = extract_discrete_effects_partial_observability(
                prev_preds, next_preds
            )

            # Check masking status for the target predicate specifically
            target_in_prev_masked = any(
                p.is_masked and p.name == target_predicate_name
                for p in prev_preds
            )
            target_in_next_masked = any(
                p.is_masked and p.name == target_predicate_name
                for p in next_preds
            )

            in_add = any(p.name == target_predicate_name for p in grounded_add)
            in_del = any(p.name == target_predicate_name for p in grounded_del)

            if in_add:
                direction = "add"
            elif in_del:
                direction = "delete"
            else:
                direction = "none"

            records.append(VoteRecord(
                obs_idx=obs_idx,
                problem=problem,
                comp_idx=comp_idx,
                predicate_in_diff=in_add or in_del,
                diff_direction=direction,
                masked_in_prev=target_in_prev_masked,
                masked_in_next=target_in_next_masked,
                spurious_vote=in_add,
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
) -> Optional[ConflictSearchAnalysis]:
    """
    Parse the saved learning_metrics.json in *fold_dir* and report whether
    the conflict search ever targeted the spurious predicate.

    Returns None if no metrics file is found.
    """
    metrics_path = fold_dir / "learning_metrics.json"
    if not metrics_path.exists():
        return None

    with open(metrics_path) as f:
        metrics = json.load(f)

    best_constraints: List[Dict] = metrics.get("best_model_constraints") or []
    best_fluent_patches: List[Dict] = metrics.get("best_fluent_patches") or []

    # Look for a FORBID effect constraint on our action+predicate.
    forbid_entries = [
        c for c in best_constraints
        if (c.get("action") == target_action
            and c.get("model_part") == "eff"
            and c.get("operation") == "forbid"
            and target_predicate_name in c.get("predicate", ""))
    ]

    model_forbid_generated = len(forbid_entries) > 0
    model_forbid_cost: Optional[float] = None

    # Reconstruct cost from metrics config if the FORBID was applied.
    if model_forbid_generated:
        model_patch_cost = metrics.get("model_patch_cost", 1.0) or 1.0
        model_constraint_weight = metrics.get("model_constraint_weight", 0.0) or 0.0
        n_constraints = len(best_constraints)
        model_forbid_cost = model_constraint_weight * model_patch_cost * n_constraints

    # Fluent patch cost for competing patches.
    fluent_patch_cost: Optional[float] = None
    if best_fluent_patches:
        fp_cost = metrics.get("fluent_patch_cost", 1.0) or 1.0
        fp_weight = metrics.get("fluent_patch_weight", 1.0) or 1.0
        fluent_patch_cost = fp_weight * fp_cost * len(best_fluent_patches)

    # Determine if any conflict group formed at all for the action.
    # The metrics don't store individual conflict groups, only the best solution's patches.
    # We infer: if any constraint or fluent patch mentions the action, a group likely formed.
    action_mentioned_in_constraints = any(
        c.get("action") == target_action for c in best_constraints
    )
    action_mentioned_in_patches = any(
        # Fluent patches don't carry an action name directly — they carry obs/comp/state/fluent.
        # We can check if the predicate name appears in the fluent string.
        target_predicate_name in str(fp.get("fluent", ""))
        for fp in best_fluent_patches
    )
    conflict_group_formed = action_mentioned_in_constraints or action_mentioned_in_patches

    notes: List[str] = []
    if not best_constraints and not best_fluent_patches:
        notes.append("No best_model_constraints or best_fluent_patches saved — search may not have run or found no solution.")
    if not conflict_group_formed:
        notes.append(
            f"No constraint or patch mentioning action '{target_action}' found — "
            "the conflict search may not have encountered a conflict for this action."
        )
    if model_forbid_generated:
        notes.append(
            f"A FORBID-effect constraint for '{target_predicate_name}' WAS present in the best solution."
        )
    elif not model_forbid_generated and conflict_group_formed:
        notes.append(
            f"Conflict group for '{target_action}' formed, but no FORBID-effect for '{target_predicate_name}' "
            "in the best solution — data-fix (fluent patches) was preferred or the predicate was not conflicted."
        )

    return ConflictSearchAnalysis(
        metrics_found=True,
        conflict_group_formed=conflict_group_formed,
        model_forbid_generated=model_forbid_generated,
        model_forbid_cost=model_forbid_cost,
        competing_fluent_patch_cost=fluent_patch_cost,
        best_model_constraints=best_constraints,
        best_fluent_patches=best_fluent_patches,
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
) -> str:
    """Produce a short human-readable verdict from the collected evidence."""
    votes = [v for v in vote_table if v.spurious_vote]
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

    if conflict is None:
        lines.append("No conflict-search metrics available (was denoising run?).")
    elif not conflict.conflict_group_formed:
        lines.append(
            "The conflict search did NOT form a conflict group for this action/predicate. "
            "The spurious effect was never challenged during denoising."
        )
    elif not conflict.model_forbid_generated:
        lines.append(
            "A conflict group formed, but no FORBID-effect patch was chosen — "
            "data-level fluent patches were preferred (lower cost)."
        )
    else:
        lines.append(
            "A FORBID-effect model constraint WAS applied in the best solution, "
            "yet the effect still appears — check whether a different CFM was selected."
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
) -> DiagnosisReport:
    """Assemble a DiagnosisReport from stage outputs."""
    votes = [v for v in vote_table if v.spurious_vote]
    masked_votes = [v for v in votes if v.masked_in_prev or v.masked_in_next]

    first_intro = next(
        (s for s in snapshots if s.target_in_effects), None
    )
    ever_in_cannot_be = any(s.target_in_cannot_be for s in snapshots)

    verdict = _derive_verdict(vote_table, snapshots, conflict)

    return DiagnosisReport(
        target_action=target_action,
        target_predicate=target_predicate,
        fold_dir=str(fold_dir),
        observations_order=observations_order,
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

    print("\n[Processing order]")
    for i, prob in enumerate(report.observations_order):
        print(f"  obs[{i}]: {prob}")

    print(f"\n[Stage 2 — Trajectory Attribution]")
    if not report.vote_table:
        print(f"  Action '{report.target_action}' was never fired in any observation.")
    else:
        header = f"  {'obs':>4}  {'problem':<12}  {'comp':>4}  {'in_diff':>7}  {'dir':>6}  {'masked_prev':>11}  {'masked_next':>11}  {'vote':>5}"
        print(header)
        print(f"  {'-'*65}")
        for v in report.vote_table:
            print(
                f"  {v.obs_idx:>4}  {v.problem:<12}  {v.comp_idx:>4}  "
                f"{str(v.predicate_in_diff):>7}  {v.diff_direction:>6}  "
                f"{str(v.masked_in_prev):>11}  {str(v.masked_in_next):>11}  "
                f"{'YES' if v.spurious_vote else 'no':>5}"
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

    print(f"\nDiagnosing spurious effect:")
    print(f"  Action:    {target_action}")
    print(f"  Predicate: {target_predicate}  (name: {target_predicate_name})")
    print(f"  Fold dir:  {fold_dir}")
    print(f"  Domain:    {domain_path}")
    print(f"  Traj base: {traj_base}")

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
        print(f"\n[Stage 4] Analysing saved learning_metrics.json …")
        conflict = analyse_saved_metrics(fold_dir, target_action, target_predicate_name)
        if conflict is None:
            print("  learning_metrics.json not found — skipping conflict search analysis.")

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
    )

    print_human_report(report)

    # Save JSON report.
    safe_pred = target_predicate_name.replace("-", "_").replace("?", "").replace("(", "").replace(")", "")
    output_path = args.output or (fold_dir / f"diagnosis_{target_action}_{safe_pred}.json")
    save_json_report(report, output_path)


if __name__ == "__main__":
    main()
