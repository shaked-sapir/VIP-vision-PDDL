"""``cdps_milp_single_round``: one MILP solve replaces the whole CDPS search.

The pipeline (``docs/cdps-milp-denoiser-design.md`` §1) is four steps:

    encode all traces  ->  solve once  ->  extract T'  ->  PI-SAM on T'

CDPS explores repairs one PI-SAM run at a time; here a single CP-SAT solve
returns the *globally cheapest* repair, and PI-SAM runs exactly once on the
result. PI-SAM remains the learner, so the safety lineage of the returned model
is unchanged — the MILP only decides which observations to trust.

Two guarantees make this a drop-in sibling of ``cdps`` rather than a new method:

- **Feasible ⟹ conflict-free** (design §4.1): if the solve succeeds, PI-SAM on
  T' raises no conflicts, so the run always ends with a conflict-free model.
  A non-empty ``conflicts`` list therefore signals an encoding bug, and is
  reported as ``pisam_conflicts_on_feasible`` rather than swallowed.
- **cost(MILP) <= cost(best CDPS CFM)** (design §4.1 converse): the reported
  ``repair_cost`` is a lower bound on what CDPS can achieve on the same fold —
  the validation check of design §7.1. It holds only with ``eq16`` off, which
  ``as_report`` records.

Artifacts mirror CDPS's exactly (``conflict_free_models/...``,
``conflict_free_solutions_log.json``) so the dashboard stack needs no changes;
two MILP-only diagnostics are added (``milp_masked_completion.json``,
``milp_repair_log.json``).
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set

from pddl_plus_parser.models import Domain, Observation
from sam_learning.core import LearnerDomain
from utilities import NegativePreconditionPolicy

from constraint_opt.factory import resolve as resolve_encoder
from planning_structs.traces import Traces

from src.pi_sam.noisy_pisam.noisy_pisam_learning import NoisyPisamLearner
from src.milp import encoder as _encoder_module  # noqa: F401  (factory registration)
from src.plan_denoising.milp_denoiser.config import CdpsMilpConfig
from src.plan_denoising.patch_accounting import net_patch_count_from_records
from src.milp.converter import (
    build_ps_domain,
    build_ps_instance_from_objects,
    observation_to_trace,
)
from src.plan_denoising.milp_denoiser.trajectory_extraction import (
    ExtractionResult,
    extract_repaired_observations,
    save_extraction_artifacts,
)

# The reference-model channel needs a reference model; a single round has none,
# so only the state channel is active (see CdpsMilpConfig.encoding_config).
_OBJECTIVES = {"state"}

SaveObservationsFn = Callable[[List[Observation], Path], None]


@dataclass
class SingleRoundResult:
    """Outcome of one ``cdps_milp_single_round`` run.

    Attributes:
        learned_domain: PI-SAM's model, or ``None`` when the MILP found no
            solution (then nothing downstream should be evaluated).
        conflicts: PI-SAM's conflicts on T'. MUST be empty when ``solved`` —
            see the module docstring.
        observations: T', the repaired (and still masked) observations.
        solved: Whether the solver returned a usable solution.
        repair_cost: Number of observed fluents the MILP flipped, across *all*
            observations in the fold — the scope the design §7.1 lower-bound
            check needs. (``cdps_milp_loop`` cannot report this, because no
            single solve there covers the whole fold; see ``loop.as_report``.)
        stats: Everything the report needs (solver status, timings, config).
    """

    learned_domain: Optional[LearnerDomain]
    conflicts: List[Any]
    observations: List[Observation]
    solved: bool
    repair_cost: int
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_conflict_free(self) -> bool:
        return self.solved and not self.conflicts

    def as_report(self) -> Dict[str, Any]:
        """Flat report dict, in the spirit of CDPS's ``SearchResult.report``."""
        report = dict(self.stats)
        report.update({
            "algorithm": "cdps_milp_single_round",
            "milp_solved": self.solved,
            "repair_cost": self.repair_cost,
            "best_cost": self.repair_cost if self.is_conflict_free else None,
            "conflict_free_model_count": 1 if self.is_conflict_free else 0,
            "pisam_conflicts_on_feasible": len(self.conflicts) if self.solved else None,
        })
        return report


# ---------------------------------------------------------------- encoding


def _build_traces(
    partial_domain: Domain,
    observations: Sequence[Observation],
    config: CdpsMilpConfig,
    gt_states_by_obs: Optional[Dict[int, Set[int]]],
    object_types_by_obs: Optional[Mapping[int, Mapping[str, str]]] = None,
):
    """``(ps_domain, obs_t)`` — every observation, encoded, or an exception.

    Each observation is grounded on its *own* objects (folds mix problems), read
    off ``Observation.grounded_objects`` so no problem file has to be re-parsed.

    An observation that cannot be encoded raises rather than being dropped. The
    arm's headline claim is "same traces as CDPS, different denoiser", and a
    quiet drop falsifies it: the row still says ``num_trajectories=8`` while the
    learner saw 7, and 7 traces are *easier*, so the arm's advantage grows for a
    reason the table cannot show. ``run_fold.run_cdps_phase`` catches this and
    records an ``error`` on this arm's row alone, so the failure costs one cell
    instead of silently biasing every comparison drawn from it.

    Args:
        object_types_by_obs: obs_idx -> (name -> declared type name), overriding
            the types the trajectory parser *inferred*. See
            :func:`converter.problem_object_types`.
    """
    ps_domain = build_ps_domain(partial_domain)
    obs_t = []
    for obs_idx, observation in enumerate(observations):
        # include_repeated_args: PI-SAM consumes T', and our observations are
        # completed over *all* object tuples, so every fluent PI-SAM will see
        # must carry a MILP variable — otherwise design §4.1 has a hole. See
        # converter.RepeatedArgsInstance.
        instance = build_ps_instance_from_objects(
            ps_domain,
            partial_domain,
            observation.grounded_objects,
            include_repeated_args=True,
            object_types=(object_types_by_obs or {}).get(obs_idx),
        )
        try:
            trace = observation_to_trace(
                instance,
                observation,
                goal_fluents=None,  # the final state is soft unless it is known GT
                gt_state_indices=(gt_states_by_obs or {}).get(obs_idx),
                gt_anchoring=config.gt_anchoring,
            )
        except ValueError as error:
            raise ValueError(
                f"Observation {obs_idx} cannot be encoded for the MILP: {error}"
            ) from error
        if trace is None:
            raise ValueError(f"Observation {obs_idx} has no components to encode")
        obs_t.append(trace)
    return ps_domain, obs_t


# ---------------------------------------------------------------- learning


def _learn_with_pisam(
    partial_domain: Domain,
    observations: List[Observation],
    gt_states_by_obs: Optional[Dict[int, Set[int]]],
    negative_preconditions_policy: NegativePreconditionPolicy,
    seed: int,
):
    """PI-SAM on T', with no patches — the MILP already did all the repairing."""
    learner = NoisyPisamLearner(
        partial_domain=deepcopy(partial_domain),
        negative_preconditions_policy=negative_preconditions_policy,
        seed=seed,
    )
    return learner.learn_action_model_with_conflicts(
        observations=observations,
        fluent_patches=set(),
        model_patches=set(),
        gt_states_by_obs=gt_states_by_obs,
    )


# ---------------------------------------------------------------- artifacts


def save_artifacts(
    result: SingleRoundResult,
    extraction: ExtractionResult,
    fold_work_dir: Path,
    save_observations_fn: Optional[SaveObservationsFn] = None,
) -> None:
    """Write the CDPS-shaped artifact suite plus the two MILP diagnostics.

    The model lands in ``conflict_free_models/conflict_free_model_0/`` when the
    run ended conflict-free and in ``conflict_free_models/final_model/``
    otherwise — the same convention CDPS uses, so the report/dashboard stack
    reads both algorithms with one code path.
    """
    fold_work_dir = Path(fold_work_dir)
    models_dir = fold_work_dir / "conflict_free_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    conflict_free = result.is_conflict_free
    target = models_dir / ("conflict_free_model_0" if conflict_free else "final_model")
    target.mkdir(parents=True, exist_ok=True)

    if result.learned_domain is not None:
        (target / "model.pddl").write_text(result.learned_domain.to_pddl())
    if save_observations_fn is not None and result.observations:
        save_observations_fn(result.observations, target / "final_observations")

    solve_seconds = result.stats.get("total_time_seconds")
    flip_records = [flip.as_patch_dict() for flip in extraction.flips]
    # The solver assigns each (state, fluent) once, so it cannot emit the
    # self-cancelling pair that inflates CDPS's count. Computed rather than
    # assumed, so that the two arms' `net_*` columns are produced by one rule
    # and the design 7.1 comparison is like-for-like. See patch_accounting.
    net_flips = net_patch_count_from_records(flip_records)
    patch_details = {
        "model_constraint_count": 0,  # the MILP never constrains the model
        "fluent_patch_count": result.repair_cost,
        "net_fluent_patch_count": net_flips,
        "cost": result.repair_cost,
        "net_cost": net_flips,
        "wall_time_seconds": solve_seconds,
        "nodes_expanded": 1,  # one solve = one "node"
        "model_constraints": [],
        "fluent_patches": flip_records,
    }
    (target / "patch_details.json").write_text(json.dumps(patch_details, indent=2))

    solutions_log = [{
        "index": 0,
        "cost": result.repair_cost,
        "net_cost": net_flips,
        "depth": 0,
        "nodes_expanded_so_far": 1,
        "wall_time_so_far": solve_seconds,
        "fluent_patch_count": result.repair_cost,
        "net_fluent_patch_count": net_flips,
        "model_constraint_count": 0,
        "model_constraints": [],
        "fluent_patches": flip_records,
    }] if conflict_free else []
    (fold_work_dir / "conflict_free_solutions_log.json").write_text(
        json.dumps(solutions_log, indent=2)
    )

    save_extraction_artifacts(extraction, fold_work_dir)


# ---------------------------------------------------------------- driver


def run_single_round(
    partial_domain: Domain,
    observations: Sequence[Observation],
    config: CdpsMilpConfig,
    time_limit_seconds: Optional[int] = None,
    gt_states_by_obs: Optional[Dict[int, Set[int]]] = None,
    negative_preconditions_policy: NegativePreconditionPolicy = NegativePreconditionPolicy.hard,
    seed: int = 42,
    fold_work_dir: Optional[Path] = None,
    save_observations_fn: Optional[SaveObservationsFn] = None,
    object_types_by_obs: Optional[Mapping[int, Mapping[str, str]]] = None,
) -> SingleRoundResult:
    """Encode every trace, solve once, extract T', and learn from it.

    Args:
        partial_domain: The domain template (predicates + action signatures).
        observations: Masked (and possibly noisy) observations — the same input
            ``learn_cdps`` receives.
        config: The validated ``cdps_milp`` block.
        time_limit_seconds: Solver budget; pass the fold's CDPS budget to make
            the head-to-head comparison fair.
        gt_states_by_obs: CDPS's GT map (obs_idx -> 0-based state indices).
            Which of those the MILP hard-fixes is ``config.gt_anchoring``'s call;
            PI-SAM always receives the full map.
        fold_work_dir: Where to write artifacts; ``None`` skips artifact writing.
        save_observations_fn: Injected serializer for T' (the benchmark's
            ``save_observations_to_dir``); ``None`` skips ``final_observations/``.
        object_types_by_obs: obs_idx -> (object name -> declared type name),
            from the problems the observations came from. ``.trajectory`` files
            declare no ``:objects``, so without this the types are *inferred*
            and objects in untyped predicate slots come back as ``object`` —
            which makes their actions ungroundable. See
            :func:`converter.problem_object_types`.

    Returns:
        A :class:`SingleRoundResult`. ``solved=False`` means the solver returned
        nothing usable (infeasible or out of time) and no model was learned.

    Raises:
        ValueError: if any observation cannot be encoded. Dropping it instead
            would quietly shrink the trace set this arm is compared on.
    """
    start = time.perf_counter()
    stats: Dict[str, Any] = {"config": config.as_stats()}

    ps_domain, obs_t = _build_traces(
        partial_domain, observations, config, gt_states_by_obs, object_types_by_obs
    )
    stats["n_traces"] = len(obs_t)
    # Kept for schema stability with rows written before _build_traces raised.
    # It is now always 0: an unencodable observation aborts the arm instead.
    stats["n_traces_dropped"] = 0
    if not obs_t:
        raise ValueError("No usable traces for the MILP")

    encoder = resolve_encoder(config.solver_encoder_key())(
        ps_domain,
        Traces(instance=None, obs_m=None, obs_t=obs_t),
        _OBJECTIVES,
        config=config.encoding_config(has_prior=False),
    )
    solved = encoder.solve(time_limit=time_limit_seconds)
    stats["milp"] = dict(encoder.solve_stats)
    stats["time_limit_seconds"] = time_limit_seconds

    if not solved:
        stats["total_time_seconds"] = round(time.perf_counter() - start, 3)
        return SingleRoundResult(
            learned_domain=None, conflicts=[], observations=[],
            solved=False, repair_cost=0, stats=stats,
        )

    extraction = extract_repaired_observations(encoder, list(observations))
    stats["extraction"] = extraction.as_stats()

    learned_domain, conflicts, learning_report = _learn_with_pisam(
        partial_domain, extraction.observations, gt_states_by_obs,
        negative_preconditions_policy, seed,
    )
    stats["learning_report"] = learning_report
    stats["total_time_seconds"] = round(time.perf_counter() - start, 3)

    result = SingleRoundResult(
        learned_domain=learned_domain,
        conflicts=list(conflicts),
        observations=extraction.observations,
        solved=True,
        repair_cost=extraction.repair_cost,
        stats=stats,
    )
    if conflicts:
        # Design §4.1 says this cannot happen. It is a bug signal, not a
        # search outcome, so make it loud instead of silently degrading.
        print(f"  [MILP] WARNING: PI-SAM raised {len(conflicts)} conflict(s) on a "
              f"feasible T' — this contradicts design §4.1, please investigate.")

    if fold_work_dir is not None:
        save_artifacts(result, extraction, Path(fold_work_dir), save_observations_fn)

    return result
