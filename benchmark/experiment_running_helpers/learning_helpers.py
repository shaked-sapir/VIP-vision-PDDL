"""
Learning wrapper functions for AMLGym experiments.

Three denoisers share one learner. All run PI-SAM (partial observability) on
masked observations; they differ only in how they decide which observations to
distrust:

- ``learn_cdps``                     — Conflict-Directed Patch Search (search).
- ``learn_cdps_milp_single_round``   — one CP-SAT solve (see
  ``src/plan_denoising/milp_denoiser/``).
- ``learn_cdps_milp_loop``           — repeated CP-SAT solves over sampled
  subsets, keeping the best-scoring model.

All three return the same ``(model_pddl, report, patched_observations)`` triple
and write the same ``conflict_free_models/`` artifact layout, so ``run_fold``
and the dashboard stack treat them interchangeably.

The old fully-observable (SAM) and plain (non-denoising) paths were removed.
"""

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser

from benchmark.experiment_running_helpers.cleaned_trajectories import (
    save_fold_observations,
    save_observations_to_dir,
)
from src.plan_denoising.conflict_search import ConflictDrivenPatchSearch
from src.plan_denoising.conflict_search_config import CDPSConfig
from src.plan_denoising.milp_denoiser.config import CdpsMilpConfig
from src.milp.converter import problem_object_types
from src.plan_denoising.milp_denoiser.loop import run_loop
from src.plan_denoising.milp_denoiser.single_round import run_single_round
from src.utils.masking import load_masked_observation


def _load_masked_observations(partial_domain, traj_paths, pre_built_observations):
    """The fold's masked observations, from memory or from disk.

    ``pre_built_observations`` (simulated runs) is authoritative when given;
    otherwise each trajectory is paired with its sibling ``.masking_info`` and
    loaded (image pipeline). Trajectories with no masking info are skipped.
    """
    if pre_built_observations is not None:
        return pre_built_observations

    observations = []
    for traj_path_str in traj_paths:
        traj_path = Path(traj_path_str)
        masking_info_path = traj_path.parent / f"{traj_path.stem}.masking_info"
        if not masking_info_path.exists():
            continue
        observations.append(
            load_masked_observation(traj_path, masking_info_path, partial_domain)
        )
    return observations


def _save_original_observations(
    observations, fold_work_dir, prepared_trajectories, pre_built_observations, tag,
):
    """Freeze the pre-denoising observations for backfills and comparisons.

    Only the image pipeline persists here; for simulated runs the
    ``SimulatedDataSource`` already wrote ``original_observations/``
    (``pre_built_observations`` is not None), so re-saving would double-write
    identical files.
    """
    if pre_built_observations is not None or fold_work_dir is None:
        return
    if not prepared_trajectories or not observations:
        return

    original_obs_dir = fold_work_dir / "original_observations"
    save_fold_observations(
        observations,
        prepared_trajectories,
        original_obs_dir,
        observation_prefix="original_observation",
    )
    print(f"  [{tag}] Saved {len(observations)} original observations "
          f"to {original_obs_dir.name}/")


def _make_save_observations_fn(prepared_trajectories):
    """Callback writing a repaired observation list into a directory, or None."""
    if not prepared_trajectories:
        return None
    return lambda obs, out_dir: save_observations_to_dir(obs, prepared_trajectories, out_dir)


def _learn_cdps_core(
    domain_ref_path, traj_paths, config: CDPSConfig, timeout_seconds,
    fold_work_dir=None, prepared_trajectories=None, gt_states_by_obs=None,
    pre_built_observations=None, events_tracing: bool = False,
):
    """Run the Conflict-Directed Patch Search (CDPS).

    Args:
        pre_built_observations: Optional list of pre-built Observation objects.
            When provided, file loading from traj_paths is skipped entirely
            (simulated data); otherwise observations are loaded from disk (image
            pipeline).
        events_tracing: If True, collect node expansion events and write
            search_trace.json to fold_work_dir after the search completes.
    """
    partial_domain = DomainParser(Path(str(domain_ref_path)), partial_parsing=True).parse_domain()

    masked_observations = _load_masked_observations(
        partial_domain, traj_paths, pre_built_observations
    )
    _save_original_observations(
        masked_observations, fold_work_dir, prepared_trajectories,
        pre_built_observations, tag="CDPS",
    )

    conflict_free_models_dir = (fold_work_dir / "conflict_free_models") if fold_work_dir else None
    save_t_prime_fn = None
    if conflict_free_models_dir is not None:
        save_t_prime_fn = _make_save_observations_fn(prepared_trajectories)
    conflict_search = ConflictDrivenPatchSearch(
        partial_domain_template=deepcopy(partial_domain),
        negative_preconditions_policy=config.negative_precondition_policy,
        seed=config.seed,
        logger=None,
        search_mode=config.search_mode,
        fluent_patch_cost=config.fluent_patch_cost,
        fluent_patch_weight=config.fluent_patch_weight,
        model_patch_cost=config.model_patch_cost,
        model_constraint_weight=config.model_constraint_weight,
        node_choosing_strategy=config.node_choosing_strategy,
        conflict_group_strategy=config.conflict_group_strategy,
        fluent_branch_mode=config.fluent_branch_mode,
        conflict_free_models_dir=conflict_free_models_dir,
        save_t_prime_fn=save_t_prime_fn,
    )

    # Set up tracing callback if requested
    trace_log = None
    on_node_expanded = None
    if events_tracing:
        from src.plan_denoising.conflict_search import NodeExpansionEvent
        trace_log = []
        def on_node_expanded(event: NodeExpansionEvent) -> None:
            trace_log.append(event)

    search_result = conflict_search.run(
        observations=masked_observations,
        max_nodes=config.max_search_nodes,
        timeout_seconds=timeout_seconds,
        gt_states_by_obs=gt_states_by_obs,
        on_node_expanded=on_node_expanded,
    )
    learned_model = search_result.learned_domain
    report = search_result.report
    patched_observations = search_result.patched_observations

    # Write trace JSON if tracing was active
    if trace_log is not None and fold_work_dir is not None:
        from benchmark.diagnosis.trace_serialization import write_trace_json
        search_params = {
            "search_mode": config.search_mode.value,
            "node_choosing_strategy": config.node_choosing_strategy.value,
            "conflict_group_strategy": config.conflict_group_strategy.value,
            "fluent_branch_mode": config.fluent_branch_mode.value,
            "fluent_patch_cost": config.fluent_patch_cost,
            "model_patch_cost": config.model_patch_cost,
            "timeout_seconds": timeout_seconds,
        }
        trace_path = fold_work_dir / "search_trace.json"
        write_trace_json(trace_log, trace_path, search_params, fold_dir=fold_work_dir)

    model = learned_model.to_pddl()

    return model, report, patched_observations


def learn_cdps(
    domain_ref_path: Path,
    prepared_trajectories: List[Tuple[Path, Path, Path, Set[int]]],
    testing_dir: Path,
    conflict_search_timeout: int = None,
    fold_work_dir: Path = None,
    fluent_patch_cost: float = 1.0,
    fluent_patch_weight: float = 1.0,
    model_patch_cost: float = 1.0,
    model_constraint_weight: float = 0.0,
    max_search_nodes: int = None,
    search_mode: str = "dfs",
    node_choosing_strategy: str = "model_patch_first",
    conflict_group_strategy: str = "most_observations",
    fluent_branch_mode: str = "group",
    pre_built_observations: Optional[list] = None,
    gt_source_indices_override: Optional[Dict[int, Set[int]]] = None,
    anchor_endpoints: bool = False,
    events_tracing: bool = False,
) -> Tuple[str, dict, any]:
    """Learn a PISAM model via the Conflict-Directed Patch Search (CDPS).

    Args:
        prepared_trajectories: List of (trajectory_path, masking_path,
            problem_pddl_path, gt_state_indices). ``gt_state_indices`` are
            STATE indices (state s = source of component s = target of
            component s-1); they feed the search's single GT map.
        conflict_search_timeout: Optional conflict-search timeout in seconds.
        pre_built_observations: Optional pre-built Observation objects (simulated
            data); when provided, file loading is skipped.
        gt_source_indices_override: Optional explicit gt_states_by_obs map
            (obs_idx -> GT state indices), overriding the per-tuple indices.
        anchor_endpoints: Label for the ``cdps_anchored`` variant. GT protection
            itself is ALWAYS on (the search never patches a GT state and never
            keeps constraints refuted by GT evidence); this flag no longer
            changes search behavior — the anchored variant differs only in its
            trajectory prep, which injects the final state as GT so it shows up
            in ``gt_state_indices``.
        (other args configure the conflict search)

    Returns:
        Tuple of (model, learning_report, patched_observations).
    """
    gt_states_by_obs = _resolve_gt_states_by_obs(
        prepared_trajectories, gt_source_indices_override
    )

    traj_paths = [str(t[0]) for t in prepared_trajectories]
    config = CDPSConfig(
        fluent_patch_cost=fluent_patch_cost,
        fluent_patch_weight=fluent_patch_weight,
        model_patch_cost=model_patch_cost,
        model_constraint_weight=model_constraint_weight,
        max_search_nodes=max_search_nodes,
        search_mode=search_mode,
        node_choosing_strategy=node_choosing_strategy,
        conflict_group_strategy=conflict_group_strategy,
        fluent_branch_mode=fluent_branch_mode,
    )
    # Runtime budget stays separate from the search-shape config (default 60s,
    # matching the previous NOISY_PISAM default).
    timeout_seconds = conflict_search_timeout if conflict_search_timeout is not None else 60

    model, report, patched_observations = _learn_cdps_core(
        domain_ref_path, traj_paths, config, timeout_seconds,
        fold_work_dir=fold_work_dir, prepared_trajectories=prepared_trajectories,
        gt_states_by_obs=gt_states_by_obs,
        pre_built_observations=pre_built_observations,
        events_tracing=events_tracing,
    )

    if timeout_seconds is not None:
        if report is None:
            report = {}
        report['actual_timeout_seconds'] = timeout_seconds
        report['fluent_patch_cost'] = config.fluent_patch_cost
        report['fluent_patch_weight'] = config.fluent_patch_weight
        report['model_patch_cost'] = config.model_patch_cost
        report['model_constraint_weight'] = config.model_constraint_weight
        report['max_search_nodes'] = config.max_search_nodes
        report['search_mode'] = config.search_mode.value
        report['node_choosing_strategy'] = config.node_choosing_strategy.value
        report['conflict_group_strategy'] = config.conflict_group_strategy.value
        report['fluent_branch_mode'] = config.fluent_branch_mode.value

    return model, report, patched_observations


def _resolve_gt_states_by_obs(
    prepared_trajectories: List[Tuple[Path, Path, Path, Set[int]]],
    gt_source_indices_override: Optional[Dict[int, Set[int]]],
) -> Optional[Dict[int, Set[int]]]:
    """The single GT map ``obs_idx -> GT state indices``; explicit override wins."""
    if gt_source_indices_override is not None:
        return gt_source_indices_override
    return {
        obs_idx: set(t[3])
        for obs_idx, t in enumerate(prepared_trajectories)
        if len(t) > 3
    } or None


def _resolve_object_types_by_obs(
    partial_domain,
    prepared_trajectories: List[Tuple[Path, Path, Path, Set[int]]],
    observations: list,
) -> Optional[Dict[int, Dict[str, str]]]:
    """obs_idx -> declared object types, read off each trajectory's problem file.

    The MILP grounds each observation on ``Observation.grounded_objects``, whose
    types the trajectory parser had to *infer* — ``.trajectory`` files carry no
    ``(:objects ...)`` block. Inference reads types off predicate slots, so an
    object appearing in an **untyped** slot comes back as ``object`` and none of
    its actions ground. The problem file states the types outright, so it wins.
    See :func:`converter.problem_object_types` for the full failure mode.

    Returns ``None`` when no problem file is reachable, which leaves the drivers
    on inferred types — the behaviour before this existed.

    Raises:
        ValueError: if an observation mentions objects its paired problem does
            not declare. That pairing is positional, and a wrong pairing would
            silently stamp one problem's types onto another problem's objects;
            these domains reuse object names across problems, so the corruption
            would not otherwise show.
    """
    constants = set(getattr(partial_domain, "constants", {}) or {})
    types_by_obs: Dict[int, Dict[str, str]] = {}
    for obs_idx, observation in enumerate(observations):
        if obs_idx >= len(prepared_trajectories):
            break
        prepared = prepared_trajectories[obs_idx]
        if len(prepared) < 3 or prepared[2] is None:
            continue
        problem_path = Path(prepared[2])
        if not problem_path.exists():
            continue

        problem = ProblemParser(problem_path, partial_domain).parse_problem()
        declared = problem_object_types(problem)
        undeclared = sorted(
            set(observation.grounded_objects) - set(declared) - constants
        )
        if undeclared:
            raise ValueError(
                f"Observation {obs_idx} mentions object(s) {undeclared} that "
                f"{problem_path} does not declare — the observation and the "
                f"problem file are not the same problem, so the declared types "
                f"cannot be trusted for it."
            )
        types_by_obs[obs_idx] = declared
    return types_by_obs or None


def _prepare_milp_driver_inputs(
    domain_ref_path: Path,
    prepared_trajectories: List[Tuple[Path, Path, Path, Set[int]]],
    fold_work_dir: Optional[Path],
    pre_built_observations: Optional[list],
    gt_source_indices_override: Optional[Dict[int, Set[int]]],
    tag: str,
) -> dict:
    """The keyword arguments both MILP drivers take, built identically for both.

    Building them in one place is what makes the two arms comparable: they see
    the same observations, the same GT map, the same object types and the same
    PI-SAM settings, so any difference in their results comes from the denoiser
    and nothing else.
    """
    cdps_defaults = CDPSConfig()
    partial_domain = DomainParser(
        Path(str(domain_ref_path)), partial_parsing=True
    ).parse_domain()

    observations = _load_masked_observations(
        partial_domain, [str(t[0]) for t in prepared_trajectories], pre_built_observations
    )
    _save_original_observations(
        observations, fold_work_dir, prepared_trajectories,
        pre_built_observations, tag=tag,
    )

    return {
        "partial_domain": partial_domain,
        "observations": observations,
        "gt_states_by_obs": _resolve_gt_states_by_obs(
            prepared_trajectories, gt_source_indices_override
        ),
        "negative_preconditions_policy": cdps_defaults.negative_precondition_policy,
        "seed": cdps_defaults.seed,
        "fold_work_dir": fold_work_dir,
        "save_observations_fn": _make_save_observations_fn(prepared_trajectories),
        "object_types_by_obs": _resolve_object_types_by_obs(
            partial_domain, prepared_trajectories, observations
        ),
    }


def _milp_outcome(result, conflict_search_timeout: Optional[int]):
    """A driver result as the ``(model, report, observations)`` triple ``run_fold`` wants."""
    report = result.as_report()
    report["actual_timeout_seconds"] = conflict_search_timeout
    model = result.learned_domain.to_pddl() if result.learned_domain is not None else None
    return model, report, result.observations


def learn_cdps_milp_single_round(
    domain_ref_path: Path,
    prepared_trajectories: List[Tuple[Path, Path, Path, Set[int]]],
    testing_dir: Path,
    conflict_search_timeout: int = None,
    fold_work_dir: Path = None,
    milp_config: Optional[CdpsMilpConfig] = None,
    pre_built_observations: Optional[list] = None,
    gt_source_indices_override: Optional[Dict[int, Set[int]]] = None,
) -> Tuple[Optional[str], dict, list]:
    """Learn a PI-SAM model, denoising with one MILP solve instead of CDPS search.

    Drop-in sibling of :func:`learn_cdps`: same inputs, same
    ``(model, report, patched_observations)`` triple, same ``fold_work_dir``
    artifact layout. What changes is the denoiser — a single CP-SAT solve picks
    the globally cheapest repair, rather than a search that pays one PI-SAM run
    per node. See ``src/plan_denoising/milp_denoiser/single_round.py``.

    Args:
        prepared_trajectories: (trajectory, masking_info, problem_pddl,
            gt_state_indices) per training problem, exactly as for CDPS.
        conflict_search_timeout: The fold's denoiser budget. Handed to the solver
            unless ``milp_config.time_limit_seconds`` overrides it — sharing the
            budget is what makes the head-to-head against CDPS fair.
        milp_config: Validated ``cdps_milp`` block; ``None`` = defaults.
        gt_source_indices_override: Explicit GT map, overriding the per-tuple
            indices (same semantics as in :func:`learn_cdps`).

    Returns:
        ``(model_pddl_or_None, report, patched_observations)``. The model is
        ``None`` only when the solver returned nothing usable, in which case
        nothing downstream should be evaluated.
    """
    config = milp_config if milp_config is not None else CdpsMilpConfig()
    result = run_single_round(
        config=config,
        time_limit_seconds=config.resolve_time_limit(conflict_search_timeout),
        **_prepare_milp_driver_inputs(
            domain_ref_path, prepared_trajectories, fold_work_dir,
            pre_built_observations, gt_source_indices_override, tag="MILP",
        ),
    )
    return _milp_outcome(result, conflict_search_timeout)


def learn_cdps_milp_loop(
    domain_ref_path: Path,
    prepared_trajectories: List[Tuple[Path, Path, Path, Set[int]]],
    testing_dir: Path,
    conflict_search_timeout: int = None,
    fold_work_dir: Path = None,
    milp_config: Optional[CdpsMilpConfig] = None,
    pre_built_observations: Optional[list] = None,
    gt_source_indices_override: Optional[Dict[int, Set[int]]] = None,
) -> Tuple[Optional[str], dict, list]:
    """Learn a PI-SAM model by repeating the MILP solve over sampled subsets.

    Same contract as :func:`learn_cdps_milp_single_round`; the denoiser is the
    multi-round driver, which scores each round's model against the *original*
    noisy observations and keeps the best (``docs/cdps-milp-loop-plan.md``).

    Args:
        conflict_search_timeout: The fold's denoiser budget. Unlike the
            single-round arm, this caps the **whole loop**, not one solve — so
            the two arms and CDPS all get the same wall clock. ``stop.
            budget_seconds`` and ``time_limit_seconds`` override the loop total
            and the per-solve cap respectively, independently of each other.

    Returns:
        ``(model_pddl_or_None, report, patched_observations)``. ``None`` means
        no round ever produced a model.
    """
    config = milp_config if milp_config is not None else CdpsMilpConfig()
    result = run_loop(
        config=config,
        cdps_budget_seconds=conflict_search_timeout,
        **_prepare_milp_driver_inputs(
            domain_ref_path, prepared_trajectories, fold_work_dir,
            pre_built_observations, gt_source_indices_override, tag="MILP-LOOP",
        ),
    )
    return _milp_outcome(result, conflict_search_timeout)
