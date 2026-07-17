"""
Learning wrapper functions for AMLGym experiments.

Only one learning path exists: PISAM (partial observability) + the
Conflict-Directed Patch Search (CDPS) denoiser, on masked observations. The old
fully-observable (SAM) and plain (non-denoising) paths have been removed.
"""

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import time

from pddl_plus_parser.lisp_parsers import DomainParser

from benchmark.algorithm_adapters.NOISY_PISAM import NOISY_PISAM
from benchmark.experiment_running_helpers.cleaned_trajectories import (
    save_fold_observations,
    save_observations_to_dir,
)
from src.pi_sam.plan_denoising.conflict_search import ConflictDrivenPatchSearch
from src.pi_sam.plan_denoising.frontier import ConflictGroupStrategy, FluentBranchMode, NodeChoosingStrategy, SearchMode
from src.utils.masking import load_masked_observation


def _resolve_search_mode(search_mode: str | SearchMode) -> SearchMode:
    if isinstance(search_mode, SearchMode):
        return search_mode
    normalized = str(search_mode).strip().lower()
    if normalized == "ucs":
        return SearchMode.UCS
    if normalized == "dfs":
        return SearchMode.ANYTIME_DFS
    raise ValueError(f"Unsupported search_mode '{search_mode}'. Expected one of: dfs, ucs.")


def _resolve_node_choosing_strategy(
    node_choosing_strategy: str | NodeChoosingStrategy,
) -> NodeChoosingStrategy:
    if isinstance(node_choosing_strategy, NodeChoosingStrategy):
        return node_choosing_strategy
    normalized = str(node_choosing_strategy).strip().lower()
    if normalized == "model_patch_first":
        return NodeChoosingStrategy.MODEL_PATCH_FIRST
    if normalized == "fluent_patch_first":
        return NodeChoosingStrategy.FLUENT_PATCH_FIRST
    if normalized == "fluent_patch_first_then_model":
        return NodeChoosingStrategy.FLUENT_PATCH_FIRST_THEN_MODEL
    if normalized == "randomized":
        return NodeChoosingStrategy.RANDOMIZED
    raise ValueError(
        "Unsupported node_choosing_strategy "
        f"'{node_choosing_strategy}'. Expected one of: "
        "model_patch_first, fluent_patch_first, fluent_patch_first_then_model, randomized."
    )


def _resolve_conflict_group_strategy(
    conflict_group_strategy: str | ConflictGroupStrategy,
) -> ConflictGroupStrategy:
    if isinstance(conflict_group_strategy, ConflictGroupStrategy):
        return conflict_group_strategy
    normalized = str(conflict_group_strategy).strip().lower()
    if normalized == "first":
        return ConflictGroupStrategy.FIRST
    if normalized == "largest":
        return ConflictGroupStrategy.LARGEST
    if normalized == "largest_model_patchable":
        return ConflictGroupStrategy.LARGEST_MODEL_PATCHABLE
    if normalized == "most_observations":
        return ConflictGroupStrategy.MOST_OBSERVATIONS
    if normalized == "smallest":
        return ConflictGroupStrategy.SMALLEST
    raise ValueError(
        f"Unsupported conflict_group_strategy '{conflict_group_strategy}'. "
        "Expected one of: first, largest, largest_model_patchable, most_observations, smallest."
    )


def _resolve_fluent_branch_mode(
    fluent_branch_mode: str | FluentBranchMode,
) -> FluentBranchMode:
    if isinstance(fluent_branch_mode, FluentBranchMode):
        return fluent_branch_mode
    normalized = str(fluent_branch_mode).strip().lower()
    if normalized == "group":
        return FluentBranchMode.GROUP
    if normalized == "single":
        return FluentBranchMode.SINGLE
    raise ValueError(
        f"Unsupported fluent_branch_mode '{fluent_branch_mode}'. "
        "Expected one of: group, single."
    )


def _learn_pisam_with_profiling(
    domain_ref_path, traj_paths, learner, algo_name, profiler,
    fold_work_dir=None, prepared_trajectories=None, gt_source_indices_by_obs=None,
    pre_built_observations=None, events_tracing: bool = False,
):
    """Run the Conflict-Directed Patch Search (CDPS) with detailed profiling.

    Args:
        pre_built_observations: Optional list of pre-built Observation objects.
            When provided, file loading from traj_paths is skipped entirely
            (simulated data); otherwise observations are loaded from disk (image
            pipeline).
        events_tracing: If True, collect node expansion events and write
            search_trace.json to fold_work_dir after the search completes.
    """
    partial_domain = DomainParser(Path(str(domain_ref_path)), partial_parsing=True).parse_domain()

    if pre_built_observations is not None:
        masked_observations = pre_built_observations
    else:
        masked_observations = []
        for traj_idx, traj_path_str in enumerate(traj_paths):
            traj_path = Path(traj_path_str)
            masking_info_path = traj_path.parent / f"{traj_path.stem}.masking_info"

            if not masking_info_path.exists():
                continue

            def timing_callback(step_name, elapsed):
                profiler.add_detailed_timing(
                    "sam_pisam_trajectory_processing_cleaned",
                    step_name, elapsed,
                    {'trajectory_index': traj_idx, 'problem_name': traj_path.stem}
                )

            start_load = time.perf_counter()
            masked_obs = load_masked_observation(traj_path, masking_info_path, partial_domain, timing_callback=timing_callback)
            load_elapsed = time.perf_counter() - start_load

            profiler.add_detailed_timing(
                "sam_pisam_trajectory_loading_cleaned",
                'load_masked_observation_total',
                load_elapsed,
                {'trajectory_index': traj_idx, 'problem_name': traj_path.stem}
            )

            masked_observations.append(masked_obs)

    if fold_work_dir is not None and prepared_trajectories and masked_observations:
        original_obs_dir = fold_work_dir / "original_observations"
        save_fold_observations(
            masked_observations,
            prepared_trajectories,
            original_obs_dir,
            observation_prefix="original_observation",
        )
        print(f"  [CDPS] Saved {len(masked_observations)} original observations to {original_obs_dir.name}/")

    start_learn = time.perf_counter()

    conflict_free_models_dir = (fold_work_dir / "conflict_free_models") if fold_work_dir else None
    save_t_prime_fn = None
    if conflict_free_models_dir is not None and prepared_trajectories:
        save_t_prime_fn = lambda obs, out_dir: save_observations_to_dir(obs, prepared_trajectories, out_dir)
    conflict_search = ConflictDrivenPatchSearch(
        partial_domain_template=deepcopy(partial_domain),
        negative_preconditions_policy=learner.negative_precondition_policy,
        seed=learner.seed,
        logger=None,
        search_mode=_resolve_search_mode(learner.search_mode),
        fluent_patch_cost=learner.fluent_patch_cost,
        fluent_patch_weight=learner.fluent_patch_weight,
        model_patch_cost=learner.model_patch_cost,
        model_constraint_weight=learner.model_constraint_weight,
        node_choosing_strategy=_resolve_node_choosing_strategy(learner.node_choosing_strategy),
        conflict_group_strategy=_resolve_conflict_group_strategy(
            getattr(learner, 'conflict_group_strategy', ConflictGroupStrategy.FIRST)
        ),
        fluent_branch_mode=_resolve_fluent_branch_mode(
            getattr(learner, 'fluent_branch_mode', FluentBranchMode.GROUP)
        ),
        conflict_free_models_dir=conflict_free_models_dir,
        save_t_prime_fn=save_t_prime_fn,
    )

    # Set up tracing callback if requested
    trace_log = None
    on_node_expanded = None
    if events_tracing:
        from src.pi_sam.plan_denoising.conflict_search import NodeExpansionEvent
        trace_log = []
        def on_node_expanded(event: NodeExpansionEvent) -> None:
            trace_log.append(event)

    learned_model, _, _, _, _, report, patched_observations = conflict_search.run(
        observations=masked_observations,
        max_nodes=learner.max_search_nodes,
        timeout_seconds=learner.timeout_seconds,
        gt_source_indices_by_obs=gt_source_indices_by_obs,
        on_node_expanded=on_node_expanded,
    )

    # Write trace JSON if tracing was active
    if trace_log is not None and fold_work_dir is not None:
        from benchmark.diagnosis.trace_serialization import write_trace_json
        search_params = {
            "search_mode": learner.search_mode.value if hasattr(learner.search_mode, 'value') else str(learner.search_mode),
            "node_choosing_strategy": learner.node_choosing_strategy.value if hasattr(learner.node_choosing_strategy, 'value') else str(learner.node_choosing_strategy),
            "conflict_group_strategy": learner.conflict_group_strategy.value if hasattr(learner.conflict_group_strategy, 'value') else str(learner.conflict_group_strategy),
            "fluent_branch_mode": learner.fluent_branch_mode.value if hasattr(learner.fluent_branch_mode, 'value') else str(learner.fluent_branch_mode),
            "fluent_patch_cost": learner.fluent_patch_cost,
            "model_patch_cost": learner.model_patch_cost,
            "timeout_seconds": learner.timeout_seconds,
        }
        trace_path = fold_work_dir / "search_trace.json"
        write_trace_json(trace_log, trace_path, search_params, fold_dir=fold_work_dir)

    model = learned_model.to_pddl()

    if profiler:
        profiler.add_timing(f"learning_process_{algo_name}_cleaned", time.perf_counter() - start_learn)
    return model, report, patched_observations


def learn_sam_pisam(
    domain_ref_path: Path,
    prepared_trajectories: List[Tuple[Path, Path, Path, Set[int]]],
    testing_dir: Path,
    conflict_search_timeout: int = None,
    profiler=None,
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
    events_tracing: bool = False,
) -> Tuple[str, dict, str, any]:
    """Learn a PISAM model via the Conflict-Directed Patch Search (CDPS).

    Args:
        prepared_trajectories: List of (trajectory_path, masking_path,
            problem_pddl_path, gt_state_indices).
        conflict_search_timeout: Optional conflict-search timeout in seconds.
        pre_built_observations: Optional pre-built Observation objects (simulated
            data); when provided, file loading is skipped.
        gt_source_indices_override: Optional explicit gt_source_indices_by_obs.
        (other args configure the conflict search)

    Returns:
        Tuple of (model, learning_report, algorithm_name, patched_observations).
    """
    algo_name = 'PISAM'

    # Determine GT source indices: explicit override takes priority.
    if gt_source_indices_override is not None:
        gt_source_indices_by_obs = gt_source_indices_override
    else:
        gt_source_indices_by_obs: Optional[Dict[int, Set[int]]] = {
            obs_idx: t[3] for obs_idx, t in enumerate(prepared_trajectories) if len(t) > 3
        } or None

    traj_paths = [str(t[0]) for t in prepared_trajectories]
    learner = NOISY_PISAM()
    if conflict_search_timeout is not None:
        learner.timeout_seconds = conflict_search_timeout
    learner.fluent_patch_cost = fluent_patch_cost
    learner.fluent_patch_weight = fluent_patch_weight
    learner.model_patch_cost = model_patch_cost
    learner.model_constraint_weight = model_constraint_weight
    learner.max_search_nodes = max_search_nodes
    learner.search_mode = _resolve_search_mode(search_mode)
    learner.node_choosing_strategy = _resolve_node_choosing_strategy(node_choosing_strategy)
    learner.conflict_group_strategy = _resolve_conflict_group_strategy(conflict_group_strategy)
    learner.fluent_branch_mode = _resolve_fluent_branch_mode(fluent_branch_mode)
    actual_learning_timeout = learner.timeout_seconds

    model, report, patched_observations = _learn_pisam_with_profiling(
        domain_ref_path, traj_paths, learner, algo_name, profiler,
        fold_work_dir=fold_work_dir, prepared_trajectories=prepared_trajectories,
        gt_source_indices_by_obs=gt_source_indices_by_obs,
        pre_built_observations=pre_built_observations,
        events_tracing=events_tracing,
    )

    if actual_learning_timeout is not None:
        if report is None:
            report = {}
        report['actual_timeout_seconds'] = actual_learning_timeout
        report['fluent_patch_cost'] = fluent_patch_cost
        report['fluent_patch_weight'] = fluent_patch_weight
        report['model_patch_cost'] = model_patch_cost
        report['model_constraint_weight'] = model_constraint_weight
        report['max_search_nodes'] = max_search_nodes
        report['search_mode'] = _resolve_search_mode(search_mode).value
        report['node_choosing_strategy'] = _resolve_node_choosing_strategy(node_choosing_strategy).value
        report['conflict_group_strategy'] = _resolve_conflict_group_strategy(conflict_group_strategy).value
        report['fluent_branch_mode'] = _resolve_fluent_branch_mode(fluent_branch_mode).value

    return model, report, algo_name, patched_observations
