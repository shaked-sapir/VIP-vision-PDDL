"""
Learning algorithm wrapper functions for AMLGym experiments.
"""

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import time

from pddl_plus_parser.lisp_parsers import DomainParser
from utilities import NegativePreconditionPolicy

from benchmark.amlgym_models.NOISY_PISAM import NOISY_PISAM
from benchmark.amlgym_models.NOISY_SAM import NOISY_SAM
from benchmark.amlgym_models.PISAM import PISAM
from benchmark.amlgym_models.PO_ROSAME import PO_ROSAME
from benchmark.amlgym_models.ROSAME import ROSAME
from benchmark.amlgym_models.SAM import SAM
from benchmark.experiment_running_helpers.cleaned_trajectories import save_observations_to_dir
from benchmark.experiment_running_helpers.trajectory_utils import setup_algorithm_workspace
from src.pi_sam import PISAMLearner
from src.pi_sam.plan_denoising.conflict_search import ConflictDrivenPatchSearch
from src.pi_sam.plan_denoising.frontier import ConflictGroupStrategy, FluentBranchMode, NodeChoosingStrategy, SearchMode
from src.utils.masking import load_masked_observation


def _parse_learning_output(learning_output, is_denoising):
    """Extract model, patched_observations, and report from learning output."""
    if is_denoising and isinstance(learning_output, tuple) and len(learning_output) == 3:
        return learning_output[0], learning_output[1], learning_output[2]
    return learning_output, None, {}


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
    domain_ref_path, traj_paths, is_denoising, learner, phase, algo_name, profiler,
    fold_work_dir=None, prepared_trajectories=None, gt_source_indices_by_obs=None,
    pre_built_observations=None,
):
    """Learn PISAM with detailed profiling.

    Args:
        pre_built_observations: Optional list of pre-built Observation objects.
            When provided, file loading from traj_paths is skipped entirely.
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
                    f"sam_pisam_trajectory_processing_{phase}",
                    step_name, elapsed,
                    {'trajectory_index': traj_idx, 'problem_name': traj_path.stem}
                )

            start_load = time.perf_counter()
            masked_obs = load_masked_observation(traj_path, masking_info_path, partial_domain, timing_callback=timing_callback)
            load_elapsed = time.perf_counter() - start_load

            profiler.add_detailed_timing(
                f"sam_pisam_trajectory_loading_{phase}",
                'load_masked_observation_total',
                load_elapsed,
                {'trajectory_index': traj_idx, 'problem_name': traj_path.stem}
            )

            masked_observations.append(masked_obs)
    
    start_learn = time.perf_counter()
    
    if is_denoising:
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
        learned_model, _, _, _, _, report, patched_observations = conflict_search.run(
            observations=masked_observations,
            max_nodes=learner.max_search_nodes,
            timeout_seconds=learner.timeout_seconds,
            gt_source_indices_by_obs=gt_source_indices_by_obs,
        )
        model = learned_model.to_pddl()
    else:
        pi_sam = PISAMLearner(partial_domain=partial_domain, negative_preconditions_policy=NegativePreconditionPolicy.hard)
        learned_model, _ = pi_sam.learn_action_model(masked_observations)
        model = learned_model.to_pddl()
        patched_observations = None
        report = {}
    
    if profiler:
        profiler.add_timing(f"learning_process_{algo_name}_{phase}", time.perf_counter() - start_learn)
    return model, report, patched_observations


def learn_sam_pisam(
    mode: str,
    domain_ref_path: Path,
    prepared_trajectories: List[Tuple[Path, Path, Path, Set[int]]],
    testing_dir: Path,
    is_denoising: bool = False,
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
) -> Tuple[str, dict, str, any]:
    """
    Learn SAM/PISAM model.

    Args:
        mode: 'masked' or 'fullyobs'
        domain_ref_path: Path to reference domain PDDL
        prepared_trajectories: List of (trajectory_path, masking_path, problem_pddl_path, gt_state_indices)
        testing_dir: Working directory
        is_denoising: If True, use NOISY_PISAM/NOISY_SAM (returns learning report and patched observations)
        conflict_search_timeout: Optional timeout in seconds for conflict search (cleaning phase)
        profiler: Optional TimingProfiler instance for detailed timing
        fold_work_dir: Optional fold working directory for saving conflict-free models
        fluent_patch_cost: Per-patch cost for fluent patches in denoising conflict search
        fluent_patch_weight: Weight multiplier for fluent patch cost in denoising conflict search
        model_patch_cost: Per-patch cost for model constraints in denoising conflict search
        model_constraint_weight: Weight multiplier for model constraint cost in denoising conflict search
        max_search_nodes: Max denoising conflict-search nodes (None = unlimited)
        search_mode: Conflict-search strategy for denoising ("dfs" or "ucs")
        node_choosing_strategy: Branch insertion ordering strategy in conflict search
        conflict_group_strategy: Which conflict group to resolve first at each node
            ("first", "largest", "largest_model_patchable", "most_observations", "smallest")
        fluent_branch_mode: How many fluent patches per data-fix branch
            ("group" = all in group, "single" = one at a time)
        pre_built_observations: Optional list of pre-built Observation objects.
            When provided, file-based loading is skipped entirely (simulated data mode).
        gt_source_indices_override: Optional explicit gt_source_indices_by_obs dict.
            When provided, overrides the indices extracted from prepared_trajectories.

    Returns:
        Tuple of (model, learning_report, algorithm_name, patched_observations)
    """
    phase = "cleaned" if is_denoising else "unclean"
    algo_name = 'PISAM' if mode == 'masked' else 'SAM'

    # Determine GT source indices: explicit override takes priority
    if gt_source_indices_override is not None:
        gt_source_indices_by_obs = gt_source_indices_override
    else:
        gt_source_indices_by_obs: Optional[Dict[int, Set[int]]] = {
            obs_idx: t[3] for obs_idx, t in enumerate(prepared_trajectories) if len(t) > 3
        } or None

    # Track the actual timeout value used (for cleaned phase only)
    actual_learning_timeout = None
    
    if mode == 'masked':
        traj_paths = [str(t[0]) for t in prepared_trajectories]
        learner = NOISY_PISAM() if is_denoising else PISAM()
        if is_denoising:
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
            # Capture actual timeout used (either explicit or default)
            actual_learning_timeout = learner.timeout_seconds

        if profiler or pre_built_observations is not None:
            # Use direct learning path (required when observations are pre-built;
            # also used when profiler is available for detailed timing).
            model, report, patched_observations = _learn_pisam_with_profiling(
                domain_ref_path, traj_paths, is_denoising, learner, phase, algo_name, profiler,
                fold_work_dir=fold_work_dir, prepared_trajectories=prepared_trajectories,
                gt_source_indices_by_obs=gt_source_indices_by_obs,
                pre_built_observations=pre_built_observations,
            )
        else:
            learn_kwargs = {}
            if is_denoising:
                learn_kwargs["gt_source_indices_by_obs"] = gt_source_indices_by_obs
            learning_output = learner.learn(
                str(domain_ref_path), traj_paths, use_problems=False, **learn_kwargs,
            )
            model, patched_observations, report = _parse_learning_output(learning_output, is_denoising)
    else:  # fullyobs
        workspace_name = "noisy_sam" if is_denoising else "sam_unclean"
        traj_paths = setup_algorithm_workspace(prepared_trajectories, workspace_name, testing_dir, mode)
        learner = NOISY_SAM() if is_denoising else SAM()
        if is_denoising:
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
            # Capture actual timeout used (either explicit or default)
            actual_learning_timeout = learner.timeout_seconds

        learn_kwargs = {}
        if is_denoising:
            learn_kwargs["gt_source_indices_by_obs"] = gt_source_indices_by_obs
        learning_output = learner.learn(
            str(domain_ref_path), traj_paths, use_problems=False, **learn_kwargs,
        )
        model, patched_observations, report = _parse_learning_output(learning_output, is_denoising)
    
    # Add actual timeout to report if denoising (cleaned phase)
    if is_denoising and actual_learning_timeout is not None:
        if report is None:
            report = {}
        report['actual_timeout_seconds'] = actual_learning_timeout
        report['fluent_patch_cost'] = fluent_patch_cost
        report['fluent_patch_weight'] = fluent_patch_weight
        report['model_patch_cost'] = model_patch_cost
        report['model_constraint_weight'] = model_constraint_weight
        report['max_search_nodes'] = max_search_nodes
        report['search_mode'] = _resolve_search_mode(search_mode).value
        report['node_choosing_strategy'] = (
            _resolve_node_choosing_strategy(node_choosing_strategy).value
        )
        report['conflict_group_strategy'] = (
            _resolve_conflict_group_strategy(conflict_group_strategy).value
        )
        report['fluent_branch_mode'] = (
            _resolve_fluent_branch_mode(fluent_branch_mode).value
        )
    
    return model, report, algo_name, patched_observations


def learn_rosame(
    mode: str,
    domain_ref_path: Path,
    prepared_trajectories: List[Tuple[Path, Path, Path]],
    testing_dir: Path,
    workspace_name: str,
    profiler=None
) -> Tuple[str, dict, str]:
    """
    Learn ROSAME/PO_ROSAME model.

    Args:
        mode: 'masked' (uses PO_ROSAME) or 'fullyobs' (uses ROSAME)
        domain_ref_path: Path to reference domain PDDL
        prepared_trajectories: List of (trajectory_path, masking_path, problem_pddl_path)
        testing_dir: Working directory
        workspace_name: Name for workspace directory

    Returns:
        Tuple of (model, learning_report, algorithm_name)
        - model: PDDL model string or None on failure
        - learning_report: Always {} (ROSAME has no learning report)
        - algorithm_name: "PO_ROSAME" or "ROSAME"
    """
    algo_name = 'PO_ROSAME' if mode == 'masked' else 'ROSAME'
    print(f"  [DEBUG] Setting up workspace for {workspace_name}...")
    traj_paths = setup_algorithm_workspace(prepared_trajectories, workspace_name, testing_dir, mode)
    print(f"  [DEBUG] Workspace setup complete, {len(traj_paths)} trajectories")

    if not traj_paths:
        print(f"  [DEBUG] No trajectories for ROSAME, skipping")
        return None, {}, algo_name

    try:
        print(f"  [DEBUG] Starting {algo_name} learning...")
        # Call static method directly on the class
        if mode == 'masked':
            model = PO_ROSAME.learn(str(domain_ref_path), traj_paths, use_problems=False, profiler=profiler)
        else:
            model = ROSAME.learn(str(domain_ref_path), traj_paths, use_problems=False, profiler=profiler)
        print(f"  [DEBUG] {algo_name} learning complete")

        if model and ":action" in model:
            return model, {}, algo_name
        else:
            raise ValueError("Invalid ROSAME model")
    except Exception as e:
        print(f"  Warning: ROSAME learning failed: {e}")
        print(f"  Domain ref: {domain_ref_path}")
        print(f"  Num trajectories: {len(traj_paths)}")
        print(f"  Workspace: {workspace_name}")

        # Check if any files are empty or malformed
        if domain_ref_path.exists():
            size = domain_ref_path.stat().st_size
            print(f"  Domain file size: {size} bytes")
            if size == 0:
                print(f"  ERROR: Domain file is EMPTY!")
        else:
            print(f"  ERROR: Domain file does not exist!")

        for i, traj_path in enumerate(traj_paths):
            traj_file = Path(traj_path)
            if traj_file.exists():
                size = traj_file.stat().st_size
                if size == 0:
                    print(f"  ERROR: Trajectory {i} is EMPTY: {traj_path}")
            else:
                print(f"  ERROR: Trajectory {i} does not exist: {traj_path}")

        return None, {}, algo_name

