"""
Helper functions for running benchmark experiments.

This module contains the main fold execution logic for the experiment_runner experiments.
"""

import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser

from benchmark.algorithms import CDPS_ALGORITHM_NAME
from benchmark.experiment_running_helpers.cleaned_trajectories import save_patched_observations
from benchmark.experiment_running_helpers.data_source import DataSource
from benchmark.experiment_running_helpers.post_process_gt_metrics import run_post_process_gt_metrics
from benchmark.experiment_running_helpers.learning_helpers import learn_cdps
from benchmark.experiment_running_helpers.result_builders import evaluate_and_build_result
from benchmark.experiment_running_helpers.resume import fold_instance_dir, save_fold_result
from benchmark.experiment_running_helpers.statistics import count_total_transitions_and_gt, load_learning_metrics
from benchmark.experiment_running_helpers.trajectory_utils import save_fold_metadata, update_fold_metadata
from benchmark.evaluation.test_states_generator import generate_predictive_power_test_states
from benchmark.evaluation.multi_solution_evaluator import evaluate_all_solutions
from benchmark.evaluation.correlation_analysis import build_correlation_table
from src.utils.pddl import ground_observation_completely, observations_equal


def check_trajectories_equal(
    prepared_trajectories: List[Tuple[Path, Path, Path]],
    patched_observations,
    domain_ref_path: Path,
):
    """
    Check whether CDPS's patched observations equal the original input trajectories.

    Args:
        prepared_trajectories: List of (trajectory_path, masking_path, problem_pddl_path).
        patched_observations: List of patched Observation objects from CDPS.
        domain_ref_path: Path to domain file.

    Returns:
        True if all patched observations equal their input, False if any differ,
        None if the check cannot be performed.
    """
    if patched_observations is None or len(patched_observations) != len(prepared_trajectories):
        return None

    domain = DomainParser(domain_ref_path).parse_domain()
    parser = TrajectoryParser(domain)

    for idx, (traj_path, *_) in enumerate(prepared_trajectories):
        original_obs = parser.parse_trajectory(traj_path)
        fully_grounded_original_obs = ground_observation_completely(domain, original_obs)
        if not observations_equal(fully_grounded_original_obs, patched_observations[idx]):
            return False
    return True


def _run_baselines(
    baselines,
    domain_ref_path: Path,
    trajectories: List[Tuple[Path, Path, Path]],
    fold_work_dir: Path,
    bench_name: str,
    fold: int,
    num_trajectories: int,
    gt_rate: int,
    test_problem_paths: List[str],
    evaluate_model_func,
    null_metrics: dict,
    testing_dir: Path,
    total_transitions: int,
    total_gt_transitions: int,
    conflict_search_timeout: int,
    planning_timeout: int,
    test_states_path_str: str,
) -> List[dict]:
    """Run each baseline runner once on the (degraded) trajectories → result rows."""
    results = []
    if not baselines:
        return results

    for runner in baselines:
        algo_name = runner.name
        print(f"  [{algo_name}] Starting {runner.display_name} learning...")

        learn_start = time.perf_counter()
        model, extra_info = runner.learn(
            domain_path=domain_ref_path,
            prepared_trajectories=trajectories,
            work_dir=fold_work_dir,
            timeout_seconds=conflict_search_timeout or 60,
        )
        learn_time = time.perf_counter() - learn_start

        print(f"  [{algo_name}] Evaluating {runner.display_name} model...")
        result = evaluate_and_build_result(
            model, algo_name, bench_name, fold, num_trajectories, gt_rate,
            test_problem_paths, domain_ref_path, testing_dir,
            evaluate_model_func, null_metrics, fold_work_dir,
            total_transitions=total_transitions,
            total_gt_transitions=total_gt_transitions,
            learning_time_seconds=learn_time,
            algorithm_specific=extra_info or {},
            planning_timeout=planning_timeout,
            test_states_path=test_states_path_str,
        )
        results.append(result)

    return results


def run_single_fold(
    fold: int,
    problem_dirs: List[Path],
    n_problems: int,
    num_trajectories: int,
    gt_rate: int,
    domain_ref_path: Path,
    testing_dir: Path,
    bench_name: str,
    data_source: DataSource,
    evaluate_model_func,
    save_learning_metrics_func,
    conflict_search_timeout: int = None,
    planning_timeout: int = 60,
    fluent_patch_cost: float = 1.0,
    fluent_patch_weight: float = 1.0,
    model_patch_cost: float = 1.0,
    model_constraint_weight: float = 0.0,
    max_search_nodes: int = None,
    search_mode: str = "dfs",
    node_choosing_strategy: str = "model_patch_first",
    conflict_group_strategy: str = "most_observations",
    fluent_branch_mode: str = "group",
    trajectory_seed: Optional[int] = None,
    output_subdir: Optional[str] = None,
    baselines: Optional[list] = None,
    run_cdps: bool = True,
    events_tracing: bool = False,
) -> List[dict]:
    """
    Run a single fold experiment with specified number of trajectories and GT rate.

    Args:
        fold: Fold number
        problem_dirs: List of all problem directories
        n_problems: Total number of problems
        num_trajectories: Number of trajectories to use for learning (1-8)
        gt_rate: Percentage of states to inject as GT (0, 10, 25, 50, 75, 100)
        domain_ref_path: Path to reference domain PDDL file
        testing_dir: Directory for test results
        bench_name: Benchmark domain name
        data_source: DataSource instance that supplies observations for this fold.
        evaluate_model_func: Function to evaluate a learned model
        save_learning_metrics_func: Function to save learning metrics
        conflict_search_timeout: Optional timeout in seconds for conflict search (cleaning phase)
        planning_timeout: Timeout in seconds for planning during evaluation (default: 60)
        fluent_patch_cost: Per-patch cost for fluent patches in denoising conflict search.
        fluent_patch_weight: Weight multiplier for fluent patch cost in denoising conflict search.
        model_patch_cost: Per-patch cost for model patches in denoising conflict search.
        model_constraint_weight: Weight multiplier for model constraint cost in denoising conflict search.
        max_search_nodes: Max conflict-search nodes in denoising phase (None = unlimited).
        search_mode: Conflict-search strategy for denoising ("dfs" or "ucs").
        node_choosing_strategy: Conflict-search branch insertion ordering strategy.
        conflict_group_strategy: Which conflict group to resolve first at each node.
        fluent_branch_mode: How many fluent patches per data-fix branch.
        trajectory_seed: Optional override for trajectory-pool sampling seed.
        output_subdir: Optional subdirectory under the fold work dir.
        baselines: Optional list of BaselineRunner instances to run alongside
            CDPS. When None or empty, only CDPS runs (subject to run_cdps).

    Returns:
        List of result dicts: the baseline rows plus the CDPS row (when run).
    """
    if baselines is None:
        baselines = []

    print(f"[PID {os.getpid()}] Fold {fold}, num_trajs={num_trajectories}, gt_rate={gt_rate}%")
    if baselines:
        baseline_names = [r.display_name for r in baselines]
        print(f"  Baselines: {', '.join(baseline_names) if baseline_names else '(none)'}")

    # Setup
    fold_work_dir = fold_instance_dir(testing_dir, fold, num_trajectories, gt_rate)
    if output_subdir is not None:
        fold_work_dir = fold_work_dir / output_subdir
    fold_work_dir.mkdir(parents=True, exist_ok=True)
    original_cwd = os.getcwd()
    os.chdir(fold_work_dir)

    # Copy domain reference file to fold directory to avoid race conditions
    local_domain_ref = fold_work_dir / "domain_reference.pddl"
    shutil.copy2(domain_ref_path, local_domain_ref)
    domain_ref_path = local_domain_ref

    null_metrics = {k: None for k in ['precision_precs_pos', 'precision_precs_neg',
                    'precision_eff_pos', 'precision_eff_neg', 'precision_overall',
                    'recall_precs_pos', 'recall_precs_neg', 'recall_eff_pos',
                    'recall_eff_neg', 'recall_overall', 'solving_ratio',
                    'false_plans_ratio', 'unsolvable_ratio', 'planning_timed_out_ratio',
                    'pred_app_precision', 'pred_app_recall',
                    'pred_eff_precision', 'pred_eff_recall']}

    try:
        # ==================================================
        # Setup: CV split and trajectory preparation
        # ==================================================
        indices = list(range(n_problems))
        random.seed(42 + fold)
        random.shuffle(indices)
        n_train = max(1, min(int(0.8 * n_problems), n_problems - 1))
        train_idx, test_idx = indices[:n_train], indices[n_train:]

        train_problem_dirs = [problem_dirs[i] for i in train_idx]
        test_problem_dirs = [problem_dirs[i] for i in test_idx]

        random.seed(trajectory_seed if trajectory_seed is not None else 42 + fold)
        if trajectory_seed is not None:
            selected_pool = random.sample(
                train_problem_dirs,
                min(num_trajectories, len(train_problem_dirs)),
            )
        else:
            selected_pool = random.sample(train_problem_dirs, min(n_train, len(train_problem_dirs)))

        # Prepare observations via the data source (file-based or simulated)
        prepared_trajectories, pre_built_observations, gt_source_indices = \
            data_source.prepare(selected_pool, num_trajectories, gt_rate, fold, fold_work_dir)

        if not prepared_trajectories:
            print(f"  ERROR: No trajectories prepared for fold {fold}")
            return []

        print(f"  ✓ Prepared {len(prepared_trajectories)} trajectories")

        # Build test problem paths
        test_problem_paths = []
        for d in test_problem_dirs:
            problem_pddl_path = d / f"{d.name}.pddl"
            if problem_pddl_path.exists():
                test_problem_paths.append(str(problem_pddl_path))
            else:
                pddl_files = list(d.glob("*.pddl"))
                if pddl_files:
                    test_problem_paths.append(str(pddl_files[0]))
                    print(f"  Warning: Used glob fallback for {d.name}, found {pddl_files[0].name}")
                else:
                    print(f"  Warning: No PDDL file found in test directory {d.name}")

        if not test_problem_paths:
            print(f"  ERROR: No test problems found for fold {fold}")
            print(f"  Test directories: {[d.name for d in test_problem_dirs]}")
            return []

        if len(test_problem_paths) < len(test_problem_dirs):
            print(f"  Warning: Only found {len(test_problem_paths)} test problems out of {len(test_problem_dirs)} directories")

        # ==================================================
        # Generate S_test (predictive power test states)
        # ==================================================
        print(f"  [S_TEST] Generating predictive power test states...")
        test_states_dir = fold_work_dir / "predictive_power_test_states"
        test_states_path = generate_predictive_power_test_states(
            domain_ref_path=domain_ref_path,
            test_problem_paths=test_problem_paths,
            output_dir=test_states_dir,
            num_trajectories_per_problem=50,
            seed=42 + fold,
        )
        test_states_path_str = str(test_states_path)
        print(f"  [S_TEST] Test states ready at {test_states_path.name}")

        # Save fold metadata with test problem names
        save_fold_metadata(fold_work_dir, prepared_trajectories, fold, num_trajectories, gt_rate, test_problem_paths)

        # Count total transitions and GT transitions
        total_transitions, total_gt_transitions = count_total_transitions_and_gt(
            prepared_trajectories
        )
        print(f"  [STATS] {total_transitions} transitions, {total_gt_transitions} GT states")

        # Common kwargs for _run_baselines
        baseline_common = dict(
            baselines=baselines,
            domain_ref_path=domain_ref_path,
            fold_work_dir=fold_work_dir,
            bench_name=bench_name,
            fold=fold,
            num_trajectories=num_trajectories,
            gt_rate=gt_rate,
            test_problem_paths=test_problem_paths,
            evaluate_model_func=evaluate_model_func,
            null_metrics=null_metrics,
            testing_dir=testing_dir,
            conflict_search_timeout=conflict_search_timeout,
            planning_timeout=planning_timeout,
            test_states_path_str=test_states_path_str,
        )

        # ==================================================
        # BASELINES — each learns once from the degraded trajectories
        # ==================================================
        baseline_results = _run_baselines(
            trajectories=prepared_trajectories,
            total_transitions=total_transitions,
            total_gt_transitions=total_gt_transitions,
            **baseline_common,
        )

        # ==================================================
        # CDPS — Conflict-Directed Patch Search (our algorithm)
        # ==================================================
        cdps_result = None
        if run_cdps:
            try:
                print(f"  [CDPS] Starting conflict-directed patch search...")
                if conflict_search_timeout is not None:
                    print(f"  [CDPS] Using conflict search timeout: {conflict_search_timeout}s")
                cleaned_model, denoising_report, patched_observations = learn_cdps(
                    domain_ref_path, prepared_trajectories, testing_dir,
                    conflict_search_timeout=conflict_search_timeout,
                    fold_work_dir=fold_work_dir,
                    fluent_patch_cost=fluent_patch_cost,
                    fluent_patch_weight=fluent_patch_weight,
                    model_patch_cost=model_patch_cost,
                    model_constraint_weight=model_constraint_weight,
                    max_search_nodes=max_search_nodes,
                    search_mode=search_mode,
                    node_choosing_strategy=node_choosing_strategy,
                    conflict_group_strategy=conflict_group_strategy,
                    fluent_branch_mode=fluent_branch_mode,
                    pre_built_observations=pre_built_observations,
                    gt_source_indices_override=gt_source_indices,
                    events_tracing=events_tracing,
                )
                print(f"  [CDPS] Search complete, saving metrics...")
                save_learning_metrics_func(fold_work_dir, denoising_report)

                denoising_learning_metrics = load_learning_metrics(fold_work_dir)

                # CDPS-owned extras (nested under algorithm_specific).
                lm = denoising_learning_metrics or {}
                cdps_specific = {
                    "nodes_in_cleaning_tree": lm.get("nodes_expanded"),
                    "conflict_free_model_count": lm.get("conflict_free_model_count"),
                    "terminated_by": lm.get("terminated_by"),
                    "conflict_search_timeout_seconds": conflict_search_timeout,
                    "learning_timeout_seconds": (
                        lm.get("actual_timeout_seconds")
                        if lm.get("actual_timeout_seconds") is not None else conflict_search_timeout
                    ),
                    "timeout_during_cleaning": lm.get("timeout_during_learning"),
                }

                print(f"  [CDPS] Evaluating learned model...")
                cdps_result = evaluate_and_build_result(
                    cleaned_model, CDPS_ALGORITHM_NAME, bench_name, fold, num_trajectories, gt_rate,
                    test_problem_paths, domain_ref_path, testing_dir,
                    evaluate_model_func, null_metrics, fold_work_dir,
                    total_transitions=total_transitions,
                    total_gt_transitions=total_gt_transitions,
                    learning_time_seconds=lm.get("learning_time_seconds"),
                    algorithm_specific=cdps_specific,
                    planning_timeout=planning_timeout,
                    test_states_path=test_states_path_str,
                )

                # Did the search change the data? (kept for fold metadata)
                patched_equals_input = check_trajectories_equal(
                    prepared_trajectories, patched_observations, domain_ref_path
                )
                if patched_equals_input is not None:
                    print(f"  [CDPS] Patched vs input trajectories are "
                          f"{'EQUAL' if patched_equals_input else 'DIFFERENT'}")

                # Save patched observations + evaluate all conflict-free models
                if patched_observations is not None:
                    print(f"  [CDPS] Saving {len(patched_observations)} patched observations...")
                    final_observations_dir = fold_work_dir / "final_observations"
                    save_patched_observations(
                        patched_observations, prepared_trajectories, final_observations_dir, domain_ref_path
                    )
                    run_post_process_gt_metrics(fold_work_dir, prepared_trajectories, domain_ref_path, gt_rate)
                    update_fold_metadata(
                        fold_work_dir, patched_equals_input=patched_equals_input,
                    )

                    conflict_free_models_dir = fold_work_dir / "conflict_free_models"
                    if conflict_free_models_dir.exists():
                        print(f"  [MULTI-EVAL] Evaluating all conflict-free models...")
                        all_solutions_results = evaluate_all_solutions(
                            conflict_free_models_dir=conflict_free_models_dir,
                            ref_domain_path=domain_ref_path,
                            test_problem_paths=test_problem_paths,
                            test_states_path=test_states_path,
                            planning_timeout=planning_timeout,
                            output_dir=fold_work_dir,
                        )
                        print(f"  [MULTI-EVAL] Evaluated {len(all_solutions_results)} solutions")
                        build_correlation_table(fold_work_dir)

            except Exception as e:
                print(f"  ERROR in CDPS phase: {e}")
                import traceback
                traceback.print_exc()
                cdps_result = evaluate_and_build_result(
                    None, CDPS_ALGORITHM_NAME, bench_name, fold, num_trajectories, gt_rate,
                    test_problem_paths, domain_ref_path, testing_dir,
                    evaluate_model_func, null_metrics, fold_work_dir,
                    total_transitions=total_transitions,
                    total_gt_transitions=total_gt_transitions,
                    learning_time_seconds=None,
                    algorithm_specific={"conflict_search_timeout_seconds": conflict_search_timeout,
                                        "error": str(e)},
                    planning_timeout=planning_timeout,
                    test_states_path=test_states_path_str,
                )

        # Build results: baselines + our CDPS row (when run)
        fold_results = baseline_results + ([cdps_result] if cdps_result is not None else [])

        # Write the resume marker last: its presence means this fold is fully done.
        save_fold_result(fold_work_dir, fold_results)
        print(f"  [FOLD COMPLETE] Returning {len(fold_results)} results for fold {fold}")
        return fold_results

    finally:
        os.chdir(original_cwd)
