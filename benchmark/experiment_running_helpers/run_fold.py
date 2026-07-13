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

from benchmark.experiment_running_helpers.cleaned_trajectories import convert_cleaned_dir_to_trajectory_list, save_patched_observations
from benchmark.experiment_running_helpers.data_source import DataSource
from benchmark.experiment_running_helpers.post_process_gt_metrics import run_post_process_gt_metrics
from benchmark.experiment_running_helpers.learning_helpers import learn_rosame, learn_sam_pisam
from benchmark.experiment_running_helpers.profiling import TimingProfiler
from benchmark.experiment_running_helpers.result_builders import evaluate_and_build_result
from benchmark.experiment_running_helpers.statistics import count_total_transitions_and_gt, load_learning_metrics
from benchmark.experiment_running_helpers.trajectory_utils import save_fold_metadata, update_fold_metadata
from benchmark.evaluation.test_states_generator import generate_predictive_power_test_states
from benchmark.evaluation.multi_solution_evaluator import evaluate_all_solutions
from benchmark.evaluation.correlation_analysis import build_correlation_table
from src.utils.pddl import ground_observation_completely, observations_equal


def check_trajectories_equal(
    prepared_trajectories: List[Tuple[Path, Path, Path]],
    cleaned_observations_or_paths,
    domain_ref_path: Path,
    is_patched_observations: bool = False
):
    """
    Check if cleaned and unclean trajectories are equal.
    
    Args:
        prepared_trajectories: List of (trajectory_path, masking_path, problem_pddl_path) for unclean
        cleaned_observations_or_paths: Either list of Observation objects (if is_patched_observations=True)
                                      or list of (trajectory_path, masking_path, problem_pddl_path) tuples
        domain_ref_path: Path to domain file
        is_patched_observations: True if cleaned_observations_or_paths contains Observation objects
        
    Returns:
        True if all trajectories are equal, False if different, None if check cannot be performed
    """
    if is_patched_observations:
        if cleaned_observations_or_paths is None or len(cleaned_observations_or_paths) != len(prepared_trajectories):
            return None
    else:
        if not cleaned_observations_or_paths or len(cleaned_observations_or_paths) != len(prepared_trajectories):
            return None
    
    domain = DomainParser(domain_ref_path).parse_domain()
    parser = TrajectoryParser(domain)
    
    for idx, (traj_path, *_) in enumerate(prepared_trajectories):
        if is_patched_observations:
            original_obs = parser.parse_trajectory(traj_path)
            fully_grounded_original_obs = ground_observation_completely(domain, original_obs)
            if not observations_equal(fully_grounded_original_obs, cleaned_observations_or_paths[idx]):
                return False
        else:
            cleaned_traj_path, _, _ = cleaned_observations_or_paths[idx]
            unclean_obs = parser.parse_trajectory(traj_path)
            fully_grounded_unclean_obs = ground_observation_completely(domain, unclean_obs)
            cleaned_obs = parser.parse_trajectory(cleaned_traj_path)
            fully_grounded_cleaned_obs = ground_observation_completely(domain, cleaned_obs)
            if not observations_equal(fully_grounded_unclean_obs, fully_grounded_cleaned_obs):
                return False
    return True


def _run_baselines_phase(
    baselines,
    mode: str,
    phase: str,
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
    profiler,
    test_states_path_str: str,
    save_learning_metrics_func,
) -> List[dict]:
    """Run all applicable baseline runners for a given phase and return result dicts."""
    results = []
    if not baselines:
        return results

    for runner in baselines:
        if not runner.supports_mode(mode):
            continue

        algo_name = runner.name
        workspace_label = f"{algo_name.lower()}_{phase}"
        print(f"  [{phase.upper()}] Starting {runner.display_name} learning...")

        with profiler.time_operation(f"learning_{workspace_label}"):
            model, extra_info = runner.learn(
                mode=mode,
                domain_path=domain_ref_path,
                prepared_trajectories=trajectories,
                work_dir=fold_work_dir,
                timeout_seconds=conflict_search_timeout or 60,
                profiler=profiler,
            )

        # Baselines don't produce our learning_metrics JSON, so pass empty
        baseline_learning_metrics = {}

        print(f"  [{phase.upper()}] Evaluating {runner.display_name} model...")
        with profiler.time_operation(f"metrics_checking_{workspace_label}"):
            result = evaluate_and_build_result(
                model, algo_name, bench_name, fold, num_trajectories, gt_rate,
                test_problem_paths, phase, domain_ref_path, testing_dir,
                evaluate_model_func, null_metrics, fold_work_dir,
                total_transitions=total_transitions,
                total_gt_transitions=total_gt_transitions,
                learning_metrics=baseline_learning_metrics,
                conflict_search_timeout=conflict_search_timeout,
                planning_timeout=planning_timeout,
                profiler=profiler,
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
    mode: str,
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
        mode: 'masked' (PISAM/PO_ROSAME) or 'fullyobs' (SAM/ROSAME)
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
            our algorithm. When None or empty, only SAM/PISAM is run.

    Returns:
        List of result dicts. Always includes unclean SAM/PISAM and cleaned
        SAM/PISAM results. Baseline results are appended when baselines are
        provided and support the current mode.
    """
    if baselines is None:
        baselines = []

    print(f"[PID {os.getpid()}] Fold {fold}, num_trajs={num_trajectories}, gt_rate={gt_rate}%, mode={mode}")
    if baselines:
        baseline_names = [r.display_name for r in baselines if r.supports_mode(mode)]
        print(f"  Baselines: {', '.join(baseline_names) if baseline_names else '(none for this mode)'}")

    # Initialize profiling
    profiler = TimingProfiler()

    # Setup
    fold_work_dir = testing_dir / f"fold{fold}_numtrajs{num_trajectories}_gtrate{gt_rate}"
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
        with profiler.time_operation("data_source_prepare"):
            prepared_trajectories, pre_built_observations, gt_source_indices = \
                data_source.prepare(selected_pool, num_trajectories, gt_rate, fold)

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
        with profiler.time_operation("generate_predictive_power_test_states"):
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

        # Count total transitions and GT transitions for unclean phase
        with profiler.time_operation("count_total_transitions_and_gt"):
            total_transitions_unclean, total_gt_transitions_unclean = count_total_transitions_and_gt(
                prepared_trajectories, domain_ref_path, gt_rate
            )
        print(f"  [STATS] Unclean phase: {total_transitions_unclean} transitions, {total_gt_transitions_unclean} GT states")

        # Common kwargs for _run_baselines_phase
        baseline_common = dict(
            baselines=baselines,
            mode=mode,
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
            profiler=profiler,
            test_states_path_str=test_states_path_str,
            save_learning_metrics_func=save_learning_metrics_func,
        )

        # ==================================================
        # PHASE 1: UNCLEAN (learning on prepared trajectories)
        # ==================================================
        print(f"  [PHASE 1] Learning on unclean trajectories...")

        # Learn SAM/PISAM
        print(f"  [PHASE 1] Starting SAM/PISAM learning...")
        sam_algo_name = 'PISAM' if mode == 'masked' else 'SAM'
        with profiler.time_operation(f"learning_sam_pisam_unclean_{sam_algo_name}"):
            sam_unclean_model, sam_report, sam_algo_name, _ = learn_sam_pisam(
                mode, domain_ref_path, prepared_trajectories, testing_dir,
                is_denoising=False,
                conflict_search_timeout=conflict_search_timeout,
                profiler=profiler,
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
            )
        print(f"  [PHASE 1] SAM/PISAM learning done, saving metrics...")
        save_learning_metrics_func(fold_work_dir, sam_report)
        
        sam_learning_metrics = load_learning_metrics(fold_work_dir, 'unclean', sam_algo_name)
        
        print(f"  [PHASE 1] Evaluating SAM/PISAM model...")
        with profiler.time_operation("metrics_checking_sam_pisam_unclean"):
            unclean_sam_result = evaluate_and_build_result(
                sam_unclean_model, sam_algo_name, bench_name, fold, num_trajectories, gt_rate,
                test_problem_paths, 'unclean', domain_ref_path, testing_dir,
                evaluate_model_func, null_metrics, fold_work_dir,
                total_transitions=total_transitions_unclean,
                total_gt_transitions=total_gt_transitions_unclean,
                learning_metrics=sam_learning_metrics,
                conflict_search_timeout=conflict_search_timeout,
                planning_timeout=planning_timeout,
                profiler=profiler,
                test_states_path=test_states_path_str,
            )

        # Run baselines on unclean trajectories
        unclean_baseline_results = _run_baselines_phase(
            phase='unclean',
            trajectories=prepared_trajectories,
            total_transitions=total_transitions_unclean,
            total_gt_transitions=total_gt_transitions_unclean,
            **baseline_common,
        )

        # ==================================================
        # PHASE 2: CLEANED (denoising with NOISY_PISAM/NOISY_SAM)
        # ==================================================
        print(f"  [PHASE 2] Denoising and re-learning...")
        
        total_transitions_cleaned = total_transitions_unclean
        total_gt_transitions_cleaned = total_gt_transitions_unclean

        cleaned_equals_unclean_pisam = None

        try:
            print(f"  [PHASE 2] Starting denoising (NOISY_SAM/NOISY_PISAM)...")
            if conflict_search_timeout is not None:
                print(f"  [PHASE 2] Using conflict search timeout: {conflict_search_timeout}s")
            denoiser_algo_name = 'NOISY_PISAM' if mode == 'masked' else 'NOISY_SAM'
            with profiler.time_operation(f"learning_sam_pisam_cleaned_{denoiser_algo_name}"):
                cleaned_model, denoising_report, denoiser_algo_name, patched_observations = learn_sam_pisam(
                    mode, domain_ref_path, prepared_trajectories, testing_dir,
                    is_denoising=True,
                    conflict_search_timeout=conflict_search_timeout,
                    profiler=profiler,
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
            print(f"  [PHASE 2] Denoising complete, saving metrics...")
            save_learning_metrics_func(fold_work_dir, denoising_report)
            
            denoising_learning_metrics = load_learning_metrics(fold_work_dir, 'cleaned', denoiser_algo_name)
            
            total_transitions_cleaned = total_transitions_unclean
            total_gt_transitions_cleaned = total_gt_transitions_unclean

            # Evaluate cleaned SAM/PISAM model
            print(f"  [PHASE 2] Evaluating denoised model...")
            with profiler.time_operation("metrics_checking_sam_pisam_cleaned"):
                cleaned_sam_result = evaluate_and_build_result(
                    cleaned_model, denoiser_algo_name, bench_name, fold, num_trajectories, gt_rate,
                    test_problem_paths, 'cleaned', domain_ref_path, testing_dir,
                    evaluate_model_func, null_metrics, fold_work_dir,
                    total_transitions=total_transitions_cleaned,
                    total_gt_transitions=total_gt_transitions_cleaned,
                    learning_metrics=denoising_learning_metrics,
                    conflict_search_timeout=conflict_search_timeout,
                    planning_timeout=planning_timeout,
                    profiler=profiler,
                    test_states_path=test_states_path_str,
                )

            # Check if cleaned and unclean trajectories are the same
            cleaned_equals_unclean_pisam = check_trajectories_equal(
                prepared_trajectories, patched_observations, domain_ref_path, is_patched_observations=True
            )
            if cleaned_equals_unclean_pisam is not None:
                if cleaned_equals_unclean_pisam:
                    print(f"  [PHASE 2] ⚠️  WARNING: Cleaned and unclean trajectories are EQUAL for {denoiser_algo_name}!")
                else:
                    print(f"  [PHASE 2] ✓ Cleaned and unclean trajectories are DIFFERENT for {denoiser_algo_name}")

            # SAVE patched observations to disk for baselines to use
            cleaned_baseline_results = []
            if patched_observations is not None:
                print(f"  [PHASE 2] Saving {len(patched_observations)} patched observations to disk...")
                final_observations_dir = fold_work_dir / "final_observations"
                save_patched_observations(
                    patched_observations, prepared_trajectories, final_observations_dir, domain_ref_path
                )
                print(f"  [PHASE 2] Patched observations saved")
                run_post_process_gt_metrics(fold_work_dir, prepared_trajectories, domain_ref_path, gt_rate)

                # Evaluate ALL conflict-free models (multi-solution evaluation)
                conflict_free_models_dir = fold_work_dir / "conflict_free_models"
                if conflict_free_models_dir.exists():
                    print(f"  [MULTI-EVAL] Evaluating all conflict-free models...")
                    with profiler.time_operation("evaluate_all_solutions"):
                        all_solutions_results = evaluate_all_solutions(
                            conflict_free_models_dir=conflict_free_models_dir,
                            ref_domain_path=domain_ref_path,
                            test_problem_paths=test_problem_paths,
                            test_states_path=test_states_path,
                            planning_timeout=planning_timeout,
                            output_dir=fold_work_dir,
                        )
                    print(f"  [MULTI-EVAL] Evaluated {len(all_solutions_results)} solutions")

                    with profiler.time_operation("build_correlation_table"):
                        build_correlation_table(fold_work_dir)

                # Run baselines on cleaned trajectories
                if baselines and final_observations_dir.exists():
                    cleaned_trajectories = convert_cleaned_dir_to_trajectory_list(
                        final_observations_dir, prepared_trajectories
                    )
                    if cleaned_trajectories:
                        # Check equality for metadata
                        cleaned_equals_unclean_baselines = check_trajectories_equal(
                            prepared_trajectories, cleaned_trajectories, domain_ref_path, is_patched_observations=False
                        )
                        update_fold_metadata(
                            fold_work_dir,
                            cleaned_equals_unclean_pisam=cleaned_equals_unclean_pisam,
                            cleaned_equals_unclean_rosame=cleaned_equals_unclean_baselines,
                        )

                        with profiler.time_operation("count_total_transitions_and_gt_cleaned"):
                            total_transitions_cleaned_bl, total_gt_transitions_cleaned_bl = count_total_transitions_and_gt(
                                cleaned_trajectories, domain_ref_path, gt_rate
                            )

                        cleaned_baseline_results = _run_baselines_phase(
                            phase='cleaned',
                            trajectories=cleaned_trajectories,
                            total_transitions=total_transitions_cleaned_bl,
                            total_gt_transitions=total_gt_transitions_cleaned_bl,
                            **baseline_common,
                        )

        except Exception as e:
            print(f"  ERROR in denoising phase: {e}")
            import traceback
            traceback.print_exc()

            cleaned_sam_result = evaluate_and_build_result(
                None, sam_algo_name, bench_name, fold, num_trajectories, gt_rate,
                test_problem_paths, 'cleaned', domain_ref_path, testing_dir,
                evaluate_model_func, null_metrics, fold_work_dir,
                total_transitions=total_transitions_cleaned,
                total_gt_transitions=total_gt_transitions_cleaned,
                learning_metrics={},
                conflict_search_timeout=conflict_search_timeout,
                planning_timeout=planning_timeout,
                profiler=profiler,
                test_states_path=test_states_path_str,
            )
            cleaned_baseline_results = []

        # Update fold metadata (if not already updated)
        metadata_path = fold_work_dir / "fold_info.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                existing_metadata = json.load(f)
            if 'cleaned_equals_unclean_pisam' not in existing_metadata:
                update_fold_metadata(
                    fold_work_dir,
                    cleaned_equals_unclean_pisam=cleaned_equals_unclean_pisam if 'cleaned_equals_unclean_pisam' in locals() else None,
                )
        
        # Save detailed timing report
        timing_report_path = fold_work_dir / "timing_report.json"
        profiler.save_report(timing_report_path)
        print(f"  [FOLD COMPLETE] Timing report saved to {timing_report_path.name}")
        
        timing_plot_path = fold_work_dir / "timing_report.png"
        profiler.plot_timing_report(timing_plot_path)
        
        # Build results list: always SAM results first, then baselines
        fold_results = [unclean_sam_result] + unclean_baseline_results + [cleaned_sam_result] + cleaned_baseline_results
        print(f"  [FOLD COMPLETE] Returning {len(fold_results)} results for fold {fold}")
        return fold_results

    finally:
        os.chdir(original_cwd)
