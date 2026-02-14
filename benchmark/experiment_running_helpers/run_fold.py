"""
Helper functions for running AMLGym experiments.

This module contains the main fold execution logic for the amlgym_testing experiments.
"""

import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import List, Tuple, Dict

from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser

from benchmark.experiment_running_helpers.cleaned_trajectories import convert_cleaned_dir_to_trajectory_list, save_patched_observations
from benchmark.experiment_running_helpers.learning_helpers import learn_rosame, learn_sam_pisam
from benchmark.experiment_running_helpers.profiling import TimingProfiler
from benchmark.experiment_running_helpers.result_builders import evaluate_and_build_result
from benchmark.experiment_running_helpers.statistics import count_total_transitions_and_gt, load_learning_metrics
from benchmark.experiment_running_helpers.trajectory_utils import prepare_fold_trajectories, save_fold_metadata, update_fold_metadata
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
    
    for idx, (traj_path, _, _) in enumerate(prepared_trajectories):
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
    evaluate_model_func,
    save_learning_metrics_func,
    conflict_search_timeout: int = None,
    planning_timeout: int = 60
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
        evaluate_model_func: Function to evaluate a learned model
        save_learning_metrics_func: Function to save learning metrics
        conflict_search_timeout: Optional timeout in seconds for conflict search (cleaning phase)
        planning_timeout: Timeout in seconds for planning during evaluation (default: 60)

    Returns:
        List of 4 dicts with results for: unclean SAM/PISAM, unclean ROSAME, cleaned SAM/PISAM, cleaned ROSAME
    """
    print(f"[PID {os.getpid()}] Fold {fold}, num_trajs={num_trajectories}, gt_rate={gt_rate}%, mode={mode}")

    # Initialize profiling
    profiler = TimingProfiler()

    # Setup
    fold_work_dir = testing_dir / f"fold{fold}_numtrajs{num_trajectories}_gtrate{gt_rate}"
    fold_work_dir.mkdir(parents=True, exist_ok=True)
    original_cwd = os.getcwd()
    os.chdir(fold_work_dir)

    # Copy domain reference file to fold directory to avoid race conditions
    # Each fold gets its own copy so SimpleDomainReader doesn't conflict
    local_domain_ref = fold_work_dir / "domain_reference.pddl"
    shutil.copy2(domain_ref_path, local_domain_ref)
    domain_ref_path = local_domain_ref  # Use local copy for all evaluations

    null_metrics = {k: None for k in ['precision_precs_pos', 'precision_precs_neg',
                    'precision_eff_pos', 'precision_eff_neg', 'precision_overall',
                    'recall_precs_pos', 'recall_precs_neg', 'recall_eff_pos',
                    'recall_eff_neg', 'recall_overall', 'solving_ratio',
                    'false_plans_ratio', 'unsolvable_ratio', 'planning_timed_out_ratio']}

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

        # Pool size = 0.8 * total problems (same as train size), so we can use up to that many trajectories
        random.seed(42 + fold)
        num_trajectories_pool = n_train  # 0.8 * n_problems
        selected_pool = random.sample(train_problem_dirs, min(num_trajectories_pool, len(train_problem_dirs)))

        # Load pre-generated trajectories
        print(f"  Loading {num_trajectories} pre-generated trajectories with gt_rate={gt_rate}%...")
        with profiler.time_operation("prepare_fold_trajectories"):
            prepared_trajectories = prepare_fold_trajectories(
                selected_pool, num_trajectories, gt_rate
            )

        if not prepared_trajectories:
            print(f"  ERROR: No trajectories prepared for fold {fold}")
            return []

        print(f"  ✓ Prepared {len(prepared_trajectories)} trajectories")

        # Build test problem paths - use same naming convention as prepare_fold_trajectories
        test_problem_paths = []
        for d in test_problem_dirs:
            # Use consistent naming: {problem_dir_name}.pddl
            problem_pddl_path = d / f"{d.name}.pddl"
            if problem_pddl_path.exists():
                test_problem_paths.append(str(problem_pddl_path))
            else:
                # Fallback: try glob if standard naming not found
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

        # Save fold metadata with test problem names
        save_fold_metadata(fold_work_dir, prepared_trajectories, fold, num_trajectories, gt_rate, test_problem_paths)

        # Count total transitions and GT transitions for unclean phase
        with profiler.time_operation("count_total_transitions_and_gt"):
            total_transitions_unclean, total_gt_transitions_unclean = count_total_transitions_and_gt(
                prepared_trajectories, domain_ref_path, gt_rate
            )
        print(f"  [STATS] Unclean phase: {total_transitions_unclean} transitions, {total_gt_transitions_unclean} GT states")

        # ==================================================
        # PHASE 1: UNCLEAN (learning on prepared trajectories)
        # ==================================================
        print(f"  [PHASE 1] Learning on unclean trajectories...")

        # Learn SAM/PISAM
        print(f"  [PHASE 1] Starting SAM/PISAM learning...")
        sam_algo_name = 'PISAM' if mode == 'masked' else 'SAM'
        with profiler.time_operation(f"learning_sam_pisam_unclean_{sam_algo_name}"):
            sam_unclean_model, sam_report, sam_algo_name, _ = learn_sam_pisam(
                mode, domain_ref_path, prepared_trajectories, testing_dir, is_denoising=False, conflict_search_timeout=conflict_search_timeout, profiler=profiler, fold_work_dir=fold_work_dir
            )
        print(f"  [PHASE 1] SAM/PISAM learning done, saving metrics...")
        save_learning_metrics_func(fold_work_dir, sam_report)
        
        # Load learning metrics for SAM/PISAM
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
                profiler=profiler
            )

        # Learn ROSAME
        print(f"  [PHASE 1] Starting ROSAME learning...")
        rosame_algo_name = 'PO_ROSAME' if mode == 'masked' else 'ROSAME'
        with profiler.time_operation(f"learning_rosame_unclean_{rosame_algo_name}"):
            rosame_unclean_model, rosame_report, rosame_algo_name = learn_rosame(
                mode, domain_ref_path, prepared_trajectories, testing_dir, "rosame_unclean", profiler=profiler
            )
        print(f"  [PHASE 1] ROSAME learning done, saving metrics...")
        save_learning_metrics_func(fold_work_dir, rosame_report)
        
        # Load learning metrics for ROSAME (usually empty, but check anyway)
        rosame_learning_metrics = load_learning_metrics(fold_work_dir, 'unclean', rosame_algo_name)
        
        print(f"  [PHASE 1] Evaluating ROSAME model...")
        with profiler.time_operation("metrics_checking_rosame_unclean"):
            unclean_rosame_result = evaluate_and_build_result(
                rosame_unclean_model, rosame_algo_name, bench_name, fold, num_trajectories, gt_rate,
                test_problem_paths, 'unclean', domain_ref_path, testing_dir,
                evaluate_model_func, null_metrics, fold_work_dir,
                total_transitions=total_transitions_unclean,
                total_gt_transitions=total_gt_transitions_unclean,
                learning_metrics=rosame_learning_metrics,
                conflict_search_timeout=conflict_search_timeout,
                planning_timeout=planning_timeout,
                profiler=profiler
            )

        # ==================================================
        # PHASE 2: CLEANED (denoising with NOISY_PISAM/NOISY_SAM)
        # ==================================================
        print(f"  [PHASE 2] Denoising and re-learning...")
        
        # Initialize comparison flags
        cleaned_equals_unclean_pisam = None
        cleaned_equals_unclean_rosame = None

        try:
            # Learn with denoiser (NOISY_PISAM/NOISY_SAM) - returns patched observations!
            print(f"  [PHASE 2] Starting denoising (NOISY_SAM/NOISY_PISAM)...")
            if conflict_search_timeout is not None:
                print(f"  [PHASE 2] Using conflict search timeout: {conflict_search_timeout}s")
            denoiser_algo_name = 'NOISY_PISAM' if mode == 'masked' else 'NOISY_SAM'
            with profiler.time_operation(f"learning_sam_pisam_cleaned_{denoiser_algo_name}"):
                cleaned_model, denoising_report, denoiser_algo_name, patched_observations = learn_sam_pisam(
                    mode, domain_ref_path, prepared_trajectories, testing_dir, is_denoising=True,
                    conflict_search_timeout=conflict_search_timeout, profiler=profiler, fold_work_dir=fold_work_dir
                )
            print(f"  [PHASE 2] Denoising complete, saving metrics...")
            save_learning_metrics_func(fold_work_dir, denoising_report)
            
            # Load learning metrics for denoising (cleaning phase)
            denoising_learning_metrics = load_learning_metrics(fold_work_dir, 'cleaned', denoiser_algo_name)
            
            # For cleaned phase, we still use the same trajectories (before cleaning)
            # So transitions count is the same, but we track cleaning metrics separately
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
                    profiler=profiler
                )

            # Check if cleaned and unclean trajectories are the same for PISAM/SAM
            cleaned_equals_unclean_pisam = check_trajectories_equal(
                prepared_trajectories, patched_observations, domain_ref_path, is_patched_observations=True
            )
            if cleaned_equals_unclean_pisam is not None:
                if cleaned_equals_unclean_pisam:
                    print(f"  [PHASE 2] ⚠️  WARNING: Cleaned and unclean trajectories are EQUAL for {denoiser_algo_name}!")
                    print(f"  [PHASE 2] ⚠️  This means the denoiser did not modify the trajectories.")
                else:
                    print(f"  [PHASE 2] ✓ Cleaned and unclean trajectories are DIFFERENT for {denoiser_algo_name}")

            # SAVE patched observations to disk for ROSAME to use
            if patched_observations is not None:
                print(f"  [PHASE 2] Saving {len(patched_observations)} patched observations to disk...")
                final_observations_dir = fold_work_dir / "final_observations"
                save_patched_observations(
                    patched_observations, prepared_trajectories, final_observations_dir, domain_ref_path
                )
                print(f"  [PHASE 2] Patched observations saved")

            # Re-learn ROSAME on PATCHED OBSERVATIONS from denoiser
            if patched_observations is not None:
                print(f"  [PHASE 2] Learning ROSAME on patched observations from denoiser...")
                final_observations_dir = fold_work_dir / "final_observations"

                if final_observations_dir.exists():
                    print(f"  [PHASE 2] Converting patched observations to trajectory list...")
                    # Convert patched observations (saved to disk) to trajectory list format
                    # Returns: List[Tuple[traj_path, masking_path, problem_pddl_path]]
                    cleaned_trajectories = convert_cleaned_dir_to_trajectory_list(
                        final_observations_dir, prepared_trajectories
                    )
                    print(f"  [PHASE 2] Converted {len(cleaned_trajectories)} cleaned trajectories")

                    if cleaned_trajectories:
                        # Check if cleaned and unclean trajectories are the same for ROSAME
                        cleaned_equals_unclean_rosame = check_trajectories_equal(
                            prepared_trajectories, cleaned_trajectories, domain_ref_path, is_patched_observations=False
                        )
                        if cleaned_equals_unclean_rosame is not None:
                            if cleaned_equals_unclean_rosame:
                                print(f"  [PHASE 2] ⚠️  WARNING: Cleaned and unclean trajectories are EQUAL for {rosame_algo_name}!")
                                print(f"  [PHASE 2] ⚠️  This means the denoiser did not modify the trajectories.")
                            else:
                                print(f"  [PHASE 2] ✓ Cleaned and unclean trajectories are DIFFERENT for {rosame_algo_name}")
                        
                        # Update fold metadata with comparison results
                        update_fold_metadata(
                            fold_work_dir,
                            cleaned_equals_unclean_pisam=cleaned_equals_unclean_pisam,
                            cleaned_equals_unclean_rosame=cleaned_equals_unclean_rosame
                        )
                        
                        # Count transitions for cleaned trajectories (used for ROSAME learning)
                        # Note: cleaned trajectories still use the same gt_rate as they were created from the same original trajectories
                        with profiler.time_operation("count_total_transitions_and_gt_cleaned"):
                            total_transitions_cleaned_rosame, total_gt_transitions_cleaned_rosame = count_total_transitions_and_gt(
                                cleaned_trajectories, domain_ref_path, gt_rate
                            )
                        print(f"  [STATS] Cleaned ROSAME phase: {total_transitions_cleaned_rosame} transitions, {total_gt_transitions_cleaned_rosame} GT states")
                        
                        # Learn ROSAME on cleaned trajectories from denoiser
                        print(f"  [PHASE 2] Starting ROSAME learning on cleaned trajectories...")
                        with profiler.time_operation(f"learning_rosame_cleaned_{rosame_algo_name}"):
                            rosame_cleaned_model, rosame_cleaned_report, rosame_cleaned_algo_name = learn_rosame(
                                mode, domain_ref_path, cleaned_trajectories, testing_dir, "rosame_cleaned", profiler=profiler
                            )
                        print(f"  [PHASE 2] Cleaned ROSAME learning done...")
                        
                        # Load learning metrics for cleaned ROSAME
                        rosame_cleaned_learning_metrics = load_learning_metrics(fold_work_dir, 'cleaned', rosame_cleaned_algo_name)

                        print(f"  [PHASE 2] Evaluating cleaned ROSAME model...")
                        with profiler.time_operation("metrics_checking_rosame_cleaned"):
                            cleaned_rosame_result = evaluate_and_build_result(
                                rosame_cleaned_model, rosame_cleaned_algo_name, bench_name, fold,
                                num_trajectories, gt_rate, test_problem_paths, 'cleaned',
                                domain_ref_path, testing_dir, evaluate_model_func, null_metrics, fold_work_dir,
                                total_transitions=total_transitions_cleaned_rosame,
                                total_gt_transitions=total_gt_transitions_cleaned_rosame,
                                learning_metrics=rosame_cleaned_learning_metrics,
                                conflict_search_timeout=conflict_search_timeout,
                                planning_timeout=planning_timeout,
                                profiler=profiler
                            )
                    else:
                        print(f"  Warning: No cleaned trajectories found in {final_observations_dir}")
                        cleaned_rosame_result = evaluate_and_build_result(
                            None, rosame_algo_name, bench_name, fold,
                            num_trajectories, gt_rate, test_problem_paths, 'cleaned',
                            domain_ref_path, testing_dir, evaluate_model_func, null_metrics, fold_work_dir,
                            total_transitions=total_transitions_cleaned,
                            total_gt_transitions=total_gt_transitions_cleaned,
                            learning_metrics={},
                            conflict_search_timeout=conflict_search_timeout,
                            planning_timeout=planning_timeout,
                            profiler=profiler
                        )
                else:
                    print(f"  Warning: final_observations directory does not exist")
                    cleaned_rosame_result = evaluate_and_build_result(
                        None, rosame_algo_name, bench_name, fold,
                        num_trajectories, gt_rate, test_problem_paths, 'cleaned',
                        domain_ref_path, testing_dir, evaluate_model_func, null_metrics, fold_work_dir,
                        total_transitions=total_transitions_cleaned,
                        total_gt_transitions=total_gt_transitions_cleaned,
                        learning_metrics={},
                        conflict_search_timeout=conflict_search_timeout,
                        planning_timeout=planning_timeout,
                        profiler=profiler
                    )
            else:
                print(f"  Warning: No patched observations returned from denoiser")
                cleaned_rosame_result = evaluate_and_build_result(
                    None, rosame_algo_name, bench_name, fold,
                    num_trajectories, gt_rate, test_problem_paths, 'cleaned',
                    domain_ref_path, testing_dir, evaluate_model_func, null_metrics, fold_work_dir,
                    total_transitions=total_transitions_cleaned,
                    total_gt_transitions=total_gt_transitions_cleaned,
                    learning_metrics={},
                    conflict_search_timeout=conflict_search_timeout,
                    planning_timeout=planning_timeout,
                    profiler=profiler
                )

        except Exception as e:
            print(f"  ERROR in denoising phase: {e}")
            import traceback
            traceback.print_exc()

            # Return null results on failure - using uniform interface
            cleaned_sam_result = evaluate_and_build_result(
                None, sam_algo_name, bench_name, fold, num_trajectories, gt_rate,
                test_problem_paths, 'cleaned', domain_ref_path, testing_dir,
                evaluate_model_func, null_metrics, fold_work_dir,
                total_transitions=total_transitions_cleaned,
                total_gt_transitions=total_gt_transitions_cleaned,
                learning_metrics={},
                conflict_search_timeout=conflict_search_timeout,
                planning_timeout=planning_timeout,
                profiler=profiler
            )
            cleaned_rosame_result = evaluate_and_build_result(
                None, rosame_algo_name, bench_name, fold, num_trajectories, gt_rate,
                test_problem_paths, 'cleaned', domain_ref_path, testing_dir,
                evaluate_model_func, null_metrics, fold_work_dir,
                total_transitions=total_transitions_cleaned,
                total_gt_transitions=total_gt_transitions_cleaned,
                learning_metrics={},
                conflict_search_timeout=conflict_search_timeout,
                planning_timeout=planning_timeout,
                profiler=profiler
            )

        # Update fold metadata with comparison results (if not already updated)
        # This handles the case where cleaned phase failed or had no patched observations
        metadata_path = fold_work_dir / "fold_info.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                existing_metadata = json.load(f)
            if 'cleaned_equals_unclean_pisam' not in existing_metadata or 'cleaned_equals_unclean_rosame' not in existing_metadata:
                # Only update if we have values that weren't already set
                update_fold_metadata(
                    fold_work_dir,
                    cleaned_equals_unclean_pisam=cleaned_equals_unclean_pisam if 'cleaned_equals_unclean_pisam' in locals() else None,
                    cleaned_equals_unclean_rosame=cleaned_equals_unclean_rosame if 'cleaned_equals_unclean_rosame' in locals() else None
                )
        
        # Save detailed timing report
        timing_report_path = fold_work_dir / "timing_report.json"
        profiler.save_report(timing_report_path)
        print(f"  [FOLD COMPLETE] Timing report saved to {timing_report_path.name}")
        
        # Generate timing visualization plot
        timing_plot_path = fold_work_dir / "timing_report.png"
        profiler.plot_timing_report(timing_plot_path)
        
        print(f"  [FOLD COMPLETE] Returning 4 results for fold {fold}")
        return [unclean_sam_result, unclean_rosame_result, cleaned_sam_result, cleaned_rosame_result]

    finally:
        os.chdir(original_cwd)
