import argparse
import json
import time
import random
from concurrent.futures import as_completed, ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import List, Dict

import pandas as pd
from amlgym.metrics import print_metrics, syntactic_precision, syntactic_recall, problem_solving

from benchmark.evaluation.predictive_metrics import evaluate_predictive_power
from benchmark.evaluation.correlation_analysis import aggregate_correlation_tables

from benchmark.experiment_running_helpers.run_fold import run_single_fold
from benchmark.experiment_running_helpers.trajectory_utils import pregenerate_all_gt_frame_axiom_files
from benchmark.experiment_running_helpers.reporting import (
    generate_excel_report,
    generate_plots,
    generate_gt_injection_plots,
    plot_stacked_solving_rate,
)

# =============================================================================
# CONFIG & PATHS
# =============================================================================

benchmark_path = Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/benchmark")
project_root = benchmark_path.parent
print_metrics()

# Global lock for thread-safe evaluation (AMLGym SimpleDomainReader is not thread-safe)
evaluation_lock = Lock()

experiment_data_dirs_masked = {
    # "blocksworld": ["multi_problem_04-12-2025T12:00:44__model=gpt-5.1__steps=50__planner"],
    "blocksworld": ["multi_problem_02-01-2026T14:16:59__model=gpt-5.1__steps=300__planner"],
    # "hanoi": ["multi_problem_06-12-2025T13:58:24__model=gpt-5.1__steps=100__planner"],
    "hanoi": ["multi_problem_02-01-2026T14:26:29__model=gpt-5.1__steps=300__planner"],
    # "hanoi": ["multi_problem_13-12-2025T14:53:55__model=gemini-2.5-pro__steps=11__planner"],
    # "n_puzzle_typed": ["multi_problem_06-12-2025T13:32:59__model=gpt-5.1__steps=100__planner"],
    "n_puzzle_typed": ["multi_problem_02-01-2026T16:55:49__model=gpt-5.1__steps=300__planner"],
    "maze": ["multi_problem_09-01-2026T15:28:23__model=gpt-5.1__steps=300__planner"],
    # "maze": ["experiment_07-12-2025T16:16:54__model=gpt-5.1__steps=100__planner"],
    # "maze": ["multi_problem_13-12-2025T18:10:23__model=gemini-2.5-pro__steps=100__planner"]
}

experiment_data_dirs_fullyobs = {
    "blocksworld": ["multi_problem_07-12-2025T17:27:33__model=gpt-5.1__steps=100__planner__NO_MASK"],
    "hanoi": ["multi_problem_07-12-2025T17:30:57__model=gpt-5.1__steps=100__planner__NO_MASK"],
    "n_puzzle_typed": ["multi_problem_06-12-2025T13:32:59__model=gpt-5.1__steps=100__planner"],
    "maze": ["multi_problem_07-12-2025T17:37:10__model=gpt-5.1__steps=100__planner__NO_MASK"]
}

domain_name_mappings = {
    'blocksworld': 'blocksworld',
    'hanoi': 'hanoi',
    'n_puzzle_typed': 'npuzzle',
    'maze': 'maze',
}

domain_properties = {
    'blocksworld': {
        "domain_path": benchmark_path / 'domains' / 'blocksworld' / 'blocksworld.pddl',
    },
    'hanoi': {
        "domain_path": benchmark_path / 'domains' / 'hanoi' / 'hanoi.pddl',
    },
    'n_puzzle_typed': {
        "domain_path": benchmark_path / 'domains' / 'n_puzzle' / 'n_puzzle.pddl',
    },
    'maze': {
        "domain_path": benchmark_path / 'domains' / 'maze' / 'maze.pddl',
    },
}

N_FOLDS = 5
# NUM_TRAJECTORIES_LIST = [1, 2, 3, 4, 5, 6, 7, 8]  # Number of full trajectories to use for learning
NUM_TRAJECTORIES_LIST = [3, 4, 5, 6, 7, 8]  # Number of full trajectories to use for learning

# Pool size per fold = 0.8 * n_problems (computed in run_fold). With 10 problems, pool=8.
NUM_TRAJECTORIES_POOL = 8  # Typical value (0.8*10); actual pool is 0.8*n_problems in run_fold
# GT_RATE_PERCENTAGES = [0, 10, 25, 50, 75, 100]  # Percentage of states to inject as GT (0 = only initial state)
GT_RATE_PERCENTAGES = [0]  # Percentage of states to inject as GT (0 = only initial state)
FRAME_AXIOM_MODE = "after_gt_only"  # "after_gt_only" or "all_states"
PLANNING_TIMEOUT = 60  # Timeout in seconds for planning during evaluation
FLUENT_PATCH_COST = 1.0
FLUENT_PATCH_WEIGHT = 1.0
MODEL_PATCH_COST = 1.0
MODEL_CONSTRAINT_WEIGHT = 0.0
MAX_SEARCH_NODES = None

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def save_learning_metrics(output_dir: Path, report: dict, trajectory_mapping: Dict[str, str] = None) -> dict:
    """Save learning metrics to JSON file."""
    metrics = {
        "learning_time_seconds": report.get("total_time_seconds", None),
        "max_depth": report.get("max_depth", None),
        "nodes_expanded": report.get("nodes_expanded", None),
        "terminated_by": report.get("terminated_by", None),
        "conflict_free_model_count": report.get("conflict_free_model_count", None),
        "actual_timeout_seconds": report.get("actual_timeout_seconds", None),  # Actual timeout used (includes defaults)
        "fluent_patch_cost": report.get("fluent_patch_cost", None),
        "fluent_patch_weight": report.get("fluent_patch_weight", None),
        "model_patch_cost": report.get("model_patch_cost", None),
        "model_constraint_weight": report.get("model_constraint_weight", None),
        "max_search_nodes": report.get("max_search_nodes", None),
        "search_mode": report.get("search_mode", None),
        "node_choosing_strategy": report.get("node_choosing_strategy", None),
        "conflict_group_strategy": report.get("conflict_group_strategy", None),
        "fluent_branch_mode": report.get("fluent_branch_mode", None),
        # Detailed patch info for the final/best model
        "best_model_constraints": report.get("final_model_constraints", None),
        "best_fluent_patches": report.get("final_fluent_patches", None),
        # Summary of all conflict-free solutions found during search
        "conflict_free_solutions_summary": report.get("conflict_free_solutions_summary", None),
    }

    # Add trajectory mapping if provided
    if trajectory_mapping:
        metrics["trajectory_mapping"] = trajectory_mapping

    with open(output_dir / "learning_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics




def evaluate_model(model_path: str, domain_ref_path: Path, test_problems: List[str],
                   planning_timeout: int = 60, profiler=None, test_states_path: str = None) -> dict:
    """Evaluate a learned model. Handles AMLGym SimpleDomainReader race conditions.

    Args:
        model_path: Path to learned model PDDL file
        domain_ref_path: Path to reference domain PDDL file
        test_problems: List of test problem paths
        planning_timeout: Timeout in seconds for planning during evaluation (default: 60)
        profiler: Optional TimingProfiler instance for detailed timing
        test_states_path: Path to test_states.json for predictive power metrics (optional)

    Returns:
        Dictionary of evaluation metrics
    """
    # NOTE: evaluation_lock is a threading lock, but we use ProcessPoolExecutor,
    # so it doesn't prevent race conditions across processes.

    def _time_metric(metric_name, func):
        """Helper to time a metric computation."""
        if profiler:
            start = time.perf_counter()
            result = func()
            elapsed = time.perf_counter() - start
            profiler.add_detailed_timing(
                'amlgym_metrics',
                metric_name,
                elapsed,
                {'model_path': model_path, 'num_test_problems': len(test_problems)}
            )
            return result
        return func()

    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Add small random delay to reduce collision probability
            if attempt > 0:
                time.sleep(random.uniform(0.1, 0.5))

            # Run all evaluations together - if any fail, retry all
            # Profile each metric computation separately
            precision = _time_metric('syntactic_precision',
                lambda: syntactic_precision(model_path, str(domain_ref_path)))
            recall = _time_metric('syntactic_recall',
                lambda: syntactic_recall(model_path, str(domain_ref_path)))
            problem_solving_result = _time_metric('problem_solving',
                lambda: problem_solving(model_path, str(domain_ref_path), test_problems, timeout=planning_timeout))

            # Success - break out of retry loop
            break

        except (FileNotFoundError, ValueError, IndexError) as e:
            # Race condition with SimpleDomainReader
            if attempt < max_retries - 1:
                # Clean up potentially corrupted _clean file
                clean_file = f"{domain_ref_path}_clean"
                try:
                    if Path(clean_file).exists():
                        Path(clean_file).unlink()
                except:
                    pass
                continue
            else:
                # Final attempt failed - return null metrics
                print(f"Warning: Evaluation failed after {max_retries} attempts: {e}")
                precision = None
                recall = None
                problem_solving_result = None

    # Predictive power metrics (applicability + predicted effects)
    if test_states_path:
        predictive = evaluate_predictive_power(
            model_path, str(domain_ref_path), test_states_path, test_problems,
        )
    else:
        predictive = {
            "pred_app_precision": None,
            "pred_app_recall": None,
            "pred_eff_precision": None,
            "pred_eff_recall": None,
        }

    return {
        'precision_precs_pos': precision.get('precs_pos') if isinstance(precision, dict) else None,
        'precision_precs_neg': precision.get('precs_neg') if isinstance(precision, dict) else None,
        'precision_eff_pos': precision.get('eff_pos') if isinstance(precision, dict) else None,
        'precision_eff_neg': precision.get('eff_neg') if isinstance(precision, dict) else None,
        'precision_overall': precision.get('mean') if isinstance(precision, dict) else precision,
        'recall_precs_pos': recall.get('precs_pos') if isinstance(recall, dict) else None,
        'recall_precs_neg': recall.get('precs_neg') if isinstance(recall, dict) else None,
        'recall_eff_pos': recall.get('eff_pos') if isinstance(recall, dict) else None,
        'recall_eff_neg': recall.get('eff_neg') if isinstance(recall, dict) else None,
        'recall_overall': recall.get('mean') if isinstance(recall, dict) else recall,
        'solving_ratio': problem_solving_result.get('solving_ratio') if isinstance(problem_solving_result, dict) else None,
        'false_plans_ratio': problem_solving_result.get('false_plans_ratio') if isinstance(problem_solving_result, dict) else None,
        'unsolvable_ratio': problem_solving_result.get('unsolvable_ratio') if isinstance(problem_solving_result, dict) else None,
        'planning_timed_out_ratio': problem_solving_result.get('timed_out') if isinstance(problem_solving_result, dict) else None,
        **predictive,
    }



# =============================================================================
# MAIN EXPERIMENT LOOP
# =============================================================================
def main(
    selected_domains: List[str] = None,
    mode: str = 'masked',
    learning_timeout_seconds: int = 180,
    planning_timeout_seconds: int = 60,
    fluent_patch_cost: float = 1.0,
    fluent_patch_weight: float = 1.0,
    model_patch_cost: float = 1.0,
    model_constraint_weight: float = 0.0,
    max_search_nodes: int = None,
    search_mode: str = "dfs",
    node_choosing_strategy: str = "model_patch_first",
    conflict_group_strategy: str = "most_observations",
    fluent_branch_mode: str = "group",
    experiment_name: str = None,
    # --- Simulated data source ---
    simulated_gt_trajectories: List[str] = None,
    simulated_masking_p: float = 0.4,
    simulated_masking_strategy: str = "percentage",
    simulated_noising_p: float = 0.15,
    simulated_noising_strategy: str = "percentage",
    simulated_seed: int = 42,
    baselines: list = None,
):
    """
    Run benchmark experiments.

    Args:
        selected_domains: List of domain names to run, or None for all domains in domain_name_mappings
        mode: Either 'masked' or 'fullyobs'
        simulated_gt_trajectories: Optional list of GT trajectory file paths (strings).
            When provided, trajectories are loaded from these files and synthetic
            masking + noise is applied in memory instead of reading pre-generated files.
        simulated_masking_p: Masking probability for simulated mode (default: 0.4).
        simulated_noising_p: Noising probability for simulated mode (default: 0.15).
        simulated_seed: Random seed for simulated noise injection (default: 42).
    """
    from src.pi_sam.masking import MaskingType
    from src.pi_sam.noising import NoisingType

    use_simulated = simulated_gt_trajectories is not None
    simulated_gt_paths = None
    if use_simulated:
        simulated_gt_paths = []
        for p in simulated_gt_trajectories:
            path = Path(p).expanduser()
            if not path.is_absolute():
                path = project_root / path
            path = path.resolve()
            simulated_gt_paths.append(path)
        missing = [p for p in simulated_gt_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Simulated GT trajectory file(s) not found:\n"
                + "\n".join(f"  - {p}" for p in missing)
            )
    unclean_results = []
    cleaned_results = []

    # Create evaluation results directory
    if experiment_name:
        evaluation_results_dir = None  # set per-domain inside the loop
    else:
        evaluation_results_dir = benchmark_path / 'data' / 'evaluation_results'
        evaluation_results_dir.mkdir(parents=True, exist_ok=True)

    # Select appropriate experiment data directories based on mode
    experiment_data_dirs = experiment_data_dirs_masked if mode == 'masked' else experiment_data_dirs_fullyobs

    # Filter domains if specific domains are requested
    if selected_domains:
        domains_to_run = {k: v for k, v in domain_name_mappings.items() if k in selected_domains}
    else:
        domains_to_run = domain_name_mappings

    # Persist effective run configuration for reproducibility/debugging.
    run_params = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "experiment_name": experiment_name,
        "selected_domains": selected_domains if selected_domains is not None else "all",
        "domains_to_run": list(domains_to_run.keys()),
        "experiment_data_dirs": {d: experiment_data_dirs[d] for d in domains_to_run.keys()},
        "n_folds": N_FOLDS,
        "num_trajectories_list": NUM_TRAJECTORIES_LIST,
        "gt_rate_percentages": GT_RATE_PERCENTAGES,
        "frame_axiom_mode": FRAME_AXIOM_MODE,
        "learning_timeout_seconds": learning_timeout_seconds,
        "planning_timeout_seconds": planning_timeout_seconds,
        "fluent_patch_cost": fluent_patch_cost,
        "fluent_patch_weight": fluent_patch_weight,
        "model_patch_cost": model_patch_cost,
        "model_constraint_weight": model_constraint_weight,
        "max_search_nodes": max_search_nodes,
        "search_mode": search_mode,
        "node_choosing_strategy": node_choosing_strategy,
        "conflict_group_strategy": conflict_group_strategy,
        "fluent_branch_mode": fluent_branch_mode,
        "baselines": [r.display_name for r in baselines] if baselines else [],
        "simulated_mode": use_simulated,
        "simulated_gt_trajectories": [str(p) for p in simulated_gt_paths] if use_simulated else None,
        "simulated_masking_p": simulated_masking_p if use_simulated else None,
        "simulated_masking_strategy": simulated_masking_strategy if use_simulated else None,
        "simulated_noising_p": simulated_noising_p if use_simulated else None,
        "simulated_noising_strategy": simulated_noising_strategy if use_simulated else None,
        "simulated_seed": simulated_seed if use_simulated else None,
    }
    if evaluation_results_dir is not None:
        run_params_path = evaluation_results_dir / "run_params.json"
        with open(run_params_path, "w") as f:
            json.dump(run_params, f, indent=2)
        print(f"Saved run params to: {run_params_path}")

    print(f"\n{'='*80}")
    print(f"RUNNING BENCHMARK IN {mode.upper()} MODE")
    print(f"Domains: {list(domains_to_run.keys())}")
    print(f"{'='*80}\n")

    for domain_name, bench_name in domains_to_run.items():
        domain_ref_path = domain_properties[domain_name]["domain_path"]

        for dir_name in experiment_data_dirs[domain_name]:
            data_dir = benchmark_path / 'data' / domain_name / dir_name
            trajectories_dir = data_dir / 'training' / 'trajectories'

            if experiment_name:
                experiment_root = benchmark_path / 'data' / 'new_experiments' / domain_name / experiment_name
                testing_dir = experiment_root / 'testing'
                evaluation_results_dir = experiment_root / 'evaluation_results'
                evaluation_results_dir.mkdir(parents=True, exist_ok=True)
                if not (evaluation_results_dir / "run_params.json").exists():
                    with open(evaluation_results_dir / "run_params.json", "w") as f:
                        json.dump(run_params, f, indent=2)
                    print(f"Saved run params to: {evaluation_results_dir / 'run_params.json'}")
            else:
                testing_dir = data_dir / 'testing'

            testing_dir.mkdir(parents=True, exist_ok=True)

            problem_dirs = sorted([d for d in trajectories_dir.iterdir() if d.is_dir()])
            n_problems = len(problem_dirs)

            if n_problems < 2:
                raise ValueError(f"Domain {bench_name} has too few problems ({n_problems}) for 80/20 CV.")

            # Validate all problem directories have PDDL files BEFORE starting experiments
            print(f"Validating {n_problems} problem directories...")
            invalid_dirs = []
            for prob_dir in problem_dirs:
                # Use consistent naming: {problem_dir_name}.pddl
                problem_pddl_path = prob_dir / f"{prob_dir.name}.pddl"
                if not problem_pddl_path.exists():
                    # Try glob as fallback
                    pddl_files = list(prob_dir.glob("*.pddl"))
                    if not pddl_files:
                        invalid_dirs.append(prob_dir.name)

            if invalid_dirs:
                raise ValueError(
                    f"Domain {bench_name} has {len(invalid_dirs)} problem directories without PDDL files:\n"
                    f"  {invalid_dirs}\n"
                    f"All problem directories must contain a PDDL file for CV to work correctly.\n"
                    f"Expected naming: {{problem_dir_name}}/{{problem_dir_name}}.pddl"
                )

            print(f"✓ All {n_problems} problem directories validated")

            print(f"\n{'=' * 80}")
            print(f"Domain: {bench_name} | data dir: {dir_name}")
            print(f"Total problems: {n_problems}")
            print(f"Number of trajectories: {NUM_TRAJECTORIES_LIST}")
            print(f"GT rates: {GT_RATE_PERCENTAGES}")
            print(f"Frame axiom mode: {FRAME_AXIOM_MODE}")
            print(f"Learning timeout: {learning_timeout_seconds}s")
            print(f"Planning timeout: {planning_timeout_seconds}s")
            print(
                f"Denoising params: fluent_cost={fluent_patch_cost}, "
                f"fluent_weight={fluent_patch_weight}, "
                f"model_cost={model_patch_cost}, "
                f"model_constraint_weight={model_constraint_weight}, "
                f"max_search_nodes={max_search_nodes if max_search_nodes is not None else 'unlimited'}, "
                f"search_mode={search_mode}, "
                f"node_choosing_strategy={node_choosing_strategy}, "
                f"conflict_group_strategy={conflict_group_strategy}, "
                f"fluent_branch_mode={fluent_branch_mode}"
            )
            print(f"CV folds: {N_FOLDS}")
            print(f"{'=' * 80}\n")

            # PRE-GENERATE all GT+frame-axiom files before experiments
            # (skipped in simulated mode — noise is injected in memory)
            if not use_simulated:
                pregenerate_all_gt_frame_axiom_files(
                    problem_dirs, domain_ref_path, GT_RATE_PERCENTAGES, FRAME_AXIOM_MODE
                )

            # NEW: Iterate over number of trajectories instead of trajectory sizes
            for num_trajectories in NUM_TRAJECTORIES_LIST:
                print(f"\n{'='*60}\nNUMBER OF TRAJECTORIES = {num_trajectories}\n{'='*60}")

                for gt_rate in GT_RATE_PERCENTAGES:
                    gt_info = f"GT rate: {gt_rate}%" if gt_rate > 0 else "Baseline (GT only at t=0)"
                    print(f"\n{'-'*60}\n{gt_info}\n{'-'*60}")

                    # Run all folds in parallel
                    n_total_jobs = N_FOLDS
                    print(f"  [MAIN] Starting {n_total_jobs} fold jobs...")
                    with ProcessPoolExecutor(max_workers=N_FOLDS) as executor:
                        futures = []
                        for fold in range(N_FOLDS):
                            fold_kwargs = dict(
                                fold=fold,
                                problem_dirs=problem_dirs,
                                n_problems=n_problems,
                                num_trajectories=num_trajectories,
                                gt_rate=gt_rate,
                                domain_ref_path=domain_ref_path,
                                testing_dir=testing_dir,
                                bench_name=bench_name,
                                mode=mode,
                                evaluate_model_func=evaluate_model,
                                save_learning_metrics_func=save_learning_metrics,
                                conflict_search_timeout=learning_timeout_seconds,
                                planning_timeout=planning_timeout_seconds,
                                fluent_patch_cost=fluent_patch_cost,
                                fluent_patch_weight=fluent_patch_weight,
                                model_patch_cost=model_patch_cost,
                                model_constraint_weight=model_constraint_weight,
                                max_search_nodes=max_search_nodes,
                                search_mode=search_mode,
                                node_choosing_strategy=node_choosing_strategy,
                                conflict_group_strategy=conflict_group_strategy,
                                fluent_branch_mode=fluent_branch_mode,
                                baselines=baselines,
                            )
                            if use_simulated:
                                fold_kwargs.update(
                                    simulated_gt_trajectories=simulated_gt_paths,
                                    simulated_masking_strategy=MaskingType(simulated_masking_strategy),
                                    simulated_masking_p=simulated_masking_p,
                                    simulated_noising_strategy=NoisingType(simulated_noising_strategy),
                                    simulated_noising_p=simulated_noising_p,
                                    simulated_seed=simulated_seed,
                                )
                            future = executor.submit(run_single_fold, **fold_kwargs)
                            futures.append(future)

                        print(f"  [MAIN] All {n_total_jobs} fold tasks submitted, waiting for completion...")

                        # Wait for all jobs to complete and collect results
                        completed_count = 0
                        completed_folds = set()
                        import time
                        start_time = time.time()
                        per_job_timeout = 1800
                        batch_timeout = per_job_timeout

                        for future in as_completed(futures, timeout=batch_timeout):
                            try:
                                completed_count += 1
                                elapsed = time.time() - start_time
                                print(f"  [MAIN] Job {completed_count}/{n_total_jobs} completed after {elapsed:.1f}s, collecting results...")
                                results_list = future.result(timeout=per_job_timeout)

                                fold_num = results_list[0]['fold'] if results_list else '?'
                                completed_folds.add(fold_num)

                                # Separate by phase
                                for result in results_list:
                                    phase = result['_internal_phase']
                                    if phase == 'unclean':
                                        unclean_results.append(result)
                                    else:
                                        cleaned_results.append(result)

                                print(f"  [MAIN] Fold {fold_num} results processed. "
                                      f"Jobs done: {completed_count}/{n_total_jobs}")
                            except TimeoutError:
                                print(f"TIMEOUT: Job {completed_count} exceeded time limit")
                                print(f"  Completed so far: {completed_count}/{n_total_jobs}")
                            except Exception as e:
                                print(f"ERROR in job {completed_count}: {e}")
                                import traceback
                                traceback.print_exc()

                    print(f"✓ All {n_total_jobs} jobs for num_trajectories={num_trajectories}, "
                          f"gt_rate={gt_rate}% completed")

                    # Write CSV files
                    timeout_suffix = f"_timeout{learning_timeout_seconds}s"
                    csv_unclean = evaluation_results_dir / f"results_{bench_name}_unclean{timeout_suffix}.csv"
                    csv_cleaned = evaluation_results_dir / f"results_{bench_name}{timeout_suffix}.csv"

                    pd.DataFrame(unclean_results).to_csv(csv_unclean, index=False)
                    pd.DataFrame(cleaned_results).to_csv(csv_cleaned, index=False)

                    # Create combined CSV (unclean + cleaned results)
                    csv_combined = evaluation_results_dir / f"results_{bench_name}_combined{timeout_suffix}.csv"

                    # Filter results for this domain
                    domain_results = [r for r in unclean_results + cleaned_results if r['domain'] == bench_name]
                    pd.DataFrame(domain_results).to_csv(csv_combined, index=False)

                    print(f"\n✓ Results written:")
                    print(f"  - Unclean: {csv_unclean}")
                    print(f"  - Cleaned: {csv_cleaned}")
                    print(f"  - Combined: {csv_combined}")

                print(f"\n✓ All folds for num_trajectories={num_trajectories} completed")

                # Generate Excel report after each num_trajectories completes
                print(f"\n{'='*60}")
                print(f"GENERATING AGGREGATED REPORT FOR NUM_TRAJECTORIES = {num_trajectories}")
                print(f"{'='*60}")

                timestamp = datetime.now().strftime("%d-%m-%YT%H:%M:%S")
                xlsx_path = evaluation_results_dir / f"benchmark_results_{timestamp}.xlsx"
                generate_excel_report(unclean_results, cleaned_results, xlsx_path)
                print(f"✓ Excel report saved to: {xlsx_path}")
                completed_numtrajs = sorted(set(r['num_trajectories'] for r in unclean_results))
                print(f"  Completed num_trajectories so far: {completed_numtrajs}")

                domain_results = [r for r in unclean_results + cleaned_results if r['domain'] == bench_name]
                if not domain_results:
                    print(f"\nWarning: No results collected for {bench_name} at num_trajectories={num_trajectories}; skipping plots")
                else:
                    # Generate GT injection analysis plots
                    print(f"\n{'='*60}")
                    print(f"GENERATING GT INJECTION PLOTS")
                    print(f"{'='*60}")
                    generate_gt_injection_plots(csv_combined, evaluation_results_dir, bench_name)

                    # Generate stacked solving rate plots
                    print(f"\n{'='*60}")
                    print(f"GENERATING STACKED SOLVING RATE PLOTS")
                    print(f"{'='*60}")
                    plot_stacked_solving_rate(csv_combined, evaluation_results_dir, bench_name)

                # Generate plots after each num_trajectories
                plots_dir = evaluation_results_dir / "plots"
                generate_plots(unclean_results, cleaned_results, plots_dir)
                print(f"✓ Plots updated with results up to num_trajectories={num_trajectories}")

    # =============================================================================
    # CROSS-FOLD CORRELATION ANALYSIS
    # =============================================================================
    print("\n" + "=" * 80)
    print("CROSS-FOLD CORRELATION ANALYSIS")
    print("=" * 80)

    for domain_name, bench_name in domains_to_run.items():
        for dir_name in experiment_data_dirs[domain_name]:
            if experiment_name:
                corr_testing_dir = benchmark_path / 'data' / 'new_experiments' / domain_name / experiment_name / 'testing'
                corr_eval_dir = benchmark_path / 'data' / 'new_experiments' / domain_name / experiment_name / 'evaluation_results'
            else:
                corr_testing_dir = benchmark_path / 'data' / domain_name / dir_name / 'testing'
                corr_eval_dir = evaluation_results_dir

            if not corr_testing_dir.exists():
                print(f"  {bench_name}: testing dir not found, skipping correlation.")
                continue

            fold_dirs = sorted(
                d for d in corr_testing_dir.iterdir()
                if d.is_dir() and d.name.startswith("fold")
                and (d / "correlation_analysis.json").exists()
            )

            if fold_dirs:
                corr_output = corr_eval_dir / f"correlation_{bench_name}"
                corr_result = aggregate_correlation_tables(fold_dirs, corr_output)
                print(f"  {bench_name}: {corr_result['n']} data points, "
                      f"output → {corr_output.name}/")
            else:
                print(f"  {bench_name}: no per-fold correlation tables found, skipping.")

    # =============================================================================
    # FINAL SUMMARY
    # =============================================================================

    # Create all-domains combined CSV file
    if evaluation_results_dir is not None:
        csv_all_combined = evaluation_results_dir / "results_all_domains_combined.csv"
        all_unclean = [dict(r, phase='unclean') for r in unclean_results]
        all_cleaned = [dict(r, phase='cleaned') for r in cleaned_results]
        all_combined_data = all_unclean + all_cleaned
        pd.DataFrame(all_combined_data).to_csv(csv_all_combined, index=False)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 80)
    print(f"\nTotal unclean results: {len(unclean_results)}")
    print(f"Total cleaned results: {len(cleaned_results)}")
    if evaluation_results_dir is not None:
        print(f"\nAll evaluation results saved to: {evaluation_results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run PDDL action model learning benchmark')
    parser.add_argument('--domain', type=str, default='all',
                       help='Domain to run (blocksworld, hanoi, n_puzzle_typed, maze, or "all" for all domains)')
    parser.add_argument('--mode', type=str, default='masked', choices=['masked', 'fullyobs'],
                       help='Mode to run: "masked" (PISAM/PO_ROSAME) or "fullyobs" (SAM/ROSAME)')
    parser.add_argument('--learning-timeout-seconds', type=int, default=180,
                        help='Timeout in seconds for denoising conflict search')
    parser.add_argument('--planning-timeout-seconds', type=int, default=PLANNING_TIMEOUT,
                        help='Timeout in seconds for planning during evaluation')
    parser.add_argument('--fluent-patch-cost', type=float, default=FLUENT_PATCH_COST,
                        help='Per-patch cost for fluent patches in conflict search')
    parser.add_argument('--fluent-patch-weight', type=float, default=FLUENT_PATCH_WEIGHT,
                        help='Weight multiplier for fluent patch cost in conflict search')
    parser.add_argument('--model-patch-cost', type=float, default=MODEL_PATCH_COST,
                        help='Per-patch cost for model patches/constraints in conflict search')
    parser.add_argument('--model-constraint-weight', type=float, default=MODEL_CONSTRAINT_WEIGHT,
                        help='Weight multiplier for model constraint cost in conflict search')
    parser.add_argument('--max-search-nodes', type=int, default=0,
                        help='Max conflict-search nodes; <=0 means unlimited')
    parser.add_argument('--search-mode', type=str, default='dfs', choices=['dfs', 'ucs'],
                        help='Conflict-search strategy for denoising: "dfs" (anytime DFS) or "ucs"')
    parser.add_argument(
        '--node-choosing-strategy',
        type=str,
        default='model_patch_first',
        choices=['model_patch_first', 'fluent_patch_first', 'fluent_patch_first_then_model', 'randomized'],
        help=(
            'Order strategy for inserting denoising branch children: '
            '"model_patch_first", "fluent_patch_first", '
            '"fluent_patch_first_then_model" (fluent-first until first solution, then model-first), '
            'or "randomized"'
        ),
    )
    parser.add_argument(
        '--conflict-group-strategy',
        type=str,
        default='most_observations',
        choices=['first', 'largest', 'largest_model_patchable', 'most_observations', 'smallest'],
        help=(
            'Which conflict group to resolve first at each search node: '
            '"first" (original: by priority then position), '
            '"largest" (prefer groups with most conflicts), '
            '"largest_model_patchable" (prefer largest non-FRAME_AXIOM groups), '
            '"most_observations" (prefer groups spanning most distinct observations), '
            '"smallest" (prefer groups with fewest conflicts — best paired with fluent_patch_first)'
        ),
    )
    parser.add_argument(
        '--fluent-branch-mode',
        type=str,
        default='group',
        choices=['group', 'single'],
        help=(
            'How many fluent patches per data-fix branch: '
            '"group" (default, all conflicts in the chosen group at once), '
            '"single" (one patch per branch — finer-grained, avoids inflated cost)'
        ),
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help=(
            'Self-contained experiment folder name. When set, testing/ and '
            'evaluation_results/ are created under '
            'benchmark/data/new_experiments/{domain}/{name}/ '
            'instead of the default locations.'
        ),
    )

    # --- Simulated data source ---
    parser.add_argument(
        '--simulated-gt-trajectories',
        type=str,
        nargs='+',
        default=None,
        help=(
            'Paths to ground-truth .trajectory files for simulated mode. '
            'When provided, synthetic masking + noise is applied in memory '
            'instead of reading pre-generated files from disk.'
        ),
    )
    parser.add_argument('--simulated-masking-p', type=float, default=0.4,
                        help='Masking probability for simulated mode (default: 0.4)')
    parser.add_argument('--simulated-masking-strategy', type=str, default='percentage',
                        choices=['percentage', 'random'],
                        help='Masking strategy for simulated mode (default: percentage)')
    parser.add_argument('--simulated-noising-p', type=float, default=0.15,
                        help='Noising probability for simulated mode (default: 0.15)')
    parser.add_argument('--simulated-noising-strategy', type=str, default='percentage',
                        choices=['percentage', 'random'],
                        help='Noising strategy for simulated mode (default: percentage)')
    parser.add_argument('--simulated-seed', type=int, default=42,
                        help='Random seed for simulated noise injection (default: 42)')

    # --- Pluggable baselines ---
    parser.add_argument(
        '--baselines',
        type=str,
        nargs='*',
        default=['rosame'],
        help=(
            'Baseline algorithms to run alongside SAM/PISAM. '
            'Pass algorithm family names (e.g., "rosame"). '
            'Use --baselines with no arguments to run NO baselines. '
            'Default: rosame.'
        ),
    )

    args = parser.parse_args()

    if args.learning_timeout_seconds <= 0:
        parser.error("--learning-timeout-seconds must be > 0")
    if args.planning_timeout_seconds <= 0:
        parser.error("--planning-timeout-seconds must be > 0")
    if args.fluent_patch_cost < 0:
        parser.error("--fluent-patch-cost must be >= 0")
    if args.fluent_patch_weight < 0:
        parser.error("--fluent-patch-weight must be >= 0")
    if args.model_patch_cost < 0:
        parser.error("--model-patch-cost must be >= 0")
    if args.model_constraint_weight < 0:
        parser.error("--model-constraint-weight must be >= 0")

    max_search_nodes = None if args.max_search_nodes <= 0 else args.max_search_nodes

    # Determine which domains to run
    if args.domain == 'all':
        selected_domains = None  # Run all domains in domain_name_mappings
    else:
        # Validate domain name
        if args.domain not in domain_properties:
            print(f"Error: Unknown domain '{args.domain}'")
            print(f"Available domains: {list(domain_properties.keys())}")
            exit(1)
        selected_domains = [args.domain]

    # Instantiate baseline runners
    from benchmark.baselines import get_baselines
    baseline_runners = get_baselines(args.baselines) if args.baselines else []
    if baseline_runners:
        print(f"Baselines: {', '.join(r.display_name for r in baseline_runners)}")
    else:
        print("Baselines: NONE (only SAM/PISAM will run)")

    main(
        selected_domains=selected_domains,
        mode=args.mode,
        learning_timeout_seconds=args.learning_timeout_seconds,
        planning_timeout_seconds=args.planning_timeout_seconds,
        fluent_patch_cost=args.fluent_patch_cost,
        fluent_patch_weight=args.fluent_patch_weight,
        model_patch_cost=args.model_patch_cost,
        model_constraint_weight=args.model_constraint_weight,
        max_search_nodes=max_search_nodes,
        search_mode=args.search_mode,
        node_choosing_strategy=args.node_choosing_strategy,
        conflict_group_strategy=args.conflict_group_strategy,
        fluent_branch_mode=args.fluent_branch_mode,
        experiment_name=args.experiment_name,
        simulated_gt_trajectories=args.simulated_gt_trajectories,
        simulated_masking_p=args.simulated_masking_p,
        simulated_masking_strategy=args.simulated_masking_strategy,
        simulated_noising_p=args.simulated_noising_p,
        simulated_noising_strategy=args.simulated_noising_strategy,
        simulated_seed=args.simulated_seed,
        baselines=baseline_runners,
    )

"""cli running command:
python -m benchmark.amlgym_testing \
  --domain blocksworld \
  --mode masked \
  --learning-timeout-seconds 300 \
  --planning-timeout-seconds 60 \
  --search-mode dfs \
  --node-choosing-strategy model_patch_first \
  --conflict-group-strategy largest \
  --model-constraint-weight 0.0 \
  --experiment-name MW=0__modelFirst__largest__FAfixed
"""