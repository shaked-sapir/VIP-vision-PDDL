"""Benchmark testing script for PDDL action model learning.

Runs cross-validation experiments on generated trajectory data, evaluating
learned models against a reference domain file.

Usage:
    python -m benchmark.benchmark_runner \\
        --domain blocks \\
        --data-dir benchmark/data/blocksworld/multi_problem_...

    The --domain arg is a key from config.yaml (blocks, hanoi, npuzzle, etc.).
    The --data-dir arg is the experiment directory output by data_generator.py.
"""

import argparse
import json
import sys
import time
from concurrent.futures import as_completed, ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
from amlgym.metrics import print_metrics

from benchmark.baselines import get_baselines
from benchmark.evaluation.correlation_analysis import aggregate_correlation_tables
from benchmark.experiment_running_helpers.data_source import DataSource, ImageDataSource, SimulatedDataSource
from benchmark.experiment_running_helpers.evaluation import evaluate_model, save_learning_metrics
from benchmark.experiment_running_helpers.normalize import normalize_experiment_data
from benchmark.experiment_running_helpers.run_fold import run_single_fold
from benchmark.experiment_running_helpers.reporting import (
    generate_excel_report,
    generate_plots,
    generate_gt_injection_plots,
    plot_stacked_solving_rate,
)
from src.pi_sam.masking import MaskingType
from src.pi_sam.noising import NoisingType
from src.utils.config import load_config


# ── Paths (derived, not hardcoded) ──────────────────────────────────────
benchmark_path = Path(__file__).resolve().parent
project_root = benchmark_path.parent


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _resolve_experiment_paths(
    data_dir: Path,
    experiment_name: Optional[str],
    domain_name: str,
    default_evaluation_results_dir: Optional[Path],
) -> tuple[Path, Path, Path, Optional[Path]]:
    """Return (data_dir, trajectories_dir, testing_dir, evaluation_results_dir)."""
    trajectories_dir = data_dir / 'training' / 'trajectories'

    if experiment_name:
        experiment_root = benchmark_path / 'running_results' / domain_name / experiment_name
        return data_dir, trajectories_dir, experiment_root / 'testing', experiment_root / 'evaluation_results'

    return data_dir, trajectories_dir, data_dir / 'testing', default_evaluation_results_dir


def _save_run_params(path: Path, run_params: dict, *, skip_if_exists: bool = False) -> None:
    """Write run_params.json, optionally skipping if the file already exists."""
    if skip_if_exists and path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(run_params, f, indent=2)
    print(f"Saved run params to: {path}")


def _results_for_domain(results: List[dict], display_domain_name: str) -> List[dict]:
    return [r for r in results if r['domain'] == display_domain_name]


def _resolve_domain_config(domain_key: str) -> dict:
    """Load domain configuration from config.yaml.

    Returns dict with:
        display_name: str  — benchmark display name (e.g. "blocksworld")
        domain_ref_path: Path — reference domain file for evaluation metrics
    """
    config = load_config()
    domain_config = config["domains"].get(domain_key)
    if domain_config is None:
        available = list(config["domains"].keys())
        raise ValueError(f"Unknown domain '{domain_key}'. Available: {available}")

    # eval_domain_file is the reference domain for evaluation (may differ from
    # the source domain file used for data generation).
    eval_domain_file = domain_config.get("eval_domain_file")
    if eval_domain_file:
        domain_ref_path = project_root / eval_domain_file
    else:
        # Fall back to the source domain file
        domain_ref_path = project_root / domain_config["domain_file"]

    return {
        "display_name": domain_config.get("display_name", domain_key),
        "domain_ref_path": domain_ref_path,
        "domain_file": project_root / domain_config["domain_file"],
    }


# =============================================================================
# MAIN EXPERIMENT LOOP
# =============================================================================
def main(
    domain_key: str,
    data_dir: Path,
    data_source: DataSource,
    mode: str = 'masked',
    n_folds: int = 5,
    num_trajectories_list: List[int] = None,
    gt_rate_percentages: List[int] = None,
    frame_axiom_mode: str = "after_gt_only",
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
    normalize: bool = True,
    force_normalize: bool = False,
    baselines: list = None,
    events_tracing: bool = False,
):
    """Run benchmark experiments on a single domain.

    Args:
        domain_key: Config key from config.yaml (e.g. "blocksworld", "hanoi").
        data_dir: Path to the experiment data directory (output of data_generator.py).
        data_source: DataSource instance controlling where observations come from.
            Use ImageDataSource() for real image-pipeline data (pre-generated files)
            or SimulatedDataSource(...) for synthetic in-memory noise injection.
        mode: Either 'masked' or 'fullyobs'.
        n_folds: Number of cross-validation folds.
        num_trajectories_list: List of trajectory counts to evaluate.
        gt_rate_percentages: List of GT injection rates (0 = baseline).
        frame_axiom_mode: "after_gt_only" or "all_states".
        normalize: Run normalization pre-processing before evaluation.
        force_normalize: Re-normalize even if normalized data exists.
    """
    if num_trajectories_list is None:
        num_trajectories_list = [3, 4, 5, 6, 7, 8]
    if gt_rate_percentages is None:
        gt_rate_percentages = [0]

    # Print metrics banner (was previously a module-level side effect)
    print_metrics()

    # Load domain configuration from config.yaml
    domain_info = _resolve_domain_config(domain_key)
    display_domain_name = domain_info["display_name"]
    domain_ref_path = domain_info["domain_ref_path"]
    domain_file = domain_info["domain_file"]

    # Validate data dir exists
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # ── Normalization (image pipeline only) ─────────────────────────────
    if normalize and isinstance(data_source, ImageDataSource):
        print("\n" + "=" * 80)
        print("NORMALIZATION PRE-PROCESSING")
        print("=" * 80)
        try:
            norm_trajs_dir, norm_domain_file = normalize_experiment_data(
                data_dir, domain_file, force=force_normalize,
            )
            # Use normalized data for evaluation
            domain_ref_path = norm_domain_file
        except ValueError as e:
            print(f"\n  ⚠ Normalization failed: {e}")
            print("  Continuing with unnormalized data...")
            norm_trajs_dir = None
    else:
        norm_trajs_dir = None

    unclean_results = []
    cleaned_results = []

    # ── Resolve paths ────────────────────────────────────────────────────
    if experiment_name:
        evaluation_results_dir = None  # set inside the loop
    else:
        evaluation_results_dir = benchmark_path / 'data' / 'evaluation_results'
        evaluation_results_dir.mkdir(parents=True, exist_ok=True)

    data_dir_resolved, trajectories_dir, testing_dir, evaluation_results_dir = \
        _resolve_experiment_paths(data_dir, experiment_name, display_domain_name, evaluation_results_dir)

    # Use normalized trajectories dir if available
    if norm_trajs_dir is not None:
        trajectories_dir = norm_trajs_dir

    if experiment_name:
        evaluation_results_dir = evaluation_results_dir  # already set by _resolve

    # ── Persist run params ───────────────────────────────────────────────
    run_params = {
        "timestamp": datetime.now().isoformat(),
        "domain_key": domain_key,
        "display_domain_name": display_domain_name,
        "data_dir": str(data_dir),
        "mode": mode,
        "experiment_name": experiment_name,
        "n_folds": n_folds,
        "num_trajectories_list": num_trajectories_list,
        "gt_rate_percentages": gt_rate_percentages,
        "frame_axiom_mode": frame_axiom_mode,
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
        "normalized": norm_trajs_dir is not None,
        "data_source_type": type(data_source).__name__,
    }
    if evaluation_results_dir is not None:
        _save_run_params(evaluation_results_dir / "run_params.json", run_params)

    # ── Validate problem directories ─────────────────────────────────────
    testing_dir.mkdir(parents=True, exist_ok=True)

    problem_dirs = sorted([d for d in trajectories_dir.iterdir() if d.is_dir()])
    n_problems = len(problem_dirs)

    if n_problems < 2:
        raise ValueError(
            f"Domain {display_domain_name} has too few problems ({n_problems}) "
            f"for 80/20 CV in {trajectories_dir}."
        )

    print(f"\nValidating {n_problems} problem directories...")
    invalid_dirs = []
    for prob_dir in problem_dirs:
        problem_pddl_path = prob_dir / f"{prob_dir.name}.pddl"
        if not problem_pddl_path.exists():
            pddl_files = list(prob_dir.glob("*.pddl"))
            if not pddl_files:
                invalid_dirs.append(prob_dir.name)

    if invalid_dirs:
        raise ValueError(
            f"Domain {display_domain_name} has {len(invalid_dirs)} problem directories "
            f"without PDDL files:\n  {invalid_dirs}\n"
            f"Expected naming: {{problem_dir_name}}/{{problem_dir_name}}.pddl"
        )

    print(f"✓ All {n_problems} problem directories validated")

    # ── Print experiment info ────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"RUNNING BENCHMARK IN {mode.upper()} MODE")
    print(f"Domain: {display_domain_name} (config key: {domain_key})")
    print(f"Data dir: {data_dir}")
    print(f"Total problems: {n_problems}")
    print(f"Number of trajectories: {num_trajectories_list}")
    print(f"GT rates: {gt_rate_percentages}")
    print(f"Frame axiom mode: {frame_axiom_mode}")
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
    print(f"CV folds: {n_folds}")
    print(f"{'=' * 80}\n")

    # One-time data source setup (pre-generates files for ImageDataSource; no-op for SimulatedDataSource)
    data_source.setup(problem_dirs, domain_ref_path, gt_rate_percentages, frame_axiom_mode)

    # ── Experiment loop ──────────────────────────────────────────────────
    for num_trajectories in num_trajectories_list:
        print(f"\n{'='*60}\nNUMBER OF TRAJECTORIES = {num_trajectories}\n{'='*60}")

        for gt_rate in gt_rate_percentages:
            gt_info = f"GT rate: {gt_rate}%" if gt_rate > 0 else "Baseline (GT only at t=0)"
            print(f"\n{'-'*60}\n{gt_info}\n{'-'*60}")

            n_total_jobs = n_folds
            print(f"  [MAIN] Starting {n_total_jobs} fold jobs...")
            with ProcessPoolExecutor(max_workers=n_folds) as executor:
                futures = []
                for fold in range(n_folds):
                    fold_kwargs = dict(
                        fold=fold,
                        problem_dirs=problem_dirs,
                        n_problems=n_problems,
                        num_trajectories=num_trajectories,
                        gt_rate=gt_rate,
                        domain_ref_path=domain_ref_path,
                        testing_dir=testing_dir,
                        bench_name=display_domain_name,
                        mode=mode,
                        data_source=data_source,
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
                        events_tracing=events_tracing,
                    )
                    future = executor.submit(run_single_fold, **fold_kwargs)
                    futures.append(future)

                print(f"  [MAIN] All {n_total_jobs} fold tasks submitted, waiting for completion...")

                completed_count = 0
                start_time = time.time()
                per_job_timeout = learning_timeout_seconds * 2
                batch_timeout = n_total_jobs * per_job_timeout

                for future in as_completed(futures, timeout=batch_timeout):
                    try:
                        completed_count += 1
                        elapsed = time.time() - start_time
                        print(f"  [MAIN] Job {completed_count}/{n_total_jobs} completed after {elapsed:.1f}s, collecting results...")
                        results_list = future.result(timeout=per_job_timeout)

                        fold_num = results_list[0]['fold'] if results_list else '?'

                        for result in results_list:
                            phase = result['_internal_phase']
                            if phase == 'unclean':
                                unclean_results.append(result)
                            else:
                                cleaned_results.append(result)

                        print(f"  [MAIN] Fold {fold_num} results processed. Jobs done: {completed_count}/{n_total_jobs}")
                    except TimeoutError:
                        print(f"TIMEOUT: Job {completed_count} exceeded time limit")
                        print(f"  Completed so far: {completed_count}/{n_total_jobs}")
                    except Exception as e:
                        print(f"ERROR in job {completed_count}: {e}")
                        import traceback
                        traceback.print_exc()

            print(f"✓ All {n_total_jobs} jobs for num_trajectories={num_trajectories}, gt_rate={gt_rate}% completed")

            # Write CSV files
            timeout_suffix = f"_timeout{learning_timeout_seconds}s"
            csv_unclean = evaluation_results_dir / f"results_{display_domain_name}_unclean{timeout_suffix}.csv"
            csv_cleaned = evaluation_results_dir / f"results_{display_domain_name}{timeout_suffix}.csv"

            pd.DataFrame(unclean_results).to_csv(csv_unclean, index=False)
            pd.DataFrame(cleaned_results).to_csv(csv_cleaned, index=False)

            csv_combined = evaluation_results_dir / f"results_{display_domain_name}_combined{timeout_suffix}.csv"
            pd.DataFrame(
                _results_for_domain(unclean_results + cleaned_results, display_domain_name)
            ).to_csv(csv_combined, index=False)

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

        domain_results = _results_for_domain(unclean_results + cleaned_results, display_domain_name)
        if not domain_results:
            print(f"\nWarning: No results collected for {display_domain_name} at num_trajectories={num_trajectories}; skipping plots")
        else:
            print(f"\n{'='*60}")
            print(f"GENERATING GT INJECTION PLOTS")
            print(f"{'='*60}")
            generate_gt_injection_plots(csv_combined, evaluation_results_dir, display_domain_name)

            print(f"\n{'='*60}")
            print(f"GENERATING STACKED SOLVING RATE PLOTS")
            print(f"{'='*60}")
            plot_stacked_solving_rate(csv_combined, evaluation_results_dir, display_domain_name)

        plots_dir = evaluation_results_dir / "plots"
        generate_plots(unclean_results, cleaned_results, plots_dir)
        print(f"✓ Plots updated with results up to num_trajectories={num_trajectories}")

    # ── Cross-fold correlation analysis ──────────────────────────────────
    print("\n" + "=" * 80)
    print("CROSS-FOLD CORRELATION ANALYSIS")
    print("=" * 80)

    if testing_dir.exists():
        fold_dirs = sorted(
            d for d in testing_dir.iterdir()
            if d.is_dir() and d.name.startswith("fold")
            and (d / "correlation_analysis.json").exists()
        )

        if fold_dirs:
            corr_output = evaluation_results_dir / f"correlation_{display_domain_name}"
            corr_result = aggregate_correlation_tables(fold_dirs, corr_output)
            print(f"  {display_domain_name}: {corr_result['n']} data points, output → {corr_output.name}/")
        else:
            print(f"  {display_domain_name}: no per-fold correlation tables found, skipping.")
    else:
        print(f"  {display_domain_name}: testing dir not found, skipping correlation.")

    # ── Final summary ────────────────────────────────────────────────────
    if evaluation_results_dir is not None:
        csv_all_combined = evaluation_results_dir / "results_all_domains_combined.csv"
        all_unclean = [dict(r, phase='unclean') for r in unclean_results]
        all_cleaned = [dict(r, phase='cleaned') for r in cleaned_results]
        pd.DataFrame(all_unclean + all_cleaned).to_csv(csv_all_combined, index=False)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 80)
    print(f"\nTotal unclean results: {len(unclean_results)}")
    print(f"Total cleaned results: {len(cleaned_results)}")
    if evaluation_results_dir is not None:
        print(f"\nAll evaluation results saved to: {evaluation_results_dir}")


# =============================================================================
# CLI
# =============================================================================

def _parse_int_list(value: str) -> List[int]:
    """Parse a comma-separated list of ints, e.g. '3,4,5,6,7,8'."""
    return [int(x.strip()) for x in value.split(',')]


if __name__ == "__main__":
    # Load config for domain validation
    _config = load_config()
    _available_domains = list(_config["domains"].keys())

    parser = argparse.ArgumentParser(
        description='Run PDDL action model learning benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python -m benchmark.benchmark_runner \\\n"
            "    --domain blocks \\\n"
            "    --data-dir benchmark/data/blocksworld/multi_problem_...\n"
        ),
    )

    # ── Required arguments ───────────────────────────────────────────────
    parser.add_argument(
        '--domain', type=str, required=True,
        choices=_available_domains,
        help=f'Domain config key ({", ".join(_available_domains)})',
    )
    parser.add_argument(
        '--data-dir', type=str, required=True,
        help='Path to the experiment data directory (output of data_generator.py)',
    )

    # ── Experiment parameters ────────────────────────────────────────────
    parser.add_argument('--mode', type=str, default='masked', choices=['masked', 'fullyobs'],
                        help='Mode: "masked" (PISAM/PO_ROSAME) or "fullyobs" (SAM/ROSAME)')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--num-trajectories', type=str, default='3,4,5,6,7,8',
                        help='Comma-separated trajectory counts to evaluate (default: 3,4,5,6,7,8)')
    parser.add_argument('--gt-rates', type=str, default='0',
                        help='Comma-separated GT injection rates in %% (default: 0)')
    parser.add_argument('--frame-axiom-mode', type=str, default='after_gt_only',
                        choices=['after_gt_only', 'all_states'],
                        help='Frame axiom propagation mode (default: after_gt_only)')
    parser.add_argument('--learning-timeout-seconds', type=int, default=180,
                        help='Timeout in seconds for denoising conflict search (default: 180)')
    parser.add_argument('--planning-timeout-seconds', type=int, default=60,
                        help='Timeout in seconds for planning during evaluation (default: 60)')

    # ── Conflict search parameters ───────────────────────────────────────
    parser.add_argument('--fluent-patch-cost', type=float, default=1.0,
                        help='Per-patch cost for fluent patches (default: 1.0)')
    parser.add_argument('--fluent-patch-weight', type=float, default=1.0,
                        help='Weight multiplier for fluent patch cost (default: 1.0)')
    parser.add_argument('--model-patch-cost', type=float, default=1.0,
                        help='Per-patch cost for model patches/constraints (default: 1.0)')
    parser.add_argument('--model-constraint-weight', type=float, default=0.0,
                        help='Weight multiplier for model constraint cost (default: 0.0)')
    parser.add_argument('--max-search-nodes', type=int, default=0,
                        help='Max conflict-search nodes; <=0 means unlimited')
    parser.add_argument('--search-mode', type=str, default='dfs', choices=['dfs', 'ucs'],
                        help='Conflict-search strategy (default: dfs)')
    parser.add_argument(
        '--node-choosing-strategy', type=str, default='model_patch_first',
        choices=['model_patch_first', 'fluent_patch_first', 'fluent_patch_first_then_model', 'randomized'],
        help='Order strategy for denoising branch children (default: model_patch_first)',
    )
    parser.add_argument(
        '--conflict-group-strategy', type=str, default='most_observations',
        choices=['first', 'largest', 'largest_model_patchable', 'most_observations', 'smallest'],
        help='Which conflict group to resolve first (default: most_observations)',
    )
    parser.add_argument(
        '--fluent-branch-mode', type=str, default='group', choices=['group', 'single'],
        help='Fluent patches per data-fix branch (default: group)',
    )

    # ── Experiment naming ────────────────────────────────────────────────
    parser.add_argument(
        '--experiment-name', type=str, default=None,
        help='Self-contained experiment folder name under benchmark/running_results/{domain}/{name}/',
    )

    # ── Normalization ────────────────────────────────────────────────────
    parser.add_argument('--no-normalize', action='store_true', default=False,
                        help='Skip normalization pre-processing')
    parser.add_argument('--force-normalize', action='store_true', default=False,
                        help='Re-normalize even if normalized data already exists')

    # ── Simulated data source (alternative to image-pipeline files) ──────
    parser.add_argument(
        '--simulated-gt-trajectories', type=str, nargs='+', default=None,
        help='GT trajectory files for SimulatedDataSource (synthetic masking + noise). '
             'Omit to use ImageDataSource (pre-generated image-pipeline files).',
    )
    parser.add_argument('--simulated-masking-p', type=float, default=0.4,
                        help='Masking probability for SimulatedDataSource (default: 0.4)')
    parser.add_argument('--simulated-masking-strategy', type=str, default='percentage',
                        choices=['percentage', 'random'],
                        help='Masking strategy for SimulatedDataSource (default: percentage)')
    parser.add_argument('--simulated-noising-p', type=float, default=0.15,
                        help='Noising probability for SimulatedDataSource (default: 0.15)')
    parser.add_argument('--simulated-noising-strategy', type=str, default='percentage',
                        choices=['percentage', 'random'],
                        help='Noising strategy for SimulatedDataSource (default: percentage)')
    parser.add_argument('--simulated-seed', type=int, default=42,
                        help='Random seed for SimulatedDataSource noise injection (default: 42)')

    # ── Tracing & baselines ──────────────────────────────────────────────
    parser.add_argument('--events-tracing', action='store_true', default=False,
                        help='Enable per-node event tracing during denoising')
    parser.add_argument(
        '--baselines', type=str, nargs='*', default=['rosame'],
        help='Baseline algorithms (e.g. "rosame"). Use "none" to skip. Default: rosame.',
    )

    args = parser.parse_args()

    # ── Validation ───────────────────────────────────────────────────────
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

    # Parse list arguments
    num_trajectories_list = _parse_int_list(args.num_trajectories)
    gt_rate_percentages = _parse_int_list(args.gt_rates)

    # Resolve data dir (allow relative paths)
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir

    # Baselines
    if not args.baselines:
        parser.error("--baselines requires a value (e.g. 'rosame', or 'none' to skip)")
    if len(args.baselines) == 1 and args.baselines[0].lower() == 'none':
        baseline_names = []
    elif any(name.lower() == 'none' for name in args.baselines):
        parser.error("Cannot combine 'none' with other baseline names")
    else:
        baseline_names = args.baselines

    baseline_runners = get_baselines(baseline_names) if baseline_names else []
    if baseline_runners:
        print(f"Baselines: {', '.join(r.display_name for r in baseline_runners)}")
    else:
        print("Baselines: NONE (only SAM/PISAM will run)")

    # ── Construct data source ─────────────────────────────────────────────
    if args.simulated_gt_trajectories:
        simulated_gt_paths = []
        for p in args.simulated_gt_trajectories:
            path = Path(p).expanduser()
            if not path.is_absolute():
                path = project_root / path
            simulated_gt_paths.append(path.resolve())
        missing = [p for p in simulated_gt_paths if not p.exists()]
        if missing:
            parser.error(
                "Simulated GT trajectory file(s) not found:\n"
                + "\n".join(f"  - {p}" for p in missing)
            )
        data_source = SimulatedDataSource(
            gt_trajectory_paths=simulated_gt_paths,
            masking_strategy=MaskingType(args.simulated_masking_strategy),
            masking_p=args.simulated_masking_p,
            noising_strategy=NoisingType(args.simulated_noising_strategy),
            noising_p=args.simulated_noising_p,
            seed=args.simulated_seed,
        )
        print(f"Data source: SimulatedDataSource ({len(simulated_gt_paths)} GT trajectories)")
    else:
        data_source = ImageDataSource()
        print("Data source: ImageDataSource (pre-generated image-pipeline files)")

    main(
        domain_key=args.domain,
        data_dir=data_dir,
        data_source=data_source,
        mode=args.mode,
        n_folds=args.n_folds,
        num_trajectories_list=num_trajectories_list,
        gt_rate_percentages=gt_rate_percentages,
        frame_axiom_mode=args.frame_axiom_mode,
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
        normalize=not args.no_normalize,
        force_normalize=args.force_normalize,
        baselines=baseline_runners,
        events_tracing=args.events_tracing,
    )
