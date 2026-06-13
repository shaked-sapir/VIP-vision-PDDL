import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import as_completed, ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd
from amlgym.metrics import print_metrics, syntactic_precision, syntactic_recall, problem_solving

from benchmark.evaluation.predictive_metrics import evaluate_predictive_power
from benchmark.evaluation.correlation_analysis import aggregate_correlation_tables

from benchmark.experiment_running_helpers.run_fold import run_single_fold
from benchmark.experiment_running_helpers.trajectory_utils import pregenerate_all_gt_frame_axiom_files

# =============================================================================
# CONFIG & PATHS
# =============================================================================

benchmark_path = Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/benchmark")
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
    # 'hanoi': 'hanoi',
    # 'n_puzzle_typed': 'npuzzle',
    # 'maze': 'maze',
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
CONFLICT_SEARCH_TIMEOUTS = [180]  # Time limits in seconds for conflict search (cleaning phase). Can specify multiple values.
PLANNING_TIMEOUT = 60  # Timeout in seconds for planning during evaluation
FLUENT_PATCH_COST = 1.0
FLUENT_PATCH_WEIGHT = 1.0
MODEL_PATCH_COST = 1.0
MODEL_CONSTRAINT_WEIGHT = 0.0
MAX_SEARCH_NODES = None

metric_cols = [
    "precision_precs_pos", "precision_precs_neg", "precision_eff_pos", "precision_eff_neg", "precision_overall",
    "recall_precs_pos", "recall_precs_neg", "recall_eff_pos", "recall_eff_neg", "recall_overall",
    "problems_count", "solving_ratio", "false_plans_ratio", "unsolvable_ratio", "planning_timed_out_ratio",
    "pred_app_precision", "pred_app_recall", "pred_eff_precision", "pred_eff_recall",
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def convert_pddl_hyphens_to_underscores(pddl_file_path: Path) -> None:
    """
    Convert all PDDL identifiers (object names, predicate names) from hyphens to underscores.

    This function works for ANY PDDL problem file, regardless of domain.
    It replaces all hyphens with underscores in PDDL identifiers while preserving:
    - Comments
    - Strings
    - PDDL keywords (define, domain, problem, :objects, :init, :goal, etc.)

    Examples of conversions:
    - Object names: player-1 → player_1, loc-3-4 → loc_3_4, disc-1 → disc_1
    - Predicate names: move-dir-up → move_dir_up, on-table → on_table, is-goal → is_goal
    - Any identifier: my-custom-object → my_custom_object

    Args:
        pddl_file_path: Path to the PDDL problem file to convert

    Returns:
        None (modifies file in-place)
    """
    import re

    # Read the file
    with open(pddl_file_path, 'r') as f:
        content = f.read()

    # Replace all hyphens with underscores in PDDL identifiers
    # Pattern explanation:
    # - \b([a-zA-Z][a-zA-Z0-9_-]*-[a-zA-Z0-9_-]*)\b matches any word that:
    #   * Starts with a letter (PDDL requirement)
    #   * Contains at least one hyphen
    #   * May contain letters, digits, underscores, and hyphens
    #   * Is bounded by word boundaries (spaces, parens, etc.)
    # - Lambda function replaces all hyphens in the matched identifier with underscores
    content = re.sub(
        r'\b([a-zA-Z][a-zA-Z0-9_-]*-[a-zA-Z0-9_-]*)\b',
        lambda m: m.group(1).replace('-', '_'),
        content
    )

    # Write back to the same file
    with open(pddl_file_path, 'w') as f:
        f.write(content)



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

    import time
    import random

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


def format_mean_std(mean_val, std_val) -> str:
    """Format value as mean±std."""
    if mean_val is None or pd.isna(mean_val):
        return ""

    # Handle non-numeric values
    if isinstance(mean_val, str):
        return mean_val

    if std_val is None or pd.isna(std_val):
        return f"{mean_val:.3f}"

    # Handle non-numeric std_val
    if isinstance(std_val, str):
        return f"{mean_val:.3f}"

    return f"{mean_val:.3f}±{std_val:.3f}"




def generate_excel_report(unclean_results: List[dict], cleaned_results: List[dict], output_path: Path):
    """Generate Excel report with aggregated results for both unclean and cleaned trajectories."""
    if not unclean_results and not cleaned_results:
        return

    # Define metric groups for Excel table structure
    precision_metrics = ["precision_precs_pos", "precision_precs_neg", "precision_eff_pos", "precision_eff_neg", "precision_overall"]
    recall_metrics = ["recall_precs_pos", "recall_precs_neg", "recall_eff_pos", "recall_eff_neg", "recall_overall"]
    problem_metrics = ["problems_count", "solving_ratio", "false_plans_ratio", "unsolvable_ratio", "planning_timed_out_ratio"]
    predictive_metrics = ["pred_app_precision", "pred_app_recall", "pred_eff_precision", "pred_eff_recall"]

    # Process both result sets and combine with phase labels
    all_results_with_phase = []

    if unclean_results:
        df_unclean = pd.DataFrame(unclean_results)
        grouped_unclean = df_unclean.groupby(["domain", "algorithm", "num_trajectories", "gt_rate"])[metric_cols].agg(["mean", "std"]).reset_index()

        flat_cols = []
        for col in grouped_unclean.columns:
            if isinstance(col, tuple):
                base, stat = col
                flat_cols.append(base if stat == "" else f"{base}_{stat}")
            else:
                flat_cols.append(col)
        grouped_unclean.columns = flat_cols

        df_avg_unclean = grouped_unclean[["domain", "algorithm", "num_trajectories", "gt_rate"]].copy()
        for m in metric_cols:
            df_avg_unclean[m] = grouped_unclean[f"{m}_mean"]
            df_avg_unclean[f"{m}_std"] = grouped_unclean[f"{m}_std"]
        df_avg_unclean["_phase"] = "unclean"

        all_results_with_phase.append(df_avg_unclean)

    if cleaned_results:
        df_cleaned = pd.DataFrame(cleaned_results)
        grouped_cleaned = df_cleaned.groupby(["domain", "algorithm", "num_trajectories", "gt_rate"])[metric_cols].agg(["mean", "std"]).reset_index()

        flat_cols = []
        for col in grouped_cleaned.columns:
            if isinstance(col, tuple):
                base, stat = col
                flat_cols.append(base if stat == "" else f"{base}_{stat}")
            else:
                flat_cols.append(col)
        grouped_cleaned.columns = flat_cols

        df_avg_cleaned = grouped_cleaned[["domain", "algorithm", "num_trajectories", "gt_rate"]].copy()
        for m in metric_cols:
            df_avg_cleaned[m] = grouped_cleaned[f"{m}_mean"]
            df_avg_cleaned[f"{m}_std"] = grouped_cleaned[f"{m}_std"]
        df_avg_cleaned["_phase"] = "cleaned"

        all_results_with_phase.append(df_avg_cleaned)

    df_avg = pd.concat(all_results_with_phase, ignore_index=True)

    # Group by (num_trajectories, gt_rate, phase)
    by_config = defaultdict(list)
    for _, row in df_avg.iterrows():
        phase = row["_phase"]
        config_key = f"numtrajs{int(row['num_trajectories'])}_gtrate{int(row['gt_rate'])}__{'unclean' if phase == 'unclean' else ''}"
        by_config[config_key].append(row.to_dict())

    def clean_excel_value(v):
        if v is None or pd.isna(v):
            return ""
        if isinstance(v, float) and (v == float("inf") or v == float("-inf")):
            return ""
        return v

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        workbook = writer.book
        thin_border = workbook.add_format({"border": 1})
        thick_left = workbook.add_format({"border": 1, "left": 2})
        thick_right = workbook.add_format({"border": 1, "right": 2})

        # Sort sheet names: numtrajs1_gtrate0__unclean, numtrajs1_gtrate0, numtrajs1_gtrate10__unclean, ...
        def sort_key(key):
            parts = key.split('__')
            config = parts[0]  # e.g., "numtrajs1_gtrate0"
            phase = 0 if len(parts) > 1 and parts[1] == 'unclean' else 1

            # Extract num_trajectories and gt_rate from config
            numtrajs_part = config.split('_')[0]  # "numtrajs1"
            gtrate_part = config.split('_')[1]    # "gtrate0"
            num_trajs = int(numtrajs_part.replace('numtrajs', ''))
            gt_rate = int(gtrate_part.replace('gtrate', ''))

            return (num_trajs, gt_rate, phase)

        for config_key in sorted(by_config.keys(), key=sort_key):
            results = by_config[config_key]
            # Sheet name: "numtrajs1_gtrate0__unclean" or "numtrajs1_gtrate0" (for cleaned)
            if config_key.endswith('__unclean'):
                sheet_name = config_key
            else:
                sheet_name = config_key.split('__')[0]  # Remove trailing "__"

            sheet = workbook.add_worksheet(sheet_name)
            writer.sheets[sheet_name] = sheet

            domains = sorted({r["domain"] for r in results})
            algorithms = sorted({r["algorithm"] for r in results})

            # Map (domain, algorithm) -> result dict
            res_map = {(r["domain"], r["algorithm"]): r for r in results}

            def write_syn_table(start_row):
                """Write syntactic P/R table with mean±std values."""
                row0, row1, row2 = start_row, start_row + 1, start_row + 2

                sheet.write(row0, 0, "", thin_border)
                sheet.write(row1, 0, "", thin_border)
                sheet.write(row2, 0, "Domain", thin_border)

                col = 1
                type_spans, metric_spans = {}, {}

                for t, metrics in [("Precision", precision_metrics), ("Recall", recall_metrics)]:
                    type_start = col
                    for m in metrics:
                        metric_start = col
                        for _alg in algorithms:
                            col += 1
                        metric_end = col - 1
                        metric_spans[(t, m)] = (metric_start, metric_end)
                    type_end = col - 1
                    type_spans[t] = (type_start, type_end)

                # Write merged headers
                for t, (c_start, c_end) in type_spans.items():
                    sheet.merge_range(row0, c_start, row0, c_end, t, thin_border)
                for (t, m), (c_start, c_end) in metric_spans.items():
                    sheet.merge_range(row1, c_start, row1, c_end, m, thin_border)

                # Write algorithm names
                col_ptr = 1
                for t, metrics in [("Precision", precision_metrics), ("Recall", recall_metrics)]:
                    for m in metrics:
                        for alg in algorithms:
                            sheet.write(row2, col_ptr, alg, thin_border)
                            col_ptr += 1

                # Write data rows with mean±std format
                for i, dom in enumerate(domains):
                    r_idx = row2 + 1 + i
                    sheet.write(r_idx, 0, dom, thin_border)
                    c = 1
                    for t, metrics in [("Precision", precision_metrics), ("Recall", recall_metrics)]:
                        for m in metrics:
                            for alg in algorithms:
                                res = res_map.get((dom, alg), {})
                                mean_val = clean_excel_value(res.get(m))
                                std_val = clean_excel_value(res.get(f"{m}_std"))
                                formatted = format_mean_std(mean_val, std_val)
                                sheet.write(r_idx, c, formatted, thin_border)
                                c += 1

                first_row, last_row, last_col = row0, row2 + len(domains), col - 1

                # Thick borders between metric groups
                for (_t, _m), (start_c, end_c) in metric_spans.items():
                    sheet.conditional_format(
                        first_row, start_c, last_row, start_c,
                        {"type": "formula", "criteria": "TRUE", "format": thick_left},
                    )
                    sheet.conditional_format(
                        first_row, end_c, last_row, end_c,
                        {"type": "formula", "criteria": "TRUE", "format": thick_right},
                    )

                return first_row, last_row, last_col

            def write_prob_table(start_row):
                """Write problem-solving table with mean±std values."""
                row0, row1, row2 = start_row, start_row + 1, start_row + 2

                sheet.write(row0, 0, "", thin_border)
                sheet.write(row1, 0, "", thin_border)
                sheet.write(row2, 0, "Domain", thin_border)

                col = 1
                metric_spans = {}
                group_start = col

                for m in problem_metrics:
                    metric_start = col
                    for _alg in algorithms:
                        col += 1
                    metric_end = col - 1
                    metric_spans[m] = (metric_start, metric_end)
                group_end = col - 1

                sheet.merge_range(row0, group_start, row0, group_end, "ProblemSolving", thin_border)

                for m, (c_start, c_end) in metric_spans.items():
                    sheet.merge_range(row1, c_start, row1, c_end, m, thin_border)

                col_ptr = 1
                for m in problem_metrics:
                    for alg in algorithms:
                        sheet.write(row2, col_ptr, alg, thin_border)
                        col_ptr += 1

                # Write data rows with mean±std format
                for i, dom in enumerate(domains):
                    r_idx = row2 + 1 + i
                    sheet.write(r_idx, 0, dom, thin_border)
                    c = 1
                    for m in problem_metrics:
                        for alg in algorithms:
                            res = res_map.get((dom, alg), {})
                            mean_val = clean_excel_value(res.get(m))
                            std_val = clean_excel_value(res.get(f"{m}_std"))
                            formatted = format_mean_std(mean_val, std_val)
                            sheet.write(r_idx, c, formatted, thin_border)
                            c += 1

                first_row, last_row, last_col = row0, row2 + len(domains), col - 1

                # Thick borders between metrics
                for _m, (start_c, end_c) in metric_spans.items():
                    sheet.conditional_format(
                        first_row, start_c, last_row, start_c,
                        {"type": "formula", "criteria": "TRUE", "format": thick_left},
                    )
                    sheet.conditional_format(
                        first_row, end_c, last_row, end_c,
                        {"type": "formula", "criteria": "TRUE", "format": thick_right},
                    )

                return first_row, last_row, last_col

            def write_pred_table(start_row):
                """Write predictive power table with mean±std values."""
                row0, row1, row2 = start_row, start_row + 1, start_row + 2

                sheet.write(row0, 0, "", thin_border)
                sheet.write(row1, 0, "", thin_border)
                sheet.write(row2, 0, "Domain", thin_border)

                col = 1
                metric_spans = {}
                group_start = col

                for m in predictive_metrics:
                    metric_start = col
                    for _alg in algorithms:
                        col += 1
                    metric_end = col - 1
                    metric_spans[m] = (metric_start, metric_end)
                group_end = col - 1

                sheet.merge_range(row0, group_start, row0, group_end, "PredictivePower", thin_border)

                for m, (c_start, c_end) in metric_spans.items():
                    sheet.merge_range(row1, c_start, row1, c_end, m, thin_border)

                col_ptr = 1
                for m in predictive_metrics:
                    for alg in algorithms:
                        sheet.write(row2, col_ptr, alg, thin_border)
                        col_ptr += 1

                # Write data rows with mean±std format
                for i, dom in enumerate(domains):
                    r_idx = row2 + 1 + i
                    sheet.write(r_idx, 0, dom, thin_border)
                    c = 1
                    for m in predictive_metrics:
                        for alg in algorithms:
                            res = res_map.get((dom, alg), {})
                            mean_val = clean_excel_value(res.get(m))
                            std_val = clean_excel_value(res.get(f"{m}_std"))
                            formatted = format_mean_std(mean_val, std_val)
                            sheet.write(r_idx, c, formatted, thin_border)
                            c += 1

                first_row, last_row, last_col = row0, row2 + len(domains), col - 1

                # Thick borders between metrics
                for _m, (start_c, end_c) in metric_spans.items():
                    sheet.conditional_format(
                        first_row, start_c, last_row, start_c,
                        {"type": "formula", "criteria": "TRUE", "format": thick_left},
                    )
                    sheet.conditional_format(
                        first_row, end_c, last_row, end_c,
                        {"type": "formula", "criteria": "TRUE", "format": thick_right},
                    )

                return first_row, last_row, last_col

            # Generate tables
            syn_first, syn_last, syn_last_col = write_syn_table(start_row=0)
            gap = 5
            prob_start = syn_last + 1 + gap
            prob_first, prob_last, prob_last_col = write_prob_table(start_row=prob_start)
            pred_start = prob_last + 1 + gap
            pred_first, pred_last, pred_last_col = write_pred_table(start_row=pred_start)


def generate_plots(unclean_results: List[dict], cleaned_results: List[dict], plots_dir: Path):
    """Generate plots per domain comparing unclean vs cleaned trajectories."""
    plots_dir.mkdir(exist_ok=True)

    def plot_metric_vs_num_trajectories(df, metric_key, metric_title, save_path, phase_label, domain_label):
        """Plot metric vs number of trajectories with error bars."""
        if df.empty:
            return

        plt.figure(figsize=(8, 5))

        algorithms = sorted(df["algorithm"].unique())
        for algo in algorithms:
            sub = df[df["algorithm"] == algo].sort_values("num_trajectories")
            x = sub["num_trajectories"]
            y = sub[metric_key]
            yerr = sub[f"{metric_key}_std"] if f"{metric_key}_std" in sub.columns else None

            plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4, label=algo)

        plt.title(f"{metric_title} vs Number of Trajectories ({phase_label} - {domain_label})")
        plt.xlabel("Number of Trajectories")
        plt.ylabel(metric_title)

        # Set x-axis ticks: 1, 2, 3, 4, 5
        plt.xticks([1, 2, 3, 4, 5])

        # Set y-axis ticks: bins of 0.1 (0, 0.1, ..., 1.0)
        plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    # Get all unique domains
    all_domains = set()
    if unclean_results:
        all_domains.update(r['domain'] for r in unclean_results)
    if cleaned_results:
        all_domains.update(r['domain'] for r in cleaned_results)

    # Generate plots for each domain
    for domain in sorted(all_domains):
        domain_upper = domain.upper()

        # Process unclean results for this domain
        if unclean_results:
            domain_unclean = [r for r in unclean_results if r['domain'] == domain]
            if domain_unclean:
                df_unclean = pd.DataFrame(domain_unclean)
                grouped_unclean = df_unclean.groupby(["algorithm", "num_trajectories"])[metric_cols].agg(["mean", "std"]).reset_index()

                flat_cols = []
                for col in grouped_unclean.columns:
                    if isinstance(col, tuple):
                        base, stat = col
                        flat_cols.append(base if stat == "" else f"{base}_{stat}")
                    else:
                        flat_cols.append(col)
                grouped_unclean.columns = flat_cols

                df_avg_unclean = grouped_unclean[["algorithm", "num_trajectories"]].copy()
                for m in metric_cols:
                    df_avg_unclean[m] = grouped_unclean[f"{m}_mean"]
                    df_avg_unclean[f"{m}_std"] = grouped_unclean[f"{m}_std"]

                plot_metric_vs_num_trajectories(df_avg_unclean, "solving_ratio", "Solving Ratio",
                                   plots_dir / f"solving_ratio_vs_num_trajectories__unclean_({domain_upper}).png",
                                   "Unclean", domain_upper)
                plot_metric_vs_num_trajectories(df_avg_unclean, "false_plans_ratio", "False Plan Ratio",
                                   plots_dir / f"false_plans_ratio_vs_num_trajectories__unclean_({domain_upper}).png",
                                   "Unclean", domain_upper)
                plot_metric_vs_num_trajectories(df_avg_unclean, "unsolvable_ratio", "Unsolvable Ratio",
                                   plots_dir / f"unsolvable_ratio_vs_num_trajectories__unclean_({domain_upper}).png",
                                   "Unclean", domain_upper)

        # Process cleaned results for this domain
        if cleaned_results:
            domain_cleaned = [r for r in cleaned_results if r['domain'] == domain]
            if domain_cleaned:
                df_cleaned = pd.DataFrame(domain_cleaned)
                grouped_cleaned = df_cleaned.groupby(["algorithm", "num_trajectories"])[metric_cols].agg(["mean", "std"]).reset_index()

                flat_cols = []
                for col in grouped_cleaned.columns:
                    if isinstance(col, tuple):
                        base, stat = col
                        flat_cols.append(base if stat == "" else f"{base}_{stat}")
                    else:
                        flat_cols.append(col)
                grouped_cleaned.columns = flat_cols

                df_avg_cleaned = grouped_cleaned[["algorithm", "num_trajectories"]].copy()
                for m in metric_cols:
                    df_avg_cleaned[m] = grouped_cleaned[f"{m}_mean"]
                    df_avg_cleaned[f"{m}_std"] = grouped_cleaned[f"{m}_std"]

                plot_metric_vs_num_trajectories(df_avg_cleaned, "solving_ratio", "Solving Ratio",
                                   plots_dir / f"solving_ratio_vs_num_trajectories_({domain_upper}).png",
                                   "Cleaned", domain_upper)
                plot_metric_vs_num_trajectories(df_avg_cleaned, "false_plans_ratio", "False Plan Ratio",
                                   plots_dir / f"false_plans_ratio_vs_num_trajectories_({domain_upper}).png",
                                   "Cleaned", domain_upper)
                plot_metric_vs_num_trajectories(df_avg_cleaned, "unsolvable_ratio", "Unsolvable Ratio",
                                   plots_dir / f"unsolvable_ratio_vs_num_trajectories_({domain_upper}).png",
                                   "Cleaned", domain_upper)


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
    inner_cv: int = 1,
):
    """
    Run benchmark experiments.

    Args:
        selected_domains: List of domain names to run, or None for all domains in domain_name_mappings
        mode: Either 'masked' or 'fullyobs'
    """
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
        "conflict_search_timeouts": CONFLICT_SEARCH_TIMEOUTS,
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
        "inner_cv": inner_cv,
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
            print(f"Conflict search timeouts: {CONFLICT_SEARCH_TIMEOUTS}")
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
            pregenerate_all_gt_frame_axiom_files(
                problem_dirs, domain_ref_path, GT_RATE_PERCENTAGES, FRAME_AXIOM_MODE
            )

            # NEW: Iterate over number of trajectories instead of trajectory sizes
            for num_trajectories in NUM_TRAJECTORIES_LIST:
                print(f"\n{'='*60}\nNUMBER OF TRAJECTORIES = {num_trajectories}\n{'='*60}")

                for gt_rate in GT_RATE_PERCENTAGES:
                    gt_info = f"GT rate: {gt_rate}%" if gt_rate > 0 else "Baseline (GT only at t=0)"
                    print(f"\n{'-'*60}\n{gt_info}\n{'-'*60}")

                    for conflict_timeout in CONFLICT_SEARCH_TIMEOUTS:
                        timeout_info = f"Conflict search timeout: {conflict_timeout}s" if conflict_timeout else "No timeout"
                        print(f"\n{'-'*40}\n{timeout_info}\n{'-'*40}")

                        # Determine effective inner CV for this num_trajectories
                        n_train = max(1, min(int(0.8 * n_problems), n_problems - 1))
                        effective_inner_cv = inner_cv
                        if inner_cv > 1 and num_trajectories >= n_train:
                            print(f"  [INNER-CV] num_trajectories ({num_trajectories}) >= "
                                  f"train pool ({n_train}), all samples identical — "
                                  f"falling back to 1 inner fold")
                            effective_inner_cv = 1

                        # Run all folds (× inner CV sub-folds) in parallel
                        n_total_jobs = N_FOLDS * effective_inner_cv
                        inner_cv_tag = f" × {effective_inner_cv} inner CV" if effective_inner_cv > 1 else ""
                        print(f"  [MAIN] Starting {n_total_jobs} jobs ({N_FOLDS} folds{inner_cv_tag})...")
                        with ProcessPoolExecutor(max_workers=N_FOLDS) as executor:
                            futures = []
                            for fold in range(N_FOLDS):
                                for inner_idx in range(effective_inner_cv):
                                    inner_fold_idx = inner_idx if effective_inner_cv > 1 else None
                                    traj_seed = (42 + fold * 1000 + inner_idx) if effective_inner_cv > 1 else None
                                    future = executor.submit(
                                        run_single_fold,
                                        fold, problem_dirs, n_problems, num_trajectories,
                                        gt_rate, domain_ref_path, testing_dir, bench_name, mode,
                                        evaluate_model, save_learning_metrics,
                                        conflict_timeout, planning_timeout_seconds,
                                        fluent_patch_cost, fluent_patch_weight,
                                        model_patch_cost, model_constraint_weight,
                                        max_search_nodes,
                                        search_mode,
                                        node_choosing_strategy,
                                        conflict_group_strategy,
                                        fluent_branch_mode,
                                        _inner_fold_idx=inner_fold_idx,
                                        _trajectory_seed=traj_seed,
                                    )
                                    futures.append(future)

                            print(f"  [MAIN] All {n_total_jobs} fold tasks submitted, waiting for completion...")

                            # Wait for all jobs to complete and collect results
                            completed_count = 0
                            completed_folds = set()
                            import time
                            start_time = time.time()
                            per_job_timeout = 1800
                            n_waves = -(-n_total_jobs // N_FOLDS)  # ceil division
                            batch_timeout = per_job_timeout * n_waves

                            for future in as_completed(futures, timeout=batch_timeout):
                                try:
                                    completed_count += 1
                                    elapsed = time.time() - start_time
                                    print(f"  [MAIN] Job {completed_count}/{n_total_jobs} completed after {elapsed:.1f}s, collecting results...")
                                    results_list = future.result(timeout=per_job_timeout)

                                    # Identify which fold this was from the results
                                    fold_num = results_list[0]['fold'] if results_list else '?'
                                    inner_id = results_list[0].get('inner_fold_idx', '') if results_list else ''
                                    completed_folds.add(fold_num)

                                    # Separate by phase
                                    for result in results_list:
                                        phase = result['_internal_phase']
                                        if phase == 'unclean':
                                            unclean_results.append(result)
                                        else:
                                            cleaned_results.append(result)

                                    inner_info = f" inner={inner_id}" if inner_id != '' else ""
                                    print(f"  [MAIN] Fold {fold_num}{inner_info} results processed. "
                                          f"Jobs done: {completed_count}/{n_total_jobs}")
                                except TimeoutError:
                                    print(f"TIMEOUT: Job {completed_count} exceeded time limit")
                                    print(f"  Completed so far: {completed_count}/{n_total_jobs}")
                                except Exception as e:
                                    print(f"ERROR in job {completed_count}: {e}")
                                    import traceback
                                    traceback.print_exc()

                        print(f"✓ All {n_total_jobs} jobs for num_trajectories={num_trajectories}, "
                              f"gt_rate={gt_rate}%, timeout={conflict_timeout}s completed")

                        # Write TWO separate CSV files after each timeout completes
                        timeout_suffix = f"_timeout{conflict_timeout}s" if conflict_timeout else "_notimeout"
                        csv_unclean = evaluation_results_dir / f"results_{bench_name}_unclean{timeout_suffix}.csv"
                        csv_cleaned = evaluation_results_dir / f"results_{bench_name}{timeout_suffix}.csv"

                        pd.DataFrame(unclean_results).to_csv(csv_unclean, index=False)
                        pd.DataFrame(cleaned_results).to_csv(csv_cleaned, index=False)

                        # Create combined CSV (unclean + cleaned results)
                        csv_combined = evaluation_results_dir / f"results_{bench_name}_combined{timeout_suffix}.csv"

                        # Filter results for this domain
                        domain_results = [r for r in unclean_results + cleaned_results if r['domain'] == bench_name]
                        pd.DataFrame(domain_results).to_csv(csv_combined, index=False)

                        print(f"\n✓ Results for timeout={conflict_timeout}s written:")
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

# =============================================================================
# PLOTTING FUNCTIONS FOR GT INJECTION ANALYSIS
# =============================================================================

def plot_metric_vs_num_trajectories_by_gt_rate(results_df, metric_name, output_dir, domain_name):
    """
    Impact of Number of Trajectories (Figure Type 2)

    Creates subplots for each GT rate (0%, 10%, 25%, 50%).
    X-axis: Number of trajectories (1, 2, 3, 4, 5)
    Y-axis: Performance metric
    Each subplot shows both cleaned and unclean algorithms for comparison.

    Args:
        results_df: DataFrame with columns: algorithm, num_trajectories, gt_rate, fold, {metric_name}, _internal_phase
        metric_name: Name of metric to plot
        output_dir: Directory to save plots
        domain_name: Name of domain for title
    """

    # Define jitter and style mapping
    jitter_config = {
        ('unclean', 'PISAM'): {'h': -0.03, 'v': 0.005},
        ('cleaned', 'PISAM'): {'h': 0.01, 'v': -0.005},
        ('unclean', 'PO_ROSAME'): {'h': -0.01, 'v': 0.01},
        ('cleaned', 'PO_ROSAME'): {'h': 0.03, 'v': -0.01},
    }
    # Color mapping for each algorithm-phase combination (4 distinct colors)
    color_map = {
        ('PISAM', 'unclean'): '#E74C3C',      # Red
        ('PO_ROSAME', 'unclean'): '#9B59B6',  # Purple
        ('PISAM', 'cleaned'): '#3498DB',      # Blue
        ('PO_ROSAME', 'cleaned'): '#2ECC71',  # Green
    }
    algo_style_map = {
        'PISAM': {'linestyle': {'unclean': '--', 'cleaned': '-'}},
        'PO_ROSAME': {'linestyle': {'unclean': '--', 'cleaned': '-'}},
    }

    gt_rates = sorted(results_df['gt_rate'].unique())
    num_plots = len(gt_rates)
    nrows, ncols = (2, 2) if num_plots <= 4 else (2, 3)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, gt_rate in enumerate(gt_rates):
        ax = axes[idx]
        rate_df = results_df[results_df['gt_rate'] == gt_rate].copy()

        for phase in ['unclean', 'cleaned']:
            phase_df = rate_df[rate_df['_internal_phase'] == phase]
            for algo in sorted(phase_df['algorithm'].unique()):
                if algo not in algo_style_map: continue

                j = jitter_config.get((phase, algo), {'h': 0, 'v': 0})
                algo_df = phase_df[phase_df['algorithm'] == algo]
                grouped = algo_df.groupby('num_trajectories')[metric_name].agg(['mean', 'std'])
                if len(grouped) == 0: continue

                num_trajs = sorted(grouped.index)
                jx = [nt + j['h'] for nt in num_trajs]
                original_y = [grouped.loc[nt, 'mean'] for nt in num_trajs]
                jy = [y + j['v'] for y in original_y]

                style = algo_style_map[algo]
                # Get distinct color for this algorithm-phase combination
                plot_color = color_map.get((algo, phase), 'black')
                # Use circle marker for unclean, square marker for cleaned
                marker = 'o' if phase == 'unclean' else 's'
                ax.plot(jx, jy, marker=marker, label=f"{algo} ({phase})",
                        color=plot_color, linestyle=style['linestyle'][phase], linewidth=2)

                # ADD DATA LABELS (Black text)
                for x_val, y_orig, y_jit in zip(jx, original_y, jy):
                    ax.text(x_val + 0.05, y_jit, f"{y_orig:.2f}",
                            color='black', fontsize=8, fontweight='bold',
                            va='center', ha='left')

        ax.set_ylim(-0.1, 1.2)  # Increased upper limit slightly for labels
        ax.set_title('Baseline (gt = only init state)' if gt_rate == 0 else f'GT Rate {gt_rate}%', fontweight='bold')
        ax.set_xlabel('Number of Trajectories', fontsize=12)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add orange border to 100% GT rate subplot
        if gt_rate == 100:
            for spine in ax.spines.values():
                spine.set_edgecolor('orange')
                spine.set_linewidth(3)
        
        # Create custom legend with 5 entries: PISAM unclean, PO_ROSAME unclean, PISAM cleaned, PO_ROSAME cleaned, 100% GT Rate
        from matplotlib.lines import Line2D
        custom_handles = []
        
        # PISAM unclean - dashed red with circle marker
        pisam_unclean_color = color_map[('PISAM', 'unclean')]
        line1 = Line2D([0, 1], [0, 0], color=pisam_unclean_color, linestyle='--', linewidth=2, marker='o', 
                      markersize=8, markevery=[0], label='PISAM unclean')
        line1.set_dashes([5, 2])  # Explicitly set dash pattern
        custom_handles.append(line1)
        # PO_ROSAME unclean - dashed purple with circle marker
        rosame_unclean_color = color_map[('PO_ROSAME', 'unclean')]
        line2 = Line2D([0, 1], [0, 0], color=rosame_unclean_color, linestyle='--', linewidth=2, marker='o', 
                      markersize=8, markevery=[0], label='PO_ROSAME unclean')
        line2.set_dashes([5, 2])  # Explicitly set dash pattern
        custom_handles.append(line2)
        # PISAM cleaned - solid blue with square marker
        pisam_cleaned_color = color_map[('PISAM', 'cleaned')]
        custom_handles.append(Line2D([0, 1], [0, 0], color=pisam_cleaned_color, linestyle='-', linewidth=2, marker='s', 
                                    markersize=7, markevery=[0], label='PISAM cleaned'))
        # PO_ROSAME cleaned - solid green with square marker
        rosame_cleaned_color = color_map[('PO_ROSAME', 'cleaned')]
        custom_handles.append(Line2D([0, 1], [0, 0], color=rosame_cleaned_color, linestyle='-', linewidth=2, marker='s', 
                                    markersize=7, markevery=[0], label='PO_ROSAME cleaned'))
        
        ax.legend(handles=custom_handles, fontsize=9, loc='lower right')

    for idx in range(len(gt_rates), len(axes)): axes[idx].set_visible(False)
    fig.suptitle(f'{domain_name.upper()}: Impact of Trajectories on {metric_name.replace("_", " ").title()}',
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(output_dir / f"{domain_name}_{metric_name}_vs_num_trajectories_by_gt_rate.png", dpi=300)
    plt.close()


def plot_metric_vs_gt_rate_by_num_trajectories(results_df, metric_name, output_dir, domain_name):
    """
    Impact of GT Rate (Figure Type 1)

    Creates subplots for all available numbers of trajectories.
    X-axis: GT rate (0%, 10%, 25%, 50%, etc.)
    Y-axis: Performance metric
    Each subplot shows both cleaned and unclean algorithms for comparison.

    Args:
        results_df: DataFrame with columns: algorithm, num_trajectories, gt_rate, fold, {metric_name}, _internal_phase
        metric_name: Name of metric to plot
        output_dir: Directory to save plots
        domain_name: Name of domain for title
    """
    jitter_config = {
        ('unclean', 'PISAM'): {'h': -0.4, 'v': 0.005},
        ('cleaned', 'PISAM'): {'h': 0.1, 'v': -0.005},
        ('unclean', 'PO_ROSAME'): {'h': -0.1, 'v': 0.01},
        ('cleaned', 'PO_ROSAME'): {'h': 0.4, 'v': -0.01},
    }
    # Color mapping for each algorithm-phase combination (4 distinct colors)
    color_map = {
        ('PISAM', 'unclean'): '#E74C3C',      # Red
        ('PO_ROSAME', 'unclean'): '#9B59B6',  # Purple
        ('PISAM', 'cleaned'): '#3498DB',      # Blue
        ('PO_ROSAME', 'cleaned'): '#2ECC71',  # Green
    }
    algo_style_map = {
        'PISAM': {'linestyle': {'unclean': '--', 'cleaned': '-'}},
        'PO_ROSAME': {'linestyle': {'unclean': '--', 'cleaned': '-'}},
    }

    # Show all available trajectory numbers from the data
    available_nums = sorted(results_df['num_trajectories'].unique())
    num_plots = len(available_nums)
    
    # Determine layout: one subplot per trajectory number
    if num_plots <= 4:
        nrows, ncols = (2, 2)
    elif num_plots <= 6:
        nrows, ncols = (2, 3)
    elif num_plots <= 8:
        nrows, ncols = (2, 4)
    elif num_plots <= 9:
        nrows, ncols = (3, 3)
    else:
        nrows, ncols = (3, 4)  # Can handle up to 12
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, num_traj in enumerate(available_nums):
        ax = axes[idx]
        num_df = results_df[results_df['num_trajectories'] == num_traj].copy()

        # Collect all GT rates that will be plotted (before jitter)
        all_gt_rates = sorted(num_df['gt_rate'].unique())
        
        for phase in ['unclean', 'cleaned']:
            phase_df = num_df[num_df['_internal_phase'] == phase]
            for algo in sorted(phase_df['algorithm'].unique()):
                if algo not in algo_style_map: continue
                j = jitter_config.get((phase, algo), {'h': 0, 'v': 0})
                algo_df = phase_df[phase_df['algorithm'] == algo]
                grouped = algo_df.groupby('gt_rate')[metric_name].agg(['mean', 'std'])

                gt_rates = sorted(grouped.index)
                jx = [r + j['h'] for r in gt_rates]
                original_y = [grouped.loc[r, 'mean'] for r in gt_rates]
                jy = [y + j['v'] for y in original_y]

                style = algo_style_map[algo]
                # Get distinct color for this algorithm-phase combination
                plot_color = color_map.get((algo, phase), 'black')
                # Use circle marker for unclean, square marker for cleaned
                marker = 'o' if phase == 'unclean' else 's'
                ax.plot(jx, jy, marker=marker, label=f"{algo} ({phase})", color=plot_color,
                        linestyle=style['linestyle'][phase], linewidth=2)

                # ADD DATA LABELS (Black text)
                for x_val, y_orig, y_jit in zip(jx, original_y, jy):
                    ax.text(x_val + 0.8, y_jit, f"{y_orig:.2f}",
                            color='black', fontsize=8, fontweight='bold',
                            va='center', ha='left')

        # Set x-axis limits and ticks based on actual GT rates (not jittered values)
        if all_gt_rates:
            x_min = min(all_gt_rates) - 5
            x_max = max(all_gt_rates) + 5
            ax.set_xlim(x_min, x_max)
            ax.set_xticks(all_gt_rates)
        
        ax.set_ylim(-0.1, 1.2)
        
        # Check if we should add 100% GT rate line
        add_100_line = False
        if all_gt_rates and (100 in all_gt_rates or max(all_gt_rates) >= 90):
            add_100_line = True
            y_min, y_max = ax.get_ylim()
            ax.axvline(x=100, color='orange', linestyle='-', linewidth=2, alpha=0.7, zorder=0)
        
        ax.set_title(f'Number of Trajectories = {num_traj}', fontweight='bold')
        ax.set_xlabel('GT Rate (%)', fontsize=12)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Create custom legend with 5 entries: PISAM unclean, PO_ROSAME unclean, PISAM cleaned, PO_ROSAME cleaned, 100% GT Rate
        from matplotlib.lines import Line2D
        custom_handles = []
        
        # PISAM unclean - dashed red with circle marker
        pisam_unclean_color = color_map[('PISAM', 'unclean')]
        line1 = Line2D([0, 1], [0, 0], color=pisam_unclean_color, linestyle='--', linewidth=2, marker='o', 
                      markersize=8, markevery=[0], label='PISAM unclean')
        line1.set_dashes([5, 2])  # Explicitly set dash pattern
        custom_handles.append(line1)
        # PO_ROSAME unclean - dashed purple with circle marker
        rosame_unclean_color = color_map[('PO_ROSAME', 'unclean')]
        line2 = Line2D([0, 1], [0, 0], color=rosame_unclean_color, linestyle='--', linewidth=2, marker='o', 
                      markersize=8, markevery=[0], label='PO_ROSAME unclean')
        line2.set_dashes([5, 2])  # Explicitly set dash pattern
        custom_handles.append(line2)
        # PISAM cleaned - solid blue with square marker
        pisam_cleaned_color = color_map[('PISAM', 'cleaned')]
        custom_handles.append(Line2D([0, 1], [0, 0], color=pisam_cleaned_color, linestyle='-', linewidth=2, marker='s', 
                                    markersize=7, markevery=[0], label='PISAM cleaned'))
        # PO_ROSAME cleaned - solid green with square marker
        rosame_cleaned_color = color_map[('PO_ROSAME', 'cleaned')]
        custom_handles.append(Line2D([0, 1], [0, 0], color=rosame_cleaned_color, linestyle='-', linewidth=2, marker='s', 
                                    markersize=7, markevery=[0], label='PO_ROSAME cleaned'))
        
        # Add 100% GT rate line to legend if it was drawn
        if add_100_line:
            custom_handles.append(Line2D([0, 1], [0, 0], color='orange', linestyle='-', linewidth=2, label='100% GT Rate'))
        
        ax.legend(handles=custom_handles, fontsize=9, loc='lower right')

    # Hide extra subplots
    for idx in range(len(available_nums), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f'{domain_name.upper()}: Impact of GT Rate on {metric_name.replace("_", " ").title()}',
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(output_dir / f"{domain_name}_{metric_name}_vs_gt_rate_by_num_trajectories.png", dpi=300)
    plt.close()


def generate_gt_injection_plots(results_csv_path, output_dir, domain_name):
    """
    Generate all GT injection analysis plots for a domain.

    Creates plots for each of the 3 problem-solving metrics:
    - solving_ratio
    - false_plans_ratio
    - unsolvable_ratio

    For each metric, creates:
    1. Metric vs num_trajectories (2x2 grid, one subplot per gt_rate)
    2. Metric vs gt_rate (1x3 grid, one subplot per representative num_trajectories)
    """
    # Load results
    df = pd.read_csv(results_csv_path)

    # Ensure required columns exist
    if 'gt_rate' not in df.columns:
        print(f"Warning: No 'gt_rate' column in {results_csv_path}, skipping GT injection plots")
        return

    # Create plots directory
    plots_dir = output_dir / 'plots' / 'gt_injection'
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating GT injection plots for {domain_name}...")

    # Metrics to plot
    metrics = ['solving_ratio', 'false_plans_ratio', 'unsolvable_ratio']

    for metric in metrics:
        if metric not in df.columns:
            print(f"  Warning: Metric '{metric}' not found in results")
            continue

        print(f"\n  Plotting {metric}...")

        # Figure Type 1: Metric vs num_trajectories (by gt_rate)
        plot_metric_vs_num_trajectories_by_gt_rate(df, metric, plots_dir, domain_name)

        # Figure Type 2: Metric vs gt_rate (by num_trajectories)
        plot_metric_vs_gt_rate_by_num_trajectories(df, metric, plots_dir, domain_name)

    print(f"\n✓ All GT injection plots saved to: {plots_dir}")


def plot_stacked_solving_rate(results_csv_path, output_dir, domain_name):
    """
    Generate stacked area charts showing solving rate over number of trajectories.
    
    Creates two plots (one per phase: unclean/cleaned), each containing subplots for all GT rates.
    Each subplot shows stacked areas for:
    - Solved (solving_ratio) - light yellow
    - Inapplicable (false_plans_ratio) - light blue  
    - Not Solved (unsolvable_ratio) - red
    
    X-axis: Number of trajectories (num_trajectories)
    Y-axis: Solving rate (0-1)
    
    Args:
        results_csv_path: Path to CSV file with results
        output_dir: Directory to save plots
        domain_name: Name of domain for title
    """
    # Load results
    df = pd.read_csv(results_csv_path)
    
    # Ensure required columns exist
    required_cols = ['algorithm', 'num_trajectories', 'gt_rate', 'solving_ratio', 
                     'false_plans_ratio', 'unsolvable_ratio', '_internal_phase']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"  Warning: Missing columns in {results_csv_path}: {missing_cols}, skipping stacked plots")
        return
    
    # Filter for PISAM only
    pisam_df = df[df['algorithm'] == 'PISAM'].copy()
    
    if pisam_df.empty:
        print(f"  Warning: No PISAM data found for {domain_name}, skipping stacked plots")
        return
    
    # Create output directory
    stacked_plots_dir = output_dir / 'plots' / 'stacked_solving_rate'
    stacked_plots_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating stacked solving rate plots for {domain_name}...")
    
    # Color scheme similar to the example image
    colors = {
        'solved': '#FFF4C4',  # Light yellow (similar to "Solved" in example)
        'inapplicable': '#B0E0E6',  # Light blue (similar to "Inapplicable" in example)
        'not_solved': '#FF6B6B'  # Light red (similar to "Not Solved" in example)
    }
    
    # Generate one plot per phase
    for phase in ['unclean', 'cleaned']:
        phase_df = pisam_df[pisam_df['_internal_phase'] == phase].copy()
        
        if phase_df.empty:
            continue
        
        # Get unique GT rates for this phase
        gt_rates = sorted(phase_df['gt_rate'].unique())
        
        if not gt_rates:
            continue
        
        # Determine subplot layout (similar to GT injection plots)
        num_plots = len(gt_rates)
        nrows, ncols = (2, 2) if num_plots <= 4 else (2, 3)
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), sharex=True, sharey=True)
        axes = axes.flatten()
        
        phase_label = 'Cleaned' if phase == 'cleaned' else 'Unclean'
        
        # Create subplot for each GT rate
        for idx, gt_rate in enumerate(gt_rates):
            ax = axes[idx]
            
            # Filter data for this GT rate
            rate_df = phase_df[phase_df['gt_rate'] == gt_rate].copy()
            
            if rate_df.empty:
                continue
            
            # Group by num_trajectories and calculate mean across folds
            grouped = rate_df.groupby('num_trajectories').agg({
                'solving_ratio': 'mean',
                'false_plans_ratio': 'mean',
                'unsolvable_ratio': 'mean'
            }).reset_index()
            
            # Sort by num_trajectories
            grouped = grouped.sort_values('num_trajectories')
            
            # Get x values (num_trajectories)
            x = grouped['num_trajectories'].values
            num_trajs_sorted = sorted(grouped['num_trajectories'].unique())
            
            # Get y values for each category
            solved = grouped['solving_ratio'].values
            inapplicable = grouped['false_plans_ratio'].values
            
            # Adjust not_solved to complement solved + inapplicable to sum to 1.0
            # This ensures the stacked areas always total 1.0, even if data doesn't sum perfectly
            not_solved = 1.0 - (solved + inapplicable)
            # Ensure values are in [0, 1] range
            not_solved = [max(0.0, min(1.0, val)) for val in not_solved]
            
            # Create stacked area chart using stackplot
            # Order: solved (bottom), inapplicable (middle), not_solved (top)
            ax.stackplot(x, solved, inapplicable, not_solved,
                        colors=[colors['solved'], colors['inapplicable'], colors['not_solved']],
                        alpha=0.8, labels=['Solved', 'Inapplicable', 'Not Solved'],
                        edgecolor='white', linewidth=0.5)
            
            # Set subplot title
            subplot_title = 'Baseline (gt = only init state)' if gt_rate == 0 else f'GT Rate {gt_rate}%'
            ax.set_title(subplot_title, fontweight='bold', fontsize=11)
            
            # Set labels
            ax.set_xlabel('Number of Trajectories', fontsize=10)
            ax.set_ylabel('Solving Rate', fontsize=10)
            
            # Set x-axis ticks based on available data
            ax.set_xticks(num_trajs_sorted)
            ax.set_xlim(min(num_trajs_sorted) - 0.5, max(num_trajs_sorted) + 0.5)
            
            # Set y-axis
            ax.set_ylim(0.0, 1.0)
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.3, axis='both')
            
            # Add legend only to first subplot
            if idx == 0:
                ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
        
        # Hide unused subplots
        for idx in range(len(gt_rates), len(axes)):
            axes[idx].set_visible(False)
        
        # Set overall title
        fig.suptitle(f'Solving Rate Over Number of Trajectories - PISAM {phase_label} - {domain_name.upper()}', 
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save plot
        filename = f"{domain_name}_PISAM_{phase}_stacked_solving_rate.png"
        plt.savefig(stacked_plots_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {filename}")
    
    print(f"\n✓ All stacked solving rate plots saved to: {stacked_plots_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run PDDL action model learning benchmark')
    parser.add_argument('--domain', type=str, default='all',
                       help='Domain to run (blocksworld, hanoi, n_puzzle_typed, maze, or "all" for all domains)')
    parser.add_argument('--mode', type=str, default='masked', choices=['masked', 'fullyobs'],
                       help='Mode to run: "masked" (PISAM/PO_ROSAME) or "fullyobs" (SAM/ROSAME)')
    parser.add_argument('--learning-timeout-seconds', type=int, default=CONFLICT_SEARCH_TIMEOUTS[0],
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
    parser.add_argument(
        '--inner-cv',
        type=int,
        default=1,
        help=(
            'Number of random trajectory subgroups per (fold, num_trajectories). '
            'Default 1 = current behavior (single prefix slice). '
            'Values > 1 = each inner fold draws an independent random sample.'
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

    # CLI controls a single timeout value for this run.
    CONFLICT_SEARCH_TIMEOUTS = [args.learning_timeout_seconds]
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
        inner_cv=args.inner_cv,
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