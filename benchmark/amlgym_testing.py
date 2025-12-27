import argparse
import json
from collections import defaultdict
from concurrent.futures import as_completed, ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd
from amlgym.algorithms import *
from amlgym.benchmarks import *
from amlgym.metrics import *

from benchmark.experiment_helpers import run_single_fold

# =============================================================================
# CONFIG & PATHS
# =============================================================================

benchmark_path = Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/benchmark")
print_metrics()

# Global lock for thread-safe evaluation (AMLGym SimpleDomainReader is not thread-safe)
evaluation_lock = Lock()

experiment_data_dirs_masked = {
    "blocksworld": ["multi_problem_04-12-2025T12:00:44__model=gpt-5.1__steps=50__planner"],
    "hanoi": ["multi_problem_06-12-2025T13:58:24__model=gpt-5.1__steps=100__planner"],
    # "hanoi": ["multi_problem_13-12-2025T14:53:55__model=gemini-2.5-pro__steps=11__planner"],
    "n_puzzle_typed": ["multi_problem_06-12-2025T13:32:59__model=gpt-5.1__steps=100__planner"],
    "maze": ["experiment_07-12-2025T16:16:54__model=gpt-5.1__steps=100__planner"],
    # "maze": ["multi_problem_13-12-2025T18:10:23__model=gemini-2.5-pro__steps=100__planner"]
}

experiment_data_dirs_fullyobs = {
    "blocksworld": ["multi_problem_07-12-2025T17:27:33__model=gpt-5.1__steps=100__planner__NO_MASK"],
    "hanoi": ["multi_problem_07-12-2025T17:30:57__model=gpt-5.1__steps=100__planner__NO_MASK"],
    "n_puzzle_typed": ["multi_problem_06-12-2025T13:32:59__model=gpt-5.1__steps=100__planner"],
    "maze": ["multi_problem_07-12-2025T17:37:10__model=gpt-5.1__steps=100__planner__NO_MASK"]
}

domain_name_mappings = {
    # 'n_puzzle_typed': 'npuzzle',
    'blocksworld': 'blocksworld',
    # 'hanoi': 'hanoi',
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
NUM_TRAJECTORIES_LIST = [1, 2, 3, 4, 5]  # Number of full trajectories to use for learning
NUM_TRAJECTORIES_POOL = 5  # Total number of trajectories to select per fold
GT_RATE_PERCENTAGES = [0, 10, 25, 50]  # Percentage of states to inject as GT (0 = only initial state)
FRAME_AXIOM_MODE = "after_gt_only"  # "after_gt_only" or "all_states"

metric_cols = [
    "precision_precs_pos", "precision_precs_neg", "precision_eff_pos", "precision_eff_neg", "precision_overall",
    "recall_precs_pos", "recall_precs_neg", "recall_eff_pos", "recall_eff_neg", "recall_overall",
    "problems_count", "solving_ratio", "false_plans_ratio", "unsolvable_ratio", "timed_out",
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
    }

    # Add trajectory mapping if provided
    if trajectory_mapping:
        metrics["trajectory_mapping"] = trajectory_mapping

    with open(output_dir / "learning_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics




def evaluate_model(model_path: str, domain_ref_path: Path, test_problems: List[str]) -> dict:
    """Evaluate a learned model. Handles AMLGym SimpleDomainReader race conditions."""
    # NOTE: evaluation_lock is a threading lock, but we use ProcessPoolExecutor,
    # so it doesn't prevent race conditions across processes.

    import time
    import random

    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Add small random delay to reduce collision probability
            if attempt > 0:
                time.sleep(random.uniform(0.1, 0.5))

            # Run all evaluations together - if any fail, retry all
            precision = syntactic_precision(model_path, str(domain_ref_path))
            recall = syntactic_recall(model_path, str(domain_ref_path))
            problem_solving_result = problem_solving(model_path, str(domain_ref_path), test_problems, timeout=60)

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
        'timed_out': problem_solving_result.get('timed_out') if isinstance(problem_solving_result, dict) else None,
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
    problem_metrics = ["problems_count", "solving_ratio", "false_plans_ratio", "unsolvable_ratio", "timed_out"]

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

            # Generate tables
            syn_first, syn_last, syn_last_col = write_syn_table(start_row=0)
            gap = 5
            prob_start = syn_last + 1 + gap
            prob_first, prob_last, prob_last_col = write_prob_table(start_row=prob_start)


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
def main(selected_domains: List[str] = None, mode: str = 'masked'):
    """
    Run benchmark experiments.

    Args:
        selected_domains: List of domain names to run, or None for all domains in domain_name_mappings
        mode: Either 'masked' or 'fullyobs'
    """
    unclean_results = []
    cleaned_results = []

    # Create evaluation results directory
    evaluation_results_dir = benchmark_path / 'data' / 'evaluation_results'
    evaluation_results_dir.mkdir(parents=True, exist_ok=True)

    # Select appropriate experiment data directories based on mode
    experiment_data_dirs = experiment_data_dirs_masked if mode == 'masked' else experiment_data_dirs_fullyobs

    # Filter domains if specific domains are requested
    if selected_domains:
        domains_to_run = {k: v for k, v in domain_name_mappings.items() if k in selected_domains}
    else:
        domains_to_run = domain_name_mappings

    print(f"\n{'='*80}")
    print(f"RUNNING BENCHMARK IN {mode.upper()} MODE")
    print(f"Domains: {list(domains_to_run.keys())}")
    print(f"{'='*80}\n")

    for domain_name, bench_name in domains_to_run.items():
        domain_ref_path = domain_properties[domain_name]["domain_path"]

        for dir_name in experiment_data_dirs[domain_name]:
            data_dir = benchmark_path / 'data' / domain_name / dir_name
            trajectories_dir = data_dir / 'training' / 'trajectories'
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
            print(f"CV folds: {N_FOLDS}")
            print(f"{'=' * 80}\n")

            # PRE-GENERATE all GT+frame-axiom files before experiments
            from benchmark.experiment_helpers import pregenerate_all_gt_frame_axiom_files
            pregenerate_all_gt_frame_axiom_files(
                problem_dirs, domain_ref_path, GT_RATE_PERCENTAGES, FRAME_AXIOM_MODE
            )

            # NEW: Iterate over number of trajectories instead of trajectory sizes
            for num_trajectories in NUM_TRAJECTORIES_LIST:
                print(f"\n{'='*60}\nNUMBER OF TRAJECTORIES = {num_trajectories}\n{'='*60}")

                for gt_rate in GT_RATE_PERCENTAGES:
                    gt_info = f"GT rate: {gt_rate}%" if gt_rate > 0 else "Baseline (GT only at t=0)"
                    print(f"\n{'-'*60}\n{gt_info}\n{'-'*60}")

                    # Run all folds in parallel for this num_trajectories and gt_rate
                    print(f"  [MAIN] Starting {N_FOLDS} folds in parallel...")
                    with ProcessPoolExecutor(max_workers=N_FOLDS) as executor:
                        futures = []
                        for fold in range(N_FOLDS):
                            future = executor.submit(
                                run_single_fold,
                                fold, problem_dirs, n_problems, num_trajectories,
                                gt_rate, domain_ref_path, testing_dir, bench_name, mode,
                                evaluate_model, save_learning_metrics
                            )
                            futures.append(future)

                        print(f"  [MAIN] All {N_FOLDS} fold tasks submitted, waiting for completion...")

                        # Wait for all folds to complete and collect results
                        completed_count = 0
                        completed_folds = set()
                        import time
                        start_time = time.time()

                        for future in as_completed(futures, timeout=3600):  # 1 hour timeout per fold batch
                            try:
                                completed_count += 1
                                elapsed = time.time() - start_time
                                print(f"  [MAIN] Fold {completed_count}/{N_FOLDS} completed after {elapsed:.1f}s, collecting results...")
                                results_list = future.result(timeout=1800)  # 30 min timeout per fold

                                # Identify which fold this was from the results
                                fold_num = results_list[0]['fold'] if results_list else '?'
                                completed_folds.add(fold_num)

                                # Separate by phase and remove internal marker
                                for result in results_list:
                                    phase = result.pop('_internal_phase')
                                    if phase == 'unclean':
                                        unclean_results.append(result)
                                    else:  # phase == 'cleaned'
                                        cleaned_results.append(result)

                                pending_folds = set(range(N_FOLDS)) - completed_folds
                                print(f"  [MAIN] Fold {fold_num} results processed. Pending: {sorted(pending_folds)}")
                            except TimeoutError:
                                print(f"TIMEOUT: Fold {completed_count} exceeded time limit")
                                print(f"  Completed so far: {completed_count}/{N_FOLDS}")
                                # Continue to wait for other folds
                            except Exception as e:
                                print(f"ERROR in fold {completed_count}: {e}")
                                import traceback
                                traceback.print_exc()

                        print(f"✓ All {N_FOLDS} folds for num_trajectories={num_trajectories}, gt_rate={gt_rate}% completed")

                # Write TWO separate CSV files after all num_trajectories and gt_rate values complete
                csv_unclean = evaluation_results_dir / f"results_{bench_name}_unclean.csv"
                csv_cleaned = evaluation_results_dir / f"results_{bench_name}.csv"

                pd.DataFrame(unclean_results).to_csv(csv_unclean, index=False)
                pd.DataFrame(cleaned_results).to_csv(csv_cleaned, index=False)

                # Create combined CSV (unclean + cleaned results)
                csv_combined = evaluation_results_dir / f"results_{bench_name}_combined.csv"

                # Filter results for this domain
                domain_results = [r for r in unclean_results + cleaned_results if r['domain'] == bench_name]
                pd.DataFrame(domain_results).to_csv(csv_combined, index=False)

                print(f"\n✓ All folds for num_trajectories={num_trajectories} completed")
                print(f"✓ Unclean results written to {csv_unclean}")
                print(f"✓ Cleaned results written to {csv_cleaned}")
                print(f"✓ Combined results written to {csv_combined}")

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

                # Generate plots after each num_trajectories
                plots_dir = evaluation_results_dir / "plots"
                generate_plots(unclean_results, cleaned_results, plots_dir)
                print(f"✓ Plots updated with results up to num_trajectories={num_trajectories}")

    # =============================================================================
    # FINAL SUMMARY
    # =============================================================================

    # Create all-domains combined CSV file
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
    print(f"\nAll evaluation results saved to: {evaluation_results_dir}")
    print(f"  - Plots: {evaluation_results_dir / 'plots'}")
    print(f"  - Excel report: {xlsx_path}")
    print(f"  - Per-domain CSVs: {csv_unclean}, {csv_cleaned}, {csv_combined}")
    print(f"  - All-domains combined CSV: {csv_all_combined}")

# =============================================================================
# PLOTTING FUNCTIONS FOR GT INJECTION ANALYSIS
# =============================================================================

def plot_metric_vs_num_trajectories_by_gt_rate(results_df, metric_name, output_dir, domain_name):
    """
    Figure Type 1: How does number of trajectories affect metrics for each GT rate?

    Creates one figure with 2x2 subplots (one per gt_rate value).
    Each subplot shows metric vs num_trajectories with lines for each algorithm.
    Baseline (gt_rate=0) is shown as horizontal dashed line in each subplot.

    Args:
        results_df: DataFrame with columns: algorithm, num_trajectories, gt_rate, fold, {metric_name}
        metric_name: Name of metric to plot (e.g., 'solving_ratio', 'false_plans_ratio', 'unsolvable_ratio')
        output_dir: Directory to save plots
        domain_name: Name of domain for title
    """

    # Define color map for algorithms (consistent across all plots)
    algo_colors = {
        'PISAM': 'C0',  # blue
        'NOISY_PISAM': 'C1',  # orange
        'ROSAME': 'C2',  # green
        'SAM': 'C0',  # blue (for fullyobs mode)
        'NOISY_SAM': 'C1',  # orange (for fullyobs mode)
    }

    # Filter for cleaned phase only
    if '_internal_phase' in results_df.columns:
        df = results_df[results_df['_internal_phase'] == 'cleaned'].copy()
    elif 'phase' in results_df.columns:
        df = results_df[results_df['phase'] == 'cleaned'].copy()
    else:
        df = results_df.copy()

    # Get all gt_rate values including 0 (sorted)
    gt_rates = sorted(df['gt_rate'].unique())

    if not gt_rates:
        print(f"  Warning: No data found for {metric_name}")
        return

    # Create grid based on number of gt_rate values (default 2x2 for up to 4 values)
    num_plots = len(gt_rates)
    if num_plots <= 4:
        nrows, ncols = 2, 2
    elif num_plots <= 6:
        nrows, ncols = 2, 3
    else:
        nrows, ncols = 3, 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 5*nrows), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, gt_rate in enumerate(gt_rates):
        ax = axes[idx]

        # Plot current gt_rate value as solid lines
        rate_df = df[df['gt_rate'] == gt_rate]
        for algo in rate_df['algorithm'].unique():
            algo_df = rate_df[rate_df['algorithm'] == algo]

            # Group by num_trajectories and compute mean/std
            grouped = algo_df.groupby('num_trajectories')[metric_name].agg(['mean', 'std'])
            num_trajs = sorted(grouped.index)
            means = [grouped.loc[nt, 'mean'] for nt in num_trajs]
            stds = [grouped.loc[nt, 'std'] for nt in num_trajs]

            color = algo_colors.get(algo, 'gray')
            label = f'{algo}' if gt_rate == 0 else f'{algo} (GT rate {gt_rate}%)'
            ax.plot(num_trajs, means, marker='o', label=label,
                   color=color, linewidth=2)
            ax.fill_between(num_trajs,
                           [m - s for m, s in zip(means, stds)],
                           [m + s for m, s in zip(means, stds)],
                           alpha=0.2, color=color)

        title = 'Baseline (GT only at t=0)' if gt_rate == 0 else f'GT Rate {gt_rate}%'
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of Trajectories', fontsize=10)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')

    # Hide unused subplots
    for idx in range(len(gt_rates), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f'{domain_name}: {metric_name.replace("_", " ").title()} vs Number of Trajectories',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / f'{domain_name}_{metric_name}_vs_num_trajectories_by_gt_rate.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved plot: {output_path}")


def plot_metric_vs_gt_rate_by_num_trajectories(results_df, metric_name, output_dir, domain_name):
    """
    Figure Type 2: How does GT rate affect metrics for fixed numbers of trajectories?

    Creates one figure with 1x3 subplots (one per num_trajectories: 1, 3, 5).
    Each subplot shows metric vs gt_rate with lines for each algorithm.

    Args:
        results_df: DataFrame with columns: algorithm, num_trajectories, gt_rate, fold, {metric_name}
        metric_name: Name of metric to plot
        output_dir: Directory to save plots
        domain_name: Name of domain for title
    """
    # Define color map for algorithms
    algo_colors = {
        'PISAM': 'C0',
        'NOISY_PISAM': 'C1',
        'ROSAME': 'C2',
        'SAM': 'C0',
        'NOISY_SAM': 'C1',
    }

    # Filter for cleaned phase only (include all gt_rate values including 0)
    if '_internal_phase' in results_df.columns:
        df = results_df[results_df['_internal_phase'] == 'cleaned'].copy()
    elif 'phase' in results_df.columns:
        df = results_df[results_df['phase'] == 'cleaned'].copy()
    else:
        df = results_df.copy()

    # Representative num_trajectories values
    representative_nums = [1, 3, 5]
    # Filter to available values
    available_nums = [n for n in representative_nums if n in df['num_trajectories'].unique()]

    if not available_nums:
        print(f"  Warning: No representative num_trajectories available for {metric_name}")
        return

    # Create 1x3 grid (or adjust based on available values)
    fig, axes = plt.subplots(1, len(available_nums), figsize=(6*len(available_nums), 5), sharey=True)
    if len(available_nums) == 1:
        axes = [axes]

    for idx, num_traj in enumerate(available_nums):
        ax = axes[idx]

        num_df = df[df['num_trajectories'] == num_traj]

        for algo in num_df['algorithm'].unique():
            algo_df = num_df[num_df['algorithm'] == algo]

            # Group by gt_rate and compute mean/std
            grouped = algo_df.groupby('gt_rate')[metric_name].agg(['mean', 'std'])
            gt_rates = sorted(grouped.index)
            means = [grouped.loc[r, 'mean'] for r in gt_rates]
            stds = [grouped.loc[r, 'std'] for r in gt_rates]

            color = algo_colors.get(algo, 'gray')
            ax.plot(gt_rates, means, marker='o', label=algo,
                   color=color, linewidth=2)
            ax.fill_between(gt_rates,
                           [m - s for m, s in zip(means, stds)],
                           [m + s for m, s in zip(means, stds)],
                           alpha=0.2, color=color)

        ax.set_title(f'Number of Trajectories = {num_traj}', fontsize=12, fontweight='bold')
        ax.set_xlabel('GT Rate (%)', fontsize=10)
        if idx == 0:
            ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')

    fig.suptitle(f'{domain_name}: {metric_name.replace("_", " ").title()} vs GT Rate',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / f'{domain_name}_{metric_name}_vs_gt_rate_by_num_trajectories.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved plot: {output_path}")


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

    # Ensure gt_rate column exists
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run PDDL action model learning benchmark')
    parser.add_argument('--domain', type=str, default='all',
                       help='Domain to run (blocksworld, hanoi, n_puzzle_typed, maze, or "all" for all domains)')
    parser.add_argument('--mode', type=str, default='masked', choices=['masked', 'fullyobs'],
                       help='Mode to run: "masked" (PISAM/PO_ROSAME) or "fullyobs" (SAM/ROSAME)')

    args = parser.parse_args()

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

    main(selected_domains=selected_domains, mode=args.mode)