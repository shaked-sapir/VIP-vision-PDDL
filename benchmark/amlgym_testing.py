import csv
import json
import os
import random
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from threading import Lock

import pandas as pd
import matplotlib.pyplot as plt
from pddl_plus_parser.lisp_parsers import TrajectoryParser, DomainParser
from pddl_plus_parser.models import Observation

from amlgym.benchmarks import *
from amlgym.algorithms import *
from amlgym.metrics import *

from benchmark.amlgym_models.NOISY_PISAM import NOISY_PISAM
from benchmark.amlgym_models.PISAM import PISAM
from benchmark.amlgym_models.PO_ROSAME import PO_ROSAME
from src.utils.pddl import observation_to_trajectory_file
from src.utils.masking import load_masking_info, save_masking_info

# =============================================================================
# CONFIG & PATHS
# =============================================================================

benchmark_path = Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/benchmark")
print_metrics()

# Global lock for thread-safe evaluation (AMLGym SimpleDomainReader is not thread-safe)
evaluation_lock = Lock()

experiment_data_dirs = {
    "blocksworld": ["multi_problem_04-12-2025T12:00:44__model=gpt-5.1__steps=50__planner"],
}

domain_name_mappings = {'blocksworld': 'blocksworld'}

not_in_amlgym_domains = {
    'blocksworld': {
        "domain_path": benchmark_path / 'domains' / 'blocksworld' / 'blocksworld.pddl',
        "problems_paths": sorted(str(p) for p in (benchmark_path / 'domains' / 'blocksworld' / 'test_problems').glob("problem*.pddl"))
    }
}

N_FOLDS = 5
TRAJECTORY_SIZES = [1, 3, 5, 7, 10, 20, 30]
NUM_TRAJECTORIES = 5  # Always use 5 trajectories

metric_cols = [
    "precision_precs_pos", "precision_precs_neg", "precision_eff_pos", "precision_eff_neg", "precision_overall",
    "recall_precs_pos", "recall_precs_neg", "recall_eff_pos", "recall_eff_neg", "recall_overall",
    "problems_count", "solving_ratio", "false_plans_ratio", "unsolvable_ratio", "timed_out",
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def truncate_trajectory(traj_path: Path, domain_path: Path, max_steps: int) -> Path:
    """Truncate trajectory to max_steps and save to a temporary file. Also truncates masking_info."""
    domain = DomainParser(domain_path).parse_domain()
    parser = TrajectoryParser(domain)
    observation = parser.parse_trajectory(traj_path)

    # Truncate observation components to max_steps
    if len(observation.components) > max_steps:
        observation.components = observation.components[:max_steps]

    # Save truncated trajectory
    output_path = traj_path.parent / f"{traj_path.stem}_truncated_{max_steps}.trajectory"
    observation_to_trajectory_file(observation, output_path)

    # Truncate and save corresponding masking_info file
    # Extract base problem name (remove _truncated_N, _final, _frame_axioms suffixes)
    problem_name = traj_path.stem.split('_truncated_')[0].split('_final')[0].split('_frame_axioms')[0]
    masking_info_path = traj_path.parent / f"{problem_name}.masking_info"

    if masking_info_path.exists():
        masking_info = load_masking_info(masking_info_path, domain)
        # Truncate to max_steps + 1 (initial state + max_steps transitions)
        truncated_masking_info = masking_info[:max_steps + 1]

        # Save truncated masking info with same stem as truncated trajectory
        # e.g., problem7_truncated_3.trajectory -> problem7_truncated_3.masking_info
        save_masking_info(output_path.parent, output_path.stem, truncated_masking_info)

    return output_path


def save_learning_metrics(output_dir: Path, report: dict, trajectory_mapping: Dict[str, str] = None):
    """Save learning metrics to JSON file."""
    metrics = {
        "learning_time_seconds": report.get("total_time_seconds", None),
        "max_depth": report.get("max_depth", None),
        "nodes_expanded": report.get("nodes_expanded", None),
    }

    # Add trajectory mapping if provided
    if trajectory_mapping:
        metrics["trajectory_mapping"] = trajectory_mapping

    with open(output_dir / "learning_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)


def run_noisy_pisam_trial(domain_path: Path, trajectories: List[Path], testing_dir: Path,
                         fold: int, traj_size: int) -> Tuple[str, List[Observation], dict]:
    """Run NOISY_PISAM and save results."""
    noisy_pisam = NOISY_PISAM()
    model, final_obs, report = noisy_pisam.learn(str(domain_path), [str(t) for t in trajectories])

    # Create output directory
    timestamp = datetime.now().strftime("%d-%m-%YT%H:%M:%S")
    output_dir = testing_dir / f"{timestamp}__fold={fold}__traj-size={traj_size}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create trajectory mapping (final_observation_X -> original trajectory name)
    trajectory_mapping = {}
    for i, (obs, traj_path) in enumerate(zip(final_obs, trajectories)):
        obs_path = output_dir / f"final_observation_{i}.trajectory"
        observation_to_trajectory_file(obs, obs_path)
        # Store original trajectory name (without path)
        trajectory_mapping[f"final_observation_{i}"] = traj_path.name

    # Save learning metrics with trajectory mapping
    save_learning_metrics(output_dir, report, trajectory_mapping)

    # Save learned model
    with open(output_dir / "learned_model.pddl", 'w') as f:
        f.write(model)

    return model, final_obs, report


def evaluate_model(model_path: str, domain_ref_path: Path, test_problems: List[str]) -> dict:
    """Evaluate a learned model. Thread-safe due to AMLGym SimpleDomainReader constraints."""
    # Use lock to prevent concurrent access to domain file (SimpleDomainReader creates _clean files)
    with evaluation_lock:
        precision = syntactic_precision(model_path, str(domain_ref_path))
        recall = syntactic_recall(model_path, str(domain_ref_path))
        problem_solving_result = problem_solving(model_path, str(domain_ref_path), test_problems, timeout=60)

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
    if std_val is None or pd.isna(std_val):
        return f"{mean_val:.3f}"
    return f"{mean_val:.3f}±{std_val:.3f}"


def run_single_fold(fold: int, problem_dirs: List[Path], n_problems: int, traj_size: int,
                    domain_ref_path: Path, testing_dir: Path, domain_name: str, bench_name: str) -> dict:
    """Run a single fold experiment and return the result."""
    print(f"[PID {os.getpid()}] Starting fold {fold}, traj_size={traj_size}")
    # CV split
    indices = list(range(n_problems))
    random.seed(42 + fold)
    random.shuffle(indices)

    n_train = max(1, int(0.8 * n_problems))
    if n_train >= n_problems:
        n_train = n_problems - 1

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_problem_dirs = [problem_dirs[i] for i in train_idx]
    test_problem_dirs = [problem_dirs[i] for i in test_idx]

    # Select NUM_TRAJECTORIES random trajectories from training set
    random.seed(42 + fold)  # Reset seed for consistent selection
    selected_dirs = random.sample(train_problem_dirs, min(NUM_TRAJECTORIES, len(train_problem_dirs)))

    # Get trajectory paths
    selected_trajs = []
    for prob_dir in selected_dirs:
        traj_files = list(prob_dir.glob("*.trajectory"))
        if traj_files:
            # Filter out already truncated/processed files
            traj_files = [f for f in traj_files if 'truncated' not in f.stem and 'final' not in f.stem and 'frame_axioms' not in f.stem]
            if traj_files:
                selected_trajs.append(traj_files[0])

    # Truncate trajectories to traj_size
    truncated_trajs = [truncate_trajectory(t, domain_ref_path, traj_size) for t in selected_trajs]

    print(f"\n--- Fold {fold + 1}/{N_FOLDS}, Traj Size {traj_size} ---")
    print(f"Train: {len(train_problem_dirs)} problems | Test: {len(test_problem_dirs)} problems")
    print(f"Selected {len(selected_trajs)} trajectories, truncated to {traj_size} steps")

    # Run NOISY_PISAM
    print(f"Running NOISY_PISAM...")
    model, final_obs, report = run_noisy_pisam_trial(
        domain_ref_path, truncated_trajs, testing_dir, fold, traj_size
    )

    # Save model temporarily for evaluation
    temp_model_path = f'NOISY_PISAM_{domain_name}_fold{fold}_size{traj_size}.pddl'
    with open(temp_model_path, 'w') as f:
        f.write(model)

    # Get test problem paths
    test_problem_paths = []
    for prob_dir in test_problem_dirs:
        pddl_files = list(prob_dir.glob("*.pddl"))
        if pddl_files:
            test_problem_paths.append(str(pddl_files[0]))

    # Evaluate
    metrics = evaluate_model(temp_model_path, domain_ref_path, test_problem_paths)

    result = {
        'domain': bench_name,
        'algorithm': 'NOISY_PISAM',
        'fold': fold,
        'traj_size': traj_size,
        'problems_count': len(test_problem_paths),
        **metrics
    }

    # Clean up truncated files (both .trajectory and .masking_info)
    for t in truncated_trajs:
        if t.exists():
            t.unlink()
        # Also clean up corresponding masking_info file
        masking_file = t.parent / f"{t.stem}.masking_info"
        if masking_file.exists():
            masking_file.unlink()

    print(f"✓ Fold {fold + 1} completed for traj_size={traj_size}")
    return result


def generate_excel_report(all_results: List[dict], output_path: Path):
    """Generate Excel report with aggregated results."""
    if not all_results:
        return

    # Define metric groups for Excel table structure
    precision_metrics = ["precision_precs_pos", "precision_precs_neg", "precision_eff_pos", "precision_eff_neg", "precision_overall"]
    recall_metrics = ["recall_precs_pos", "recall_precs_neg", "recall_eff_pos", "recall_eff_neg", "recall_overall"]
    problem_metrics = ["problems_count", "solving_ratio", "false_plans_ratio", "unsolvable_ratio", "timed_out"]

    df_all = pd.DataFrame(all_results)
    grouped = df_all.groupby(["domain", "algorithm", "traj_size"])[metric_cols].agg(["mean", "std"]).reset_index()

    # Flatten columns
    flat_cols = []
    for col in grouped.columns:
        if isinstance(col, tuple):
            base, stat = col
            flat_cols.append(base if stat == "" else f"{base}_{stat}")
        else:
            flat_cols.append(col)
    grouped.columns = flat_cols

    df_avg = grouped[["domain", "algorithm", "traj_size"]].copy()
    for m in metric_cols:
        df_avg[m] = grouped[f"{m}_mean"]
        df_avg[f"{m}_std"] = grouped[f"{m}_std"]

    # Group by trajectory size
    by_size = defaultdict(list)
    for _, row in df_avg.iterrows():
        by_size[str(int(row["traj_size"]))].append(row.to_dict())

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

        for traj_size in sorted(by_size.keys(), key=lambda x: int(x)):
            results = by_size[traj_size]
            sheet_name = f"size={traj_size}"
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


# =============================================================================
# MAIN EXPERIMENT LOOP
# =============================================================================
def main():
    all_results = []

    for domain_name, bench_name in domain_name_mappings.items():
        domain_ref_path = not_in_amlgym_domains[domain_name]["domain_path"]
        test_problems = not_in_amlgym_domains[domain_name]["problems_paths"]

        for dir_name in experiment_data_dirs[domain_name]:
            data_dir = benchmark_path / 'data' / domain_name / dir_name
            trajectories_dir = data_dir / 'training' / 'trajectories'
            testing_dir = data_dir / 'testing'
            testing_dir.mkdir(parents=True, exist_ok=True)

            problem_dirs = sorted([d for d in trajectories_dir.iterdir() if d.is_dir()])
            n_problems = len(problem_dirs)

            if n_problems < 2:
                raise ValueError(f"Domain {domain_name} has too few problems ({n_problems}) for 80/20 CV.")

            print(f"\n{'=' * 80}")
            print(f"Domain: {bench_name} | data dir: {dir_name}")
            print(f"Total problems: {n_problems}")
            print(f"Trajectory sizes: {TRAJECTORY_SIZES}")
            print(f"CV folds: {N_FOLDS}")
            print(f"{'=' * 80}\n")

            # IMPORTANT: Size first, then folds (so cheap computations finish first)
            for traj_size in TRAJECTORY_SIZES:
                print(f"\n{'='*60}\nTRAJECTORY SIZE = {traj_size}\n{'='*60}")

                # Run all folds in parallel for this trajectory size
                # with ThreadPoolExecutor(max_workers=N_FOLDS) as executor:
                with ProcessPoolExecutor(max_workers=N_FOLDS) as executor:
                    futures = []
                    for fold in range(N_FOLDS):
                        future = executor.submit(
                            run_single_fold,
                            fold, problem_dirs, n_problems, traj_size,
                            domain_ref_path, testing_dir, domain_name, bench_name
                        )
                        futures.append(future)

                    # Wait for all folds to complete and collect results
                    fold_results = []
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            fold_results.append(result)
                        except Exception as e:
                            print(f"ERROR in fold: {e}")
                            import traceback
                            traceback.print_exc()

                # Add all results from this trajectory size
                all_results.extend(fold_results)

                # Write results to CSV after all folds complete
                csv_path = benchmark_path / f"results_{domain_name}.csv"
                df_results = pd.DataFrame(all_results)
                df_results.to_csv(csv_path, index=False)
                print(f"\n✓ All folds for traj_size={traj_size} completed")
                print(f"✓ Results written to {csv_path}")

                # Generate Excel report after each trajectory size completes
                print(f"\n{'='*60}")
                print(f"GENERATING AGGREGATED REPORT FOR TRAJECTORY SIZE = {traj_size}")
                print(f"{'='*60}")

                timestamp = datetime.now().strftime("%d-%m-%YT%H:%M:%S")
                xlsx_path = benchmark_path / f"benchmark_results_{timestamp}.xlsx"
                generate_excel_report(all_results, xlsx_path)
                print(f"✓ Excel report saved to: {xlsx_path}")
                print(f"  Sheets completed so far: {sorted(set(r['traj_size'] for r in all_results))}")

    # =============================================================================
    # GENERATE FINAL PLOTS
    # =============================================================================

    print("\n" + "=" * 80)
    print("GENERATING FINAL PLOTS")
    print("=" * 80)

    # Aggregate for plotting
    df_all = pd.DataFrame(all_results)
    grouped = df_all.groupby(["domain", "algorithm", "traj_size"])[metric_cols].agg(["mean", "std"]).reset_index()

    flat_cols = []
    for col in grouped.columns:
        if isinstance(col, tuple):
            base, stat = col
            flat_cols.append(base if stat == "" else f"{base}_{stat}")
        else:
            flat_cols.append(col)
    grouped.columns = flat_cols

    df_avg = grouped[["domain", "algorithm", "traj_size"]].copy()
    for m in metric_cols:
        df_avg[m] = grouped[f"{m}_mean"]
        df_avg[f"{m}_std"] = grouped[f"{m}_std"]

    # =============================================================================
    # PLOTTING
    # =============================================================================

    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    plots_dir = benchmark_path / "plots"
    plots_dir.mkdir(exist_ok=True)

    def plot_metric_vs_size(df, metric_key, metric_title, save_dir):
        """Plot metric vs trajectory size with error bars."""
        plt.figure(figsize=(8, 5))

        algorithms = sorted(df["algorithm"].unique())
        for algo in algorithms:
            sub = df[df["algorithm"] == algo].sort_values("traj_size")
            x = sub["traj_size"]
            y = sub[metric_key]
            yerr = sub[f"{metric_key}_std"] if f"{metric_key}_std" in sub.columns else None

            plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4, label=algo)

        plt.title(f"{metric_title} vs Trajectory Size")
        plt.xlabel("Trajectory Size (steps)")
        plt.ylabel(metric_title)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        save_path = Path(save_dir) / f"{metric_key}_vs_traj_size.png"
        plt.savefig(save_path)
        print(f"✓ Saved plot: {save_path}")
        plt.close()

    plot_metric_vs_size(df_avg, "solving_ratio", "Solving Ratio", plots_dir)
    plot_metric_vs_size(df_avg, "false_plans_ratio", "False Plan Ratio", plots_dir)
    plot_metric_vs_size(df_avg, "unsolvable_ratio", "Unsolvable Ratio", plots_dir)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()