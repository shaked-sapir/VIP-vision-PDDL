"""
Reporting, plotting, and Excel generation functions for AMLGym experiments.

Extracted from benchmark_runner.py — all functions are stateless and operate
on DataFrames, CSV paths, or result dicts.
"""

from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd


# =============================================================================
# SHARED CONSTANTS
# =============================================================================

metric_cols = [
    "precision_precs_pos", "precision_precs_neg", "precision_eff_pos", "precision_eff_neg", "precision_overall",
    "recall_precs_pos", "recall_precs_neg", "recall_eff_pos", "recall_eff_neg", "recall_overall",
    "problems_count", "solving_ratio", "false_plans_ratio", "unsolvable_ratio", "planning_timed_out_ratio",
    "pred_app_precision", "pred_app_recall", "pred_eff_precision", "pred_eff_recall",
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

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


def _load_results_csv_or_none(results_csv_path: Path, context: str) -> Optional[pd.DataFrame]:
    """Load a results CSV, returning None when the file is missing or empty."""
    path = Path(results_csv_path)
    if not path.exists():
        print(f"Warning: {context}: results file not found ({path}), skipping")
        return None
    if path.stat().st_size == 0:
        print(f"Warning: {context}: results file is empty ({path}), skipping")
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        print(f"Warning: {context}: no columns in results file ({path}), skipping")
        return None
    if df.empty:
        print(f"Warning: {context}: results file has no rows ({path}), skipping")
        return None
    return df


# =============================================================================
# EXCEL REPORT
# =============================================================================

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


# =============================================================================
# BASIC PLOTS
# =============================================================================

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
# GT INJECTION PLOTS
# =============================================================================

def plot_metric_vs_num_trajectories_by_gt_rate(results_df, metric_name, output_dir, domain_name):
    """Impact of Number of Trajectories — subplots per GT rate."""
    from matplotlib.lines import Line2D

    # Define jitter and style mapping
    jitter_config = {
        ('unclean', 'PISAM'): {'h': -0.03, 'v': 0.005},
        ('cleaned', 'PISAM'): {'h': 0.01, 'v': -0.005},
        ('unclean', 'PO_ROSAME'): {'h': -0.01, 'v': 0.01},
        ('cleaned', 'PO_ROSAME'): {'h': 0.03, 'v': -0.01},
    }
    color_map = {
        ('PISAM', 'unclean'): '#E74C3C',
        ('PO_ROSAME', 'unclean'): '#9B59B6',
        ('PISAM', 'cleaned'): '#3498DB',
        ('PO_ROSAME', 'cleaned'): '#2ECC71',
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
                plot_color = color_map.get((algo, phase), 'black')
                marker = 'o' if phase == 'unclean' else 's'
                ax.plot(jx, jy, marker=marker, label=f"{algo} ({phase})",
                        color=plot_color, linestyle=style['linestyle'][phase], linewidth=2)

                for x_val, y_orig, y_jit in zip(jx, original_y, jy):
                    ax.text(x_val + 0.05, y_jit, f"{y_orig:.2f}",
                            color='black', fontsize=8, fontweight='bold',
                            va='center', ha='left')

        ax.set_ylim(-0.1, 1.2)
        ax.set_title('Baseline (gt = only init state)' if gt_rate == 0 else f'GT Rate {gt_rate}%', fontweight='bold')
        ax.set_xlabel('Number of Trajectories', fontsize=12)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
        ax.grid(True, alpha=0.3)

        if gt_rate == 100:
            for spine in ax.spines.values():
                spine.set_edgecolor('orange')
                spine.set_linewidth(3)

        custom_handles = []
        pisam_unclean_color = color_map[('PISAM', 'unclean')]
        line1 = Line2D([0, 1], [0, 0], color=pisam_unclean_color, linestyle='--', linewidth=2, marker='o',
                      markersize=8, markevery=[0], label='PISAM unclean')
        line1.set_dashes([5, 2])
        custom_handles.append(line1)
        rosame_unclean_color = color_map[('PO_ROSAME', 'unclean')]
        line2 = Line2D([0, 1], [0, 0], color=rosame_unclean_color, linestyle='--', linewidth=2, marker='o',
                      markersize=8, markevery=[0], label='PO_ROSAME unclean')
        line2.set_dashes([5, 2])
        custom_handles.append(line2)
        pisam_cleaned_color = color_map[('PISAM', 'cleaned')]
        custom_handles.append(Line2D([0, 1], [0, 0], color=pisam_cleaned_color, linestyle='-', linewidth=2, marker='s',
                                    markersize=7, markevery=[0], label='PISAM cleaned'))
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
    """Impact of GT Rate — subplots per number of trajectories."""
    from matplotlib.lines import Line2D

    jitter_config = {
        ('unclean', 'PISAM'): {'h': -0.4, 'v': 0.005},
        ('cleaned', 'PISAM'): {'h': 0.1, 'v': -0.005},
        ('unclean', 'PO_ROSAME'): {'h': -0.1, 'v': 0.01},
        ('cleaned', 'PO_ROSAME'): {'h': 0.4, 'v': -0.01},
    }
    color_map = {
        ('PISAM', 'unclean'): '#E74C3C',
        ('PO_ROSAME', 'unclean'): '#9B59B6',
        ('PISAM', 'cleaned'): '#3498DB',
        ('PO_ROSAME', 'cleaned'): '#2ECC71',
    }
    algo_style_map = {
        'PISAM': {'linestyle': {'unclean': '--', 'cleaned': '-'}},
        'PO_ROSAME': {'linestyle': {'unclean': '--', 'cleaned': '-'}},
    }

    available_nums = sorted(results_df['num_trajectories'].unique())
    num_plots = len(available_nums)

    if num_plots <= 4:
        nrows, ncols = (2, 2)
    elif num_plots <= 6:
        nrows, ncols = (2, 3)
    elif num_plots <= 8:
        nrows, ncols = (2, 4)
    elif num_plots <= 9:
        nrows, ncols = (3, 3)
    else:
        nrows, ncols = (3, 4)

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, num_traj in enumerate(available_nums):
        ax = axes[idx]
        num_df = results_df[results_df['num_trajectories'] == num_traj].copy()

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
                plot_color = color_map.get((algo, phase), 'black')
                marker = 'o' if phase == 'unclean' else 's'
                ax.plot(jx, jy, marker=marker, label=f"{algo} ({phase})", color=plot_color,
                        linestyle=style['linestyle'][phase], linewidth=2)

                for x_val, y_orig, y_jit in zip(jx, original_y, jy):
                    ax.text(x_val + 0.8, y_jit, f"{y_orig:.2f}",
                            color='black', fontsize=8, fontweight='bold',
                            va='center', ha='left')

        if all_gt_rates:
            x_min = min(all_gt_rates) - 5
            x_max = max(all_gt_rates) + 5
            ax.set_xlim(x_min, x_max)
            ax.set_xticks(all_gt_rates)

        ax.set_ylim(-0.1, 1.2)

        add_100_line = False
        if all_gt_rates and (100 in all_gt_rates or max(all_gt_rates) >= 90):
            add_100_line = True
            ax.axvline(x=100, color='orange', linestyle='-', linewidth=2, alpha=0.7, zorder=0)

        ax.set_title(f'Number of Trajectories = {num_traj}', fontweight='bold')
        ax.set_xlabel('GT Rate (%)', fontsize=12)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
        ax.grid(True, alpha=0.3)

        custom_handles = []
        pisam_unclean_color = color_map[('PISAM', 'unclean')]
        line1 = Line2D([0, 1], [0, 0], color=pisam_unclean_color, linestyle='--', linewidth=2, marker='o',
                      markersize=8, markevery=[0], label='PISAM unclean')
        line1.set_dashes([5, 2])
        custom_handles.append(line1)
        rosame_unclean_color = color_map[('PO_ROSAME', 'unclean')]
        line2 = Line2D([0, 1], [0, 0], color=rosame_unclean_color, linestyle='--', linewidth=2, marker='o',
                      markersize=8, markevery=[0], label='PO_ROSAME unclean')
        line2.set_dashes([5, 2])
        custom_handles.append(line2)
        pisam_cleaned_color = color_map[('PISAM', 'cleaned')]
        custom_handles.append(Line2D([0, 1], [0, 0], color=pisam_cleaned_color, linestyle='-', linewidth=2, marker='s',
                                    markersize=7, markevery=[0], label='PISAM cleaned'))
        rosame_cleaned_color = color_map[('PO_ROSAME', 'cleaned')]
        custom_handles.append(Line2D([0, 1], [0, 0], color=rosame_cleaned_color, linestyle='-', linewidth=2, marker='s',
                                    markersize=7, markevery=[0], label='PO_ROSAME cleaned'))

        if add_100_line:
            custom_handles.append(Line2D([0, 1], [0, 0], color='orange', linestyle='-', linewidth=2, label='100% GT Rate'))

        ax.legend(handles=custom_handles, fontsize=9, loc='lower right')

    for idx in range(len(available_nums), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f'{domain_name.upper()}: Impact of GT Rate on {metric_name.replace("_", " ").title()}',
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(output_dir / f"{domain_name}_{metric_name}_vs_gt_rate_by_num_trajectories.png", dpi=300)
    plt.close()


def generate_gt_injection_plots(results_csv_path, output_dir, domain_name):
    """Generate all GT injection analysis plots for a domain."""
    df = _load_results_csv_or_none(results_csv_path, "GT injection plots")
    if df is None:
        return

    if 'gt_rate' not in df.columns:
        print(f"Warning: No 'gt_rate' column in {results_csv_path}, skipping GT injection plots")
        return

    plots_dir = output_dir / 'plots' / 'gt_injection'
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating GT injection plots for {domain_name}...")

    metrics = ['solving_ratio', 'false_plans_ratio', 'unsolvable_ratio']

    for metric in metrics:
        if metric not in df.columns:
            print(f"  Warning: Metric '{metric}' not found in results")
            continue

        print(f"\n  Plotting {metric}...")
        plot_metric_vs_num_trajectories_by_gt_rate(df, metric, plots_dir, domain_name)
        plot_metric_vs_gt_rate_by_num_trajectories(df, metric, plots_dir, domain_name)

    print(f"\n✓ All GT injection plots saved to: {plots_dir}")


# =============================================================================
# STACKED SOLVING RATE PLOTS
# =============================================================================

def plot_stacked_solving_rate(results_csv_path, output_dir, domain_name):
    """Generate stacked area charts showing solving rate over number of trajectories."""
    df = _load_results_csv_or_none(results_csv_path, "stacked solving rate plots")
    if df is None:
        return

    required_cols = ['algorithm', 'num_trajectories', 'gt_rate', 'solving_ratio',
                     'false_plans_ratio', 'unsolvable_ratio', '_internal_phase']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"  Warning: Missing columns in {results_csv_path}: {missing_cols}, skipping stacked plots")
        return

    pisam_df = df[df['algorithm'] == 'PISAM'].copy()

    if pisam_df.empty:
        print(f"  Warning: No PISAM data found for {domain_name}, skipping stacked plots")
        return

    stacked_plots_dir = output_dir / 'plots' / 'stacked_solving_rate'
    stacked_plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating stacked solving rate plots for {domain_name}...")

    colors = {
        'solved': '#FFF4C4',
        'inapplicable': '#B0E0E6',
        'not_solved': '#FF6B6B'
    }

    for phase in ['unclean', 'cleaned']:
        phase_df = pisam_df[pisam_df['_internal_phase'] == phase].copy()

        if phase_df.empty:
            continue

        gt_rates = sorted(phase_df['gt_rate'].unique())

        if not gt_rates:
            continue

        num_plots = len(gt_rates)
        nrows, ncols = (2, 2) if num_plots <= 4 else (2, 3)
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), sharex=True, sharey=True)
        axes = axes.flatten()

        phase_label = 'Cleaned' if phase == 'cleaned' else 'Unclean'

        for idx, gt_rate in enumerate(gt_rates):
            ax = axes[idx]

            rate_df = phase_df[phase_df['gt_rate'] == gt_rate].copy()

            if rate_df.empty:
                continue

            grouped = rate_df.groupby('num_trajectories').agg({
                'solving_ratio': 'mean',
                'false_plans_ratio': 'mean',
                'unsolvable_ratio': 'mean'
            }).reset_index()

            grouped = grouped.sort_values('num_trajectories')

            x = grouped['num_trajectories'].values
            num_trajs_sorted = sorted(grouped['num_trajectories'].unique())

            solved = grouped['solving_ratio'].values
            inapplicable = grouped['false_plans_ratio'].values

            not_solved = 1.0 - (solved + inapplicable)
            not_solved = [max(0.0, min(1.0, val)) for val in not_solved]

            ax.stackplot(x, solved, inapplicable, not_solved,
                        colors=[colors['solved'], colors['inapplicable'], colors['not_solved']],
                        alpha=0.8, labels=['Solved', 'Inapplicable', 'Not Solved'],
                        edgecolor='white', linewidth=0.5)

            subplot_title = 'Baseline (gt = only init state)' if gt_rate == 0 else f'GT Rate {gt_rate}%'
            ax.set_title(subplot_title, fontweight='bold', fontsize=11)

            ax.set_xlabel('Number of Trajectories', fontsize=10)
            ax.set_ylabel('Solving Rate', fontsize=10)

            ax.set_xticks(num_trajs_sorted)
            ax.set_xlim(min(num_trajs_sorted) - 0.5, max(num_trajs_sorted) + 0.5)

            ax.set_ylim(0.0, 1.0)
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

            ax.grid(True, linestyle='--', alpha=0.3, axis='both')

            if idx == 0:
                ax.legend(loc='upper right', framealpha=0.9, fontsize=9)

        for idx in range(len(gt_rates), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(f'Solving Rate Over Number of Trajectories - PISAM {phase_label} - {domain_name.upper()}',
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        filename = f"{domain_name}_PISAM_{phase}_stacked_solving_rate.png"
        plt.savefig(stacked_plots_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {filename}")

    print(f"\n✓ All stacked solving rate plots saved to: {stacked_plots_dir}")
