from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _df_to_md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "_No data._"
    shown = df.head(max_rows).fillna("")
    cols = list(shown.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = [
        "| " + " | ".join(str(rec[c]) for c in cols) + " |"
        for _, rec in shown.iterrows()
    ]
    tail = f"\n\n_... showing first {max_rows} of {len(df)} rows._" if len(df) > max_rows else ""
    return "\n".join([header, sep] + rows) + tail


def _top_delta_rows(delta_df: pd.DataFrame, delta_cols: List[str], top_k: int = 10) -> pd.DataFrame:
    if delta_df is None or delta_df.empty:
        return pd.DataFrame()
    tmp = delta_df.copy()
    present = [c for c in delta_cols if c in tmp.columns]
    if not present:
        return pd.DataFrame()
    tmp["abs_delta_score"] = tmp[present].abs().sum(axis=1)
    keep = [
        c
        for c in [
            "baseline_experiment",
            "experiment_id",
            "phase",
            "num_trajectories",
            "gt_rate",
            "abs_delta_score",
            *present,
        ]
        if c in tmp.columns
    ]
    return tmp.sort_values("abs_delta_score", ascending=False)[keep].head(top_k)


def write_markdown_report(
    output_dir: Path,
    *,
    root: Path,
    domain: str,
    model_policy: str,
    baseline_experiment: str | None,
    experiments_df: pd.DataFrame,
    fluent_raw_df: pd.DataFrame,
    planning_raw_df: pd.DataFrame,
    fluent_agg_df: pd.DataFrame,
    planning_agg_df: pd.DataFrame,
    fluent_timeout_comp: Dict[str, pd.DataFrame],
    fluent_weight_comp: Dict[str, pd.DataFrame],
    planning_timeout_comp: Dict[str, pd.DataFrame],
    planning_weight_comp: Dict[str, pd.DataFrame],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "comparison_report.md"

    lines: List[str] = [
        "# Experiment Comparison Report",
        "",
        f"- Generated at: `{datetime.now().isoformat()}`",
        f"- Root: `{root}`",
        f"- Domain filter: `{domain}`",
        f"- Fluent model policy: `{model_policy}`",
        f"- Baseline override: `{baseline_experiment}`",
        "",
        "## Data Coverage",
        "",
        f"- Experiments discovered: **{len(experiments_df)}**",
        f"- Fluent raw rows: **{len(fluent_raw_df)}**",
        f"- Planning raw rows: **{len(planning_raw_df)}**",
        f"- Fluent aggregated rows: **{len(fluent_agg_df)}**",
        f"- Planning aggregated rows: **{len(planning_agg_df)}**",
        "",
        "## Experiments Index",
        "",
    ]

    exp_cols = [
        c
        for c in [
            "experiment_id",
            "date_tag",
            "timeout_seconds",
            "model_constraint_weight",
            "mode",
            "has_run_params",
        ]
        if c in experiments_df.columns
    ]
    lines.append(_df_to_md_table(experiments_df[exp_cols] if exp_cols else experiments_df, max_rows=100))
    lines.extend(
        [
            "",
            "## Fluent TP/FP/FN Comparisons",
            "",
            "### By Timeout",
            "",
            _df_to_md_table(fluent_timeout_comp.get("wide", pd.DataFrame()), max_rows=40),
            "",
            "#### Top Absolute Delta Rows (Timeout Grouping)",
            "",
            _df_to_md_table(
                _top_delta_rows(
                    fluent_timeout_comp.get("delta", pd.DataFrame()),
                    ["tp_delta_vs_baseline", "fp_delta_vs_baseline", "fn_delta_vs_baseline"],
                ),
                max_rows=20,
            ),
            "",
            "### By Model Constraint Weight",
            "",
            _df_to_md_table(fluent_weight_comp.get("wide", pd.DataFrame()), max_rows=40),
            "",
            "#### Top Absolute Delta Rows (Weight Grouping)",
            "",
            _df_to_md_table(
                _top_delta_rows(
                    fluent_weight_comp.get("delta", pd.DataFrame()),
                    ["tp_delta_vs_baseline", "fp_delta_vs_baseline", "fn_delta_vs_baseline"],
                ),
                max_rows=20,
            ),
            "",
            "## Planning Ratio Comparisons (Solved / False Plans / Unsolvable)",
            "",
            "### By Timeout",
            "",
            _df_to_md_table(planning_timeout_comp.get("wide", pd.DataFrame()), max_rows=40),
            "",
            "#### Top Absolute Delta Rows (Timeout Grouping)",
            "",
            _df_to_md_table(
                _top_delta_rows(
                    planning_timeout_comp.get("delta", pd.DataFrame()),
                    [
                        "solving_ratio_delta_vs_baseline",
                        "false_plans_ratio_delta_vs_baseline",
                        "unsolvable_ratio_delta_vs_baseline",
                    ],
                ),
                max_rows=20,
            ),
            "",
            "### By Model Constraint Weight",
            "",
            _df_to_md_table(planning_weight_comp.get("wide", pd.DataFrame()), max_rows=40),
            "",
            "#### Top Absolute Delta Rows (Weight Grouping)",
            "",
            _df_to_md_table(
                _top_delta_rows(
                    planning_weight_comp.get("delta", pd.DataFrame()),
                    [
                        "solving_ratio_delta_vs_baseline",
                        "false_plans_ratio_delta_vs_baseline",
                        "unsolvable_ratio_delta_vs_baseline",
                    ],
                ),
                max_rows=20,
            ),
            "",
        ]
    )

    report_path.write_text("\n".join(lines))
    return report_path


def write_interaction_report(
    output_dir: Path,
    *,
    varying_params: List[str],
    interaction_effects_df: pd.DataFrame,
    interaction_combos_df: pd.DataFrame,
    single_param_summary_df: pd.DataFrame,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "interaction_report.md"

    lines: List[str] = [
        "# Parameter Interaction Report",
        "",
        f"- Generated at: `{datetime.now().isoformat()}`",
        f"- Varying parameters observed: `{', '.join(varying_params) if varying_params else 'None'}`",
        "",
        "## Single-Parameter Deltas (Summary)",
        "",
        _df_to_md_table(
            single_param_summary_df.sort_values(
                by=["avg_abs_delta_score"],
                ascending=False,
            )
            if not single_param_summary_df.empty and "avg_abs_delta_score" in single_param_summary_df.columns
            else single_param_summary_df,
            max_rows=40,
        ),
        "",
        "## Interaction Effect Ranking",
        "",
        _df_to_md_table(interaction_effects_df, max_rows=60),
        "",
        "## Parameter Combination Performance",
        "",
        _df_to_md_table(interaction_combos_df, max_rows=60),
        "",
    ]

    report_path.write_text("\n".join(lines))
    return report_path

