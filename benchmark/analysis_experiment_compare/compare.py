from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


def _pick_baseline(group_df: pd.DataFrame, baseline_experiment: Optional[str]) -> str:
    if baseline_experiment and baseline_experiment in set(group_df["experiment_id"]):
        return baseline_experiment
    # Earliest date tag wins (empty date tags sorted last by tuple trick).
    ordered = (
        group_df[["experiment_id", "date_tag"]]
        .drop_duplicates()
        .sort_values(by=["date_tag", "experiment_id"], ascending=[True, True])
    )
    return ordered.iloc[0]["experiment_id"]


def build_group_comparisons(
    agg_df: pd.DataFrame,
    group_key: str,
    metric_cols: List[str],
    baseline_experiment: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Return two frames: wide comparison + baseline deltas for each group."""
    if agg_df.empty:
        return {"wide": pd.DataFrame(), "delta": pd.DataFrame()}

    wide_rows: List[pd.DataFrame] = []
    delta_rows: List[pd.DataFrame] = []

    for gval, gdf in agg_df.groupby(group_key):
        if gdf["experiment_id"].nunique() < 2:
            continue

        keep = ["experiment_id", "phase", "num_trajectories", "gt_rate"]
        mean_cols = [f"{m}_mean" for m in metric_cols]
        sub = gdf[keep + mean_cols].copy()

        # Wide table
        wide = sub.melt(
            id_vars=keep,
            value_vars=mean_cols,
            var_name="metric",
            value_name="value",
        ).pivot_table(
            index=["phase", "num_trajectories", "gt_rate", "metric"],
            columns="experiment_id",
            values="value",
            aggfunc="first",
        ).reset_index()
        wide.insert(0, group_key, gval)
        wide_rows.append(wide)

        # Delta vs baseline
        baseline = _pick_baseline(gdf, baseline_experiment)
        baseline_sub = (
            sub[sub["experiment_id"] == baseline]
            .set_index(["phase", "num_trajectories", "gt_rate"])[mean_cols]
            .add_prefix("baseline_")
        )

        merged = sub.merge(
            baseline_sub,
            left_on=["phase", "num_trajectories", "gt_rate"],
            right_index=True,
            how="left",
        )
        for m in metric_cols:
            merged[f"{m}_delta_vs_baseline"] = (
                merged[f"{m}_mean"] - merged[f"baseline_{m}_mean"]
            )
        merged.insert(0, group_key, gval)
        merged.insert(1, "baseline_experiment", baseline)
        delta_rows.append(merged)

    return {
        "wide": pd.concat(wide_rows, ignore_index=True) if wide_rows else pd.DataFrame(),
        "delta": pd.concat(delta_rows, ignore_index=True) if delta_rows else pd.DataFrame(),
    }

