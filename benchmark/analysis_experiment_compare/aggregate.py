from __future__ import annotations

from typing import List

import pandas as pd


def _aggregate(
    df: pd.DataFrame,
    group_cols: List[str],
    metric_cols: List[str],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    agg_map = {}
    for m in metric_cols:
        agg_map[m] = ["mean", "std"]
    agg_df = df.groupby(group_cols).agg(agg_map).reset_index()
    agg_df.columns = [
        col if isinstance(col, str) else f"{col[0]}_{col[1]}" if col[1] else col[0]
        for col in agg_df.columns
    ]
    fold_counts = df.groupby(group_cols)["fold"].nunique().reset_index(name="n_folds_observed")
    return agg_df.merge(fold_counts, on=group_cols, how="left")


def aggregate_fluent_metrics(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "experiment_id",
        "timeout_seconds",
        "model_constraint_weight",
        "date_tag",
        "algorithm",
        "phase",
        "num_trajectories",
        "gt_rate",
    ]
    metric_cols = ["tp", "fp", "fn", "precision", "recall"]
    return _aggregate(df, group_cols, metric_cols)


def aggregate_planning_metrics(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "experiment_id",
        "timeout_seconds",
        "model_constraint_weight",
        "date_tag",
        "algorithm",
        "phase",
        "num_trajectories",
        "gt_rate",
    ]
    metric_cols = ["solving_ratio", "false_plans_ratio", "unsolvable_ratio"]
    return _aggregate(df, group_cols, metric_cols)

