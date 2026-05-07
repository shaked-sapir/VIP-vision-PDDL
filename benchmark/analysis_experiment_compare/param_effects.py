from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd


PARAMETER_COLUMNS: Sequence[str] = (
    "mode",
    "frame_axiom_mode",
    "learning_timeout_seconds",
    "planning_timeout_seconds",
    "fluent_patch_cost",
    "fluent_patch_weight",
    "model_patch_cost",
    "model_constraint_weight",
    "max_search_nodes",
    "node_choosing_strategy",
)


def _normalize_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (list, dict)):
        return str(value)
    return value


def _present_param_cols(experiments_df: pd.DataFrame) -> List[str]:
    return [col for col in PARAMETER_COLUMNS if col in experiments_df.columns]


def varying_parameter_columns(experiments_df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for col in _present_param_cols(experiments_df):
        values = {_normalize_value(v) for v in experiments_df[col].tolist()}
        if len(values) > 1:
            cols.append(col)
    return cols


def _pair_delta_rows(
    *,
    agg_df: pd.DataFrame,
    experiment_a: str,
    experiment_b: str,
    changed_param: str,
    changed_from: Any,
    changed_to: Any,
    metric_set: str,
    metric_cols: Sequence[str],
) -> pd.DataFrame:
    if agg_df.empty:
        return pd.DataFrame()

    keep_keys = ["algorithm", "phase", "num_trajectories", "gt_rate"]
    left_cols = [*keep_keys, *[f"{m}_mean" for m in metric_cols]]

    left = agg_df[agg_df["experiment_id"] == experiment_a][left_cols].rename(
        columns={f"{m}_mean": f"{m}_mean_a" for m in metric_cols}
    )
    right = agg_df[agg_df["experiment_id"] == experiment_b][left_cols].rename(
        columns={f"{m}_mean": f"{m}_mean_b" for m in metric_cols}
    )
    merged = left.merge(right, on=keep_keys, how="inner")
    if merged.empty:
        return pd.DataFrame()

    for metric in metric_cols:
        merged[f"{metric}_delta"] = merged[f"{metric}_mean_b"] - merged[f"{metric}_mean_a"]

    delta_cols = [f"{metric}_delta" for metric in metric_cols]
    merged["delta_abs_score"] = merged[delta_cols].abs().sum(axis=1)
    merged.insert(0, "metric_set", metric_set)
    merged.insert(1, "experiment_a", experiment_a)
    merged.insert(2, "experiment_b", experiment_b)
    merged.insert(3, "changed_param", changed_param)
    merged.insert(4, "changed_from", changed_from)
    merged.insert(5, "changed_to", changed_to)

    return merged


def build_single_param_deltas(
    *,
    experiments_df: pd.DataFrame,
    fluent_agg_df: pd.DataFrame,
    planning_agg_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if experiments_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    param_cols = _present_param_cols(experiments_df)
    exp_rows = experiments_df.sort_values("experiment_id")[["experiment_id", *param_cols]]
    exp_records = exp_rows.to_dict("records")

    delta_frames: List[pd.DataFrame] = []

    for left, right in combinations(exp_records, 2):
        diffs = [
            col
            for col in param_cols
            if _normalize_value(left.get(col)) != _normalize_value(right.get(col))
        ]
        if len(diffs) != 1:
            continue

        changed_param = diffs[0]
        changed_from = left.get(changed_param)
        changed_to = right.get(changed_param)

        delta_frames.append(
            _pair_delta_rows(
                agg_df=fluent_agg_df,
                experiment_a=left["experiment_id"],
                experiment_b=right["experiment_id"],
                changed_param=changed_param,
                changed_from=changed_from,
                changed_to=changed_to,
                metric_set="fluents",
                metric_cols=("tp", "fp", "fn"),
            )
        )
        delta_frames.append(
            _pair_delta_rows(
                agg_df=planning_agg_df,
                experiment_a=left["experiment_id"],
                experiment_b=right["experiment_id"],
                changed_param=changed_param,
                changed_from=changed_from,
                changed_to=right.get(changed_param),
                metric_set="planning",
                metric_cols=("solving_ratio", "false_plans_ratio", "unsolvable_ratio"),
            )
        )

    deltas_df = pd.concat(delta_frames, ignore_index=True) if delta_frames else pd.DataFrame()
    if deltas_df.empty:
        return deltas_df, pd.DataFrame()

    deltas_df["pair_id"] = deltas_df["experiment_a"] + " -> " + deltas_df["experiment_b"]

    summary_agg: Dict[str, Any] = {
        "pair_id": "nunique",
        "delta_abs_score": "mean",
    }
    for col in deltas_df.columns:
        if col.endswith("_delta"):
            summary_agg[col] = ["mean", lambda s: s.abs().mean()]

    summary = (
        deltas_df.groupby(
            ["metric_set", "changed_param", "changed_from", "changed_to", "phase", "gt_rate"],
            dropna=False,
        )
        .agg(summary_agg)
        .reset_index()
    )
    summary.columns = [
        col
        if isinstance(col, str)
        else f"{col[0]}_{col[1]}".replace("<lambda_0>", "abs_mean")
        for col in summary.columns
    ]
    summary = summary.rename(
        columns={
            "pair_id_nunique": "pairs_count",
            "delta_abs_score_mean": "avg_abs_delta_score",
        }
    )

    return deltas_df, summary


def _summary_by_experiment(df: pd.DataFrame, metric_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    subset = df[(df["phase"] == "cleaned") & (df["gt_rate"] == 0)]
    if subset.empty:
        subset = df[df["phase"] == "cleaned"]
    if subset.empty:
        subset = df
    grouped = subset.groupby("experiment_id")[[f"{m}_mean" for m in metric_cols]].mean().reset_index()
    return grouped


def build_interaction_effects(
    *,
    experiments_df: pd.DataFrame,
    fluent_agg_df: pd.DataFrame,
    planning_agg_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    varying_params = varying_parameter_columns(experiments_df)
    if len(varying_params) < 2:
        return pd.DataFrame(), pd.DataFrame()

    planning_summary = _summary_by_experiment(
        planning_agg_df,
        metric_cols=("solving_ratio", "false_plans_ratio", "unsolvable_ratio"),
    )
    fluent_summary = _summary_by_experiment(
        fluent_agg_df,
        metric_cols=("tp", "fp", "fn"),
    )
    if planning_summary.empty and fluent_summary.empty:
        return pd.DataFrame(), pd.DataFrame()

    param_view = experiments_df[["experiment_id", *varying_params]].drop_duplicates()
    effect_rows: List[Dict[str, Any]] = []
    combo_rows: List[Dict[str, Any]] = []

    for p1, p2 in combinations(varying_params, 2):
        for metric_set, summary_df, metric_cols in (
            ("planning", planning_summary, ("solving_ratio", "false_plans_ratio", "unsolvable_ratio")),
            ("fluents", fluent_summary, ("tp", "fp", "fn")),
        ):
            if summary_df.empty:
                continue
            joined = summary_df.merge(param_view, on="experiment_id", how="left")
            grouped = joined.groupby([p1, p2], dropna=False)[[f"{m}_mean" for m in metric_cols]].mean().reset_index()
            if grouped.empty:
                continue

            effect_row: Dict[str, Any] = {
                "metric_set": metric_set,
                "param_a": p1,
                "param_b": p2,
                "combinations_observed": len(grouped),
            }
            for metric in metric_cols:
                col = f"{metric}_mean"
                min_v = grouped[col].min()
                max_v = grouped[col].max()
                effect_row[f"{metric}_range"] = max_v - min_v

            if metric_set == "planning":
                effect_row["combined_range_score"] = (
                    effect_row["solving_ratio_range"]
                    + effect_row["false_plans_ratio_range"]
                    + effect_row["unsolvable_ratio_range"]
                )
                grouped["objective_score"] = (
                    grouped["solving_ratio_mean"]
                    - grouped["false_plans_ratio_mean"]
                    - grouped["unsolvable_ratio_mean"]
                )
            else:
                effect_row["combined_range_score"] = (
                    effect_row["tp_range"] + effect_row["fp_range"] + effect_row["fn_range"]
                )
                grouped["objective_score"] = grouped["tp_mean"] - grouped["fp_mean"] - grouped["fn_mean"]

            best_idx = grouped["objective_score"].idxmax()
            worst_idx = grouped["objective_score"].idxmin()
            best_row = grouped.loc[best_idx]
            worst_row = grouped.loc[worst_idx]
            effect_row["best_combo"] = f"{p1}={best_row[p1]}, {p2}={best_row[p2]}"
            effect_row["worst_combo"] = f"{p1}={worst_row[p1]}, {p2}={worst_row[p2]}"
            effect_rows.append(effect_row)

            temp = grouped.copy()
            temp.insert(0, "metric_set", metric_set)
            temp.insert(1, "param_a", p1)
            temp.insert(2, "param_b", p2)
            combo_rows.append(temp)

    effects_df = pd.DataFrame(effect_rows).sort_values(
        by=["metric_set", "combined_range_score"],
        ascending=[True, False],
    )
    combos_df = pd.concat(combo_rows, ignore_index=True) if combo_rows else pd.DataFrame()
    return effects_df, combos_df

