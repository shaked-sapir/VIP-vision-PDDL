from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from benchmark.analysis_experiment_compare.aggregate import (
    aggregate_fluent_metrics,
    aggregate_planning_metrics,
)
from benchmark.analysis_experiment_compare.compare import build_group_comparisons
from benchmark.analysis_experiment_compare.discover import discover_experiments
from benchmark.analysis_experiment_compare.export import write_comparison_set, write_csv
from benchmark.analysis_experiment_compare.load_fluents import load_fluent_rows
from benchmark.analysis_experiment_compare.load_planning import load_planning_rows
from benchmark.analysis_experiment_compare.param_effects import (
    build_interaction_effects,
    build_single_param_deltas,
    varying_parameter_columns,
)
from benchmark.analysis_experiment_compare.report import (
    write_interaction_report,
    write_markdown_report,
)


def run(
    root: Path,
    output_dir: Path,
    domain: str,
    model_policy: str,
    baseline_experiment: str | None,
) -> None:
    experiments = discover_experiments(root)
    experiments_df = pd.DataFrame(experiments)

    fluent_raw_df = load_fluent_rows(experiments, model_policy=model_policy)
    planning_raw_df = load_planning_rows(experiments, domain_filter=domain)

    fluent_agg_df = aggregate_fluent_metrics(fluent_raw_df)
    planning_agg_df = aggregate_planning_metrics(planning_raw_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(experiments_df, output_dir / "experiments_index.csv")
    write_csv(fluent_raw_df, output_dir / "fluents_raw.csv")
    write_csv(planning_raw_df, output_dir / "planning_raw.csv")
    write_csv(fluent_agg_df, output_dir / "fluents_aggregated.csv")
    write_csv(planning_agg_df, output_dir / "planning_aggregated.csv")

    fluent_timeout_comp = build_group_comparisons(
        fluent_agg_df,
        group_key="timeout_seconds",
        metric_cols=["tp", "fp", "fn"],
        baseline_experiment=baseline_experiment,
    )
    fluent_weight_comp = build_group_comparisons(
        fluent_agg_df,
        group_key="model_constraint_weight",
        metric_cols=["tp", "fp", "fn"],
        baseline_experiment=baseline_experiment,
    )
    planning_timeout_comp = build_group_comparisons(
        planning_agg_df,
        group_key="timeout_seconds",
        metric_cols=["solving_ratio", "false_plans_ratio", "unsolvable_ratio"],
        baseline_experiment=baseline_experiment,
    )
    planning_weight_comp = build_group_comparisons(
        planning_agg_df,
        group_key="model_constraint_weight",
        metric_cols=["solving_ratio", "false_plans_ratio", "unsolvable_ratio"],
        baseline_experiment=baseline_experiment,
    )

    write_comparison_set(output_dir, "fluents_by_timeout", fluent_timeout_comp)
    write_comparison_set(output_dir, "fluents_by_weight", fluent_weight_comp)
    write_comparison_set(output_dir, "planning_by_timeout", planning_timeout_comp)
    write_comparison_set(output_dir, "planning_by_weight", planning_weight_comp)

    single_param_deltas_df, single_param_summary_df = build_single_param_deltas(
        experiments_df=experiments_df,
        fluent_agg_df=fluent_agg_df,
        planning_agg_df=planning_agg_df,
    )
    write_csv(single_param_deltas_df, output_dir / "single_param_deltas.csv")
    write_csv(single_param_summary_df, output_dir / "single_param_summary.csv")

    interaction_effects_df, interaction_combos_df = build_interaction_effects(
        experiments_df=experiments_df,
        fluent_agg_df=fluent_agg_df,
        planning_agg_df=planning_agg_df,
    )
    write_csv(interaction_effects_df, output_dir / "interaction_effects.csv")
    write_csv(interaction_combos_df, output_dir / "interaction_combos.csv")

    report_path = write_markdown_report(
        output_dir,
        root=root,
        domain=domain,
        model_policy=model_policy,
        baseline_experiment=baseline_experiment,
        experiments_df=experiments_df,
        fluent_raw_df=fluent_raw_df,
        planning_raw_df=planning_raw_df,
        fluent_agg_df=fluent_agg_df,
        planning_agg_df=planning_agg_df,
        fluent_timeout_comp=fluent_timeout_comp,
        fluent_weight_comp=fluent_weight_comp,
        planning_timeout_comp=planning_timeout_comp,
        planning_weight_comp=planning_weight_comp,
    )
    print(f"Wrote markdown report: {report_path}")

    interaction_report_path = write_interaction_report(
        output_dir,
        varying_params=varying_parameter_columns(experiments_df),
        interaction_effects_df=interaction_effects_df,
        interaction_combos_df=interaction_combos_df,
        single_param_summary_df=single_param_summary_df,
    )
    print(f"Wrote interaction report: {interaction_report_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare benchmark experiments by timeout and model constraint weight.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("benchmark/running_results/blocksworld"),
        help="Root directory that contains experiment folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/analysis_experiment_compare/output"),
        help="Directory to write comparison outputs.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="blocksworld",
        help="Domain filter for planning csv rows (default: blocksworld).",
    )
    parser.add_argument(
        "--model-policy",
        type=str,
        choices=["final_model", "all"],
        default="final_model",
        help="How to select fluent metrics.json per fold (default: final_model).",
    )
    parser.add_argument(
        "--baseline-experiment",
        type=str,
        default=None,
        help="Optional explicit experiment_id baseline for delta calculation.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run(
        root=args.root,
        output_dir=args.output_dir,
        domain=args.domain,
        model_policy=args.model_policy,
        baseline_experiment=args.baseline_experiment,
    )

