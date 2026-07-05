# Experiment Comparison Toolkit (Isolated)

This folder provides a standalone pipeline to compare experiment result folders under:

- `benchmark/running_results/blocksworld`

It does not modify benchmark runtime code.

## What it compares

Across experiment folders (dated runs), grouped by:

1. same `timeout_seconds`
2. same `model_constraint_weight` (defaults to `0.0` when `run_params.json` is missing)

Metrics compared over:

- `num_trajectories`
- `gt_rate`
- `phase` (`unclean` / `cleaned`)

### Fluent-count metrics

From fold `metrics.json` files:

- `tp`, `fp`, `fn` (plus precision/recall in raw/aggregated exports)

### Planning outcome ratios

From `evaluation_results/results_*_combined_timeout*.csv`:

- `solving_ratio`
- `false_plans_ratio`
- `unsolvable_ratio`

## Usage

From repo root:

```bash
python -m benchmark.analysis_experiment_compare.main \
  --root benchmark/running_results/blocksworld \
  --output-dir benchmark/analysis_experiment_compare/output \
  --domain blocksworld \
  --model-policy final_model
```

Optional explicit baseline:

```bash
python -m benchmark.analysis_experiment_compare.main \
  --baseline-experiment "dfs_timeout=60s__300steps__02052026"
```

## Outputs

The output directory contains:

- `experiments_index.csv`
- `fluents_raw.csv`
- `planning_raw.csv`
- `fluents_aggregated.csv`
- `planning_aggregated.csv`
- `fluents_by_timeout_wide.csv`
- `fluents_by_timeout_delta.csv`
- `fluents_by_weight_wide.csv`
- `fluents_by_weight_delta.csv`
- `planning_by_timeout_wide.csv`
- `planning_by_timeout_delta.csv`
- `planning_by_weight_wide.csv`
- `planning_by_weight_delta.csv`
- `single_param_deltas.csv` (experiment pairs that differ by exactly one run parameter)
- `single_param_summary.csv` (aggregated impact of single-parameter changes)
- `interaction_effects.csv` (ranked parameter-pair interaction effect sizes)
- `interaction_combos.csv` (observed performance per parameter pair combination)
- `comparison_report.md` (single consolidated markdown summary)
- `interaction_report.md` (focused report for single-param and interaction analyses)

## Notes

- Baseline for delta is earliest `date_tag` in each comparison group unless overridden.
- Fluent extraction uses `final_model` if present per fold; otherwise first conflict-free model.

