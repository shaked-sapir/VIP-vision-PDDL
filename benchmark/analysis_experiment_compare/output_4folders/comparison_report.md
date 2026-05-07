# Experiment Comparison Report

- Generated at: `2026-05-03T15:14:57.722569`
- Root: `benchmark/data/new_experiments/blocks`
- Domain filter: `blocksworld`
- Fluent model policy: `final_model`
- Baseline override: `None`

## Data Coverage

- Experiments discovered: **4**
- Fluent raw rows: **1920**
- Planning raw rows: **3840**
- Fluent aggregated rows: **384**
- Planning aggregated rows: **768**

## Experiments Index

| experiment_id | date_tag | timeout_seconds | model_constraint_weight | mode | has_run_params |
| --- | --- | --- | --- | --- | --- |
| dfs_timeout=180s__300steps__03052026 | 03052026 | 180 | 1.0 | masked | True |
| dfs_timeout=180s__300steps__28042026 | 28042026 | 180 | 0.0 | masked | True |
| dfs_timeout=60s__300steps__02052026 | 02052026 | 60 | 1.0 | masked | True |
| dfs_timeout=60s__300steps__28042026 | 28042026 | 60 | 0.0 | masked | True |

## Fluent TP/FP/FN Comparisons

### By Timeout

| timeout_seconds | phase | num_trajectories | gt_rate | metric | dfs_timeout=60s__300steps__02052026 | dfs_timeout=60s__300steps__28042026 | dfs_timeout=180s__300steps__03052026 | dfs_timeout=180s__300steps__28042026 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 60 | cleaned | 1 | 0 | fn_mean | 31.6 | 32.6 |  |  |
| 60 | cleaned | 1 | 0 | fp_mean | 29.2 | 32.8 |  |  |
| 60 | cleaned | 1 | 0 | tp_mean | 107.4 | 106.4 |  |  |
| 60 | cleaned | 1 | 10 | fn_mean | 2.8 | 2.8 |  |  |
| 60 | cleaned | 1 | 10 | fp_mean | 0.0 | 0.0 |  |  |
| 60 | cleaned | 1 | 10 | tp_mean | 136.2 | 136.2 |  |  |
| 60 | cleaned | 1 | 25 | fn_mean | 2.0 | 2.0 |  |  |
| 60 | cleaned | 1 | 25 | fp_mean | 0.0 | 0.0 |  |  |
| 60 | cleaned | 1 | 25 | tp_mean | 137.0 | 137.0 |  |  |
| 60 | cleaned | 1 | 50 | fn_mean | 0.4 | 0.4 |  |  |
| 60 | cleaned | 1 | 50 | fp_mean | 0.0 | 0.0 |  |  |
| 60 | cleaned | 1 | 50 | tp_mean | 138.6 | 138.6 |  |  |
| 60 | cleaned | 1 | 75 | fn_mean | 0.4 | 0.4 |  |  |
| 60 | cleaned | 1 | 75 | fp_mean | 0.0 | 0.0 |  |  |
| 60 | cleaned | 1 | 75 | tp_mean | 138.6 | 138.6 |  |  |
| 60 | cleaned | 1 | 100 | fn_mean | 0.0 | 0.0 |  |  |
| 60 | cleaned | 1 | 100 | fp_mean | 0.0 | 0.0 |  |  |
| 60 | cleaned | 1 | 100 | tp_mean | 139.0 | 139.0 |  |  |
| 60 | cleaned | 2 | 0 | fn_mean | 54.4 | 53.0 |  |  |
| 60 | cleaned | 2 | 0 | fp_mean | 61.6 | 55.0 |  |  |
| 60 | cleaned | 2 | 0 | tp_mean | 224.8 | 226.2 |  |  |
| 60 | cleaned | 2 | 10 | fn_mean | 5.8 | 5.8 |  |  |
| 60 | cleaned | 2 | 10 | fp_mean | 0.0 | 0.0 |  |  |
| 60 | cleaned | 2 | 10 | tp_mean | 273.4 | 273.4 |  |  |
| 60 | cleaned | 2 | 25 | fn_mean | 3.8 | 3.8 |  |  |
| 60 | cleaned | 2 | 25 | fp_mean | 0.0 | 0.0 |  |  |
| 60 | cleaned | 2 | 25 | tp_mean | 275.4 | 275.4 |  |  |
| 60 | cleaned | 2 | 50 | fn_mean | 1.2 | 1.2 |  |  |
| 60 | cleaned | 2 | 50 | fp_mean | 0.0 | 0.0 |  |  |
| 60 | cleaned | 2 | 50 | tp_mean | 278.0 | 278.0 |  |  |
| 60 | cleaned | 2 | 75 | fn_mean | 0.4 | 0.4 |  |  |
| 60 | cleaned | 2 | 75 | fp_mean | 0.0 | 0.0 |  |  |
| 60 | cleaned | 2 | 75 | tp_mean | 278.8 | 278.8 |  |  |
| 60 | cleaned | 2 | 100 | fn_mean | 0.0 | 0.0 |  |  |
| 60 | cleaned | 2 | 100 | fp_mean | 0.0 | 0.0 |  |  |
| 60 | cleaned | 2 | 100 | tp_mean | 279.2 | 279.2 |  |  |
| 60 | cleaned | 3 | 0 | fn_mean | 61.4 | 68.2 |  |  |
| 60 | cleaned | 3 | 0 | fp_mean | 79.4 | 74.8 |  |  |
| 60 | cleaned | 3 | 0 | tp_mean | 311.6 | 304.8 |  |  |
| 60 | cleaned | 3 | 10 | fn_mean | 6.8 | 6.8 |  |  |

_... showing first 40 of 576 rows._

#### Top Absolute Delta Rows (Timeout Grouping)

| baseline_experiment | experiment_id | phase | num_trajectories | gt_rate | abs_delta_score | tp_delta_vs_baseline | fp_delta_vs_baseline | fn_delta_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dfs_timeout=180s__300steps__03052026 | dfs_timeout=180s__300steps__28042026 | cleaned | 8 | 0 | 54.400000000000006 | -2.0 | -50.400000000000006 | 2.0 |
| dfs_timeout=180s__300steps__03052026 | dfs_timeout=180s__300steps__28042026 | cleaned | 6 | 0 | 35.199999999999946 | 11.799999999999955 | -11.599999999999994 | -11.799999999999997 |
| dfs_timeout=180s__300steps__03052026 | dfs_timeout=180s__300steps__28042026 | cleaned | 4 | 0 | 20.399999999999977 | -5.199999999999989 | 10.0 | 5.199999999999989 |
| dfs_timeout=180s__300steps__03052026 | dfs_timeout=180s__300steps__28042026 | cleaned | 3 | 0 | 18.59999999999998 | -5.399999999999977 | 7.799999999999997 | 5.400000000000006 |
| dfs_timeout=60s__300steps__02052026 | dfs_timeout=60s__300steps__28042026 | cleaned | 3 | 0 | 18.200000000000024 | -6.800000000000011 | -4.6000000000000085 | 6.800000000000004 |
| dfs_timeout=60s__300steps__02052026 | dfs_timeout=60s__300steps__28042026 | cleaned | 7 | 0 | 17.200000000000017 | 7.0 | -3.200000000000017 | -7.0 |
| dfs_timeout=60s__300steps__02052026 | dfs_timeout=60s__300steps__28042026 | cleaned | 6 | 0 | 16.79999999999997 | -1.7999999999999545 | -13.200000000000017 | 1.7999999999999972 |
| dfs_timeout=60s__300steps__02052026 | dfs_timeout=60s__300steps__28042026 | cleaned | 4 | 0 | 13.000000000000043 | -5.400000000000034 | 2.200000000000003 | 5.400000000000006 |
| dfs_timeout=60s__300steps__02052026 | dfs_timeout=60s__300steps__28042026 | cleaned | 8 | 0 | 13.0 | 1.0 | -11.0 | -1.0 |
| dfs_timeout=180s__300steps__03052026 | dfs_timeout=180s__300steps__28042026 | cleaned | 7 | 0 | 12.399999999999991 | 1.3999999999999773 | -9.600000000000023 | -1.3999999999999915 |

### By Model Constraint Weight

| model_constraint_weight | phase | num_trajectories | gt_rate | metric | dfs_timeout=180s__300steps__28042026 | dfs_timeout=60s__300steps__28042026 | dfs_timeout=180s__300steps__03052026 | dfs_timeout=60s__300steps__02052026 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0 | cleaned | 1 | 0 | fn_mean | 29.0 | 32.6 |  |  |
| 0.0 | cleaned | 1 | 0 | fp_mean | 30.0 | 32.8 |  |  |
| 0.0 | cleaned | 1 | 0 | tp_mean | 110.0 | 106.4 |  |  |
| 0.0 | cleaned | 1 | 10 | fn_mean | 2.8 | 2.8 |  |  |
| 0.0 | cleaned | 1 | 10 | fp_mean | 0.0 | 0.0 |  |  |
| 0.0 | cleaned | 1 | 10 | tp_mean | 136.2 | 136.2 |  |  |
| 0.0 | cleaned | 1 | 25 | fn_mean | 2.0 | 2.0 |  |  |
| 0.0 | cleaned | 1 | 25 | fp_mean | 0.0 | 0.0 |  |  |
| 0.0 | cleaned | 1 | 25 | tp_mean | 137.0 | 137.0 |  |  |
| 0.0 | cleaned | 1 | 50 | fn_mean | 0.4 | 0.4 |  |  |
| 0.0 | cleaned | 1 | 50 | fp_mean | 0.0 | 0.0 |  |  |
| 0.0 | cleaned | 1 | 50 | tp_mean | 138.6 | 138.6 |  |  |
| 0.0 | cleaned | 1 | 75 | fn_mean | 0.4 | 0.4 |  |  |
| 0.0 | cleaned | 1 | 75 | fp_mean | 0.0 | 0.0 |  |  |
| 0.0 | cleaned | 1 | 75 | tp_mean | 138.6 | 138.6 |  |  |
| 0.0 | cleaned | 1 | 100 | fn_mean | 0.0 | 0.0 |  |  |
| 0.0 | cleaned | 1 | 100 | fp_mean | 0.0 | 0.0 |  |  |
| 0.0 | cleaned | 1 | 100 | tp_mean | 139.0 | 139.0 |  |  |
| 0.0 | cleaned | 2 | 0 | fn_mean | 53.8 | 53.0 |  |  |
| 0.0 | cleaned | 2 | 0 | fp_mean | 62.4 | 55.0 |  |  |
| 0.0 | cleaned | 2 | 0 | tp_mean | 225.4 | 226.2 |  |  |
| 0.0 | cleaned | 2 | 10 | fn_mean | 5.8 | 5.8 |  |  |
| 0.0 | cleaned | 2 | 10 | fp_mean | 0.0 | 0.0 |  |  |
| 0.0 | cleaned | 2 | 10 | tp_mean | 273.4 | 273.4 |  |  |
| 0.0 | cleaned | 2 | 25 | fn_mean | 3.8 | 3.8 |  |  |
| 0.0 | cleaned | 2 | 25 | fp_mean | 0.0 | 0.0 |  |  |
| 0.0 | cleaned | 2 | 25 | tp_mean | 275.4 | 275.4 |  |  |
| 0.0 | cleaned | 2 | 50 | fn_mean | 1.2 | 1.2 |  |  |
| 0.0 | cleaned | 2 | 50 | fp_mean | 0.0 | 0.0 |  |  |
| 0.0 | cleaned | 2 | 50 | tp_mean | 278.0 | 278.0 |  |  |
| 0.0 | cleaned | 2 | 75 | fn_mean | 0.4 | 0.4 |  |  |
| 0.0 | cleaned | 2 | 75 | fp_mean | 0.0 | 0.0 |  |  |
| 0.0 | cleaned | 2 | 75 | tp_mean | 278.8 | 278.8 |  |  |
| 0.0 | cleaned | 2 | 100 | fn_mean | 0.0 | 0.0 |  |  |
| 0.0 | cleaned | 2 | 100 | fp_mean | 0.0 | 0.0 |  |  |
| 0.0 | cleaned | 2 | 100 | tp_mean | 279.2 | 279.2 |  |  |
| 0.0 | cleaned | 3 | 0 | fn_mean | 70.4 | 68.2 |  |  |
| 0.0 | cleaned | 3 | 0 | fp_mean | 91.6 | 74.8 |  |  |
| 0.0 | cleaned | 3 | 0 | tp_mean | 302.6 | 304.8 |  |  |
| 0.0 | cleaned | 3 | 10 | fn_mean | 6.8 | 6.8 |  |  |

_... showing first 40 of 576 rows._

#### Top Absolute Delta Rows (Weight Grouping)

| baseline_experiment | experiment_id | phase | num_trajectories | gt_rate | abs_delta_score | tp_delta_vs_baseline | fp_delta_vs_baseline | fn_delta_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dfs_timeout=180s__300steps__28042026 | dfs_timeout=60s__300steps__28042026 | cleaned | 7 | 0 | 78.60000000000007 | 25.800000000000068 | -27.0 | -25.799999999999997 |
| dfs_timeout=60s__300steps__02052026 | dfs_timeout=180s__300steps__03052026 | cleaned | 7 | 0 | 73.80000000000004 | -20.200000000000045 | 33.400000000000006 | 20.19999999999999 |
| dfs_timeout=180s__300steps__28042026 | dfs_timeout=60s__300steps__28042026 | cleaned | 6 | 0 | 66.99999999999996 | -12.799999999999955 | -41.400000000000006 | 12.799999999999997 |
| dfs_timeout=60s__300steps__02052026 | dfs_timeout=180s__300steps__03052026 | cleaned | 8 | 0 | 56.79999999999998 | -8.399999999999977 | 40.0 | 8.400000000000006 |
| dfs_timeout=60s__300steps__02052026 | dfs_timeout=180s__300steps__03052026 | cleaned | 6 | 0 | 41.399999999999935 | -0.7999999999999545 | 39.79999999999998 | 0.7999999999999972 |
| dfs_timeout=60s__300steps__02052026 | dfs_timeout=180s__300steps__03052026 | cleaned | 5 | 0 | 32.39999999999999 | -11.0 | 10.399999999999991 | 11.0 |
| dfs_timeout=180s__300steps__28042026 | dfs_timeout=60s__300steps__28042026 | cleaned | 8 | 0 | 23.399999999999977 | 11.399999999999977 | -0.5999999999999943 | -11.400000000000006 |
| dfs_timeout=180s__300steps__28042026 | dfs_timeout=60s__300steps__28042026 | cleaned | 5 | 0 | 22.60000000000001 | 5.600000000000023 | -11.399999999999991 | -5.599999999999994 |
| dfs_timeout=180s__300steps__28042026 | dfs_timeout=60s__300steps__28042026 | cleaned | 3 | 0 | 21.19999999999999 | 2.1999999999999886 | -16.799999999999997 | -2.200000000000003 |
| dfs_timeout=180s__300steps__28042026 | dfs_timeout=60s__300steps__28042026 | cleaned | 4 | 0 | 18.399999999999963 | 6.399999999999977 | -5.599999999999994 | -6.3999999999999915 |

## Planning Ratio Comparisons (Solved / False Plans / Unsolvable)

### By Timeout

| timeout_seconds | phase | num_trajectories | gt_rate | metric | dfs_timeout=60s__300steps__02052026 | dfs_timeout=60s__300steps__28042026 | dfs_timeout=180s__300steps__03052026 | dfs_timeout=180s__300steps__28042026 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 60 | cleaned | 1 | 0 | false_plans_ratio_mean | 0.0 | 0.2 |  |  |
| 60 | cleaned | 1 | 0 | solving_ratio_mean | 0.3 | 0.2 |  |  |
| 60 | cleaned | 1 | 0 | unsolvable_ratio_mean | 0.7 | 0.6 |  |  |
| 60 | cleaned | 1 | 10 | false_plans_ratio_mean | 0.0 | 0.0 |  |  |
| 60 | cleaned | 1 | 10 | solving_ratio_mean | 1.0 | 1.0 |  |  |
| 60 | cleaned | 1 | 10 | unsolvable_ratio_mean | 0.0 | 0.0 |  |  |
| 60 | cleaned | 1 | 25 | false_plans_ratio_mean | 0.0 | 0.0 |  |  |
| 60 | cleaned | 1 | 25 | solving_ratio_mean | 1.0 | 1.0 |  |  |
| 60 | cleaned | 1 | 25 | unsolvable_ratio_mean | 0.0 | 0.0 |  |  |
| 60 | cleaned | 1 | 50 | false_plans_ratio_mean | 0.0 | 0.0 |  |  |
| 60 | cleaned | 1 | 50 | solving_ratio_mean | 1.0 | 1.0 |  |  |
| 60 | cleaned | 1 | 50 | unsolvable_ratio_mean | 0.0 | 0.0 |  |  |
| 60 | cleaned | 1 | 75 | false_plans_ratio_mean | 0.0 | 0.0 |  |  |
| 60 | cleaned | 1 | 75 | solving_ratio_mean | 1.0 | 1.0 |  |  |
| 60 | cleaned | 1 | 75 | unsolvable_ratio_mean | 0.0 | 0.0 |  |  |
| 60 | cleaned | 1 | 100 | false_plans_ratio_mean | 0.0 | 0.0 |  |  |
| 60 | cleaned | 1 | 100 | solving_ratio_mean | 1.0 | 1.0 |  |  |
| 60 | cleaned | 1 | 100 | unsolvable_ratio_mean | 0.0 | 0.0 |  |  |
| 60 | cleaned | 2 | 0 | false_plans_ratio_mean | 0.2 | 0.2 |  |  |
| 60 | cleaned | 2 | 0 | solving_ratio_mean | 0.2 | 0.2 |  |  |
| 60 | cleaned | 2 | 0 | unsolvable_ratio_mean | 0.6 | 0.6 |  |  |
| 60 | cleaned | 2 | 10 | false_plans_ratio_mean | 0.0 | 0.0 |  |  |
| 60 | cleaned | 2 | 10 | solving_ratio_mean | 1.0 | 1.0 |  |  |
| 60 | cleaned | 2 | 10 | unsolvable_ratio_mean | 0.0 | 0.0 |  |  |
| 60 | cleaned | 2 | 25 | false_plans_ratio_mean | 0.0 | 0.0 |  |  |
| 60 | cleaned | 2 | 25 | solving_ratio_mean | 1.0 | 1.0 |  |  |
| 60 | cleaned | 2 | 25 | unsolvable_ratio_mean | 0.0 | 0.0 |  |  |
| 60 | cleaned | 2 | 50 | false_plans_ratio_mean | 0.0 | 0.0 |  |  |
| 60 | cleaned | 2 | 50 | solving_ratio_mean | 1.0 | 1.0 |  |  |
| 60 | cleaned | 2 | 50 | unsolvable_ratio_mean | 0.0 | 0.0 |  |  |
| 60 | cleaned | 2 | 75 | false_plans_ratio_mean | 0.0 | 0.0 |  |  |
| 60 | cleaned | 2 | 75 | solving_ratio_mean | 1.0 | 1.0 |  |  |
| 60 | cleaned | 2 | 75 | unsolvable_ratio_mean | 0.0 | 0.0 |  |  |
| 60 | cleaned | 2 | 100 | false_plans_ratio_mean | 0.0 | 0.0 |  |  |
| 60 | cleaned | 2 | 100 | solving_ratio_mean | 1.0 | 1.0 |  |  |
| 60 | cleaned | 2 | 100 | unsolvable_ratio_mean | 0.0 | 0.0 |  |  |
| 60 | cleaned | 3 | 0 | false_plans_ratio_mean | 0.7 | 0.4 |  |  |
| 60 | cleaned | 3 | 0 | solving_ratio_mean | 0.2 | 0.2 |  |  |
| 60 | cleaned | 3 | 0 | unsolvable_ratio_mean | 0.1 | 0.4 |  |  |
| 60 | cleaned | 3 | 10 | false_plans_ratio_mean | 0.0 | 0.0 |  |  |

_... showing first 40 of 576 rows._

#### Top Absolute Delta Rows (Timeout Grouping)

| baseline_experiment | experiment_id | phase | num_trajectories | gt_rate | abs_delta_score | solving_ratio_delta_vs_baseline | false_plans_ratio_delta_vs_baseline | unsolvable_ratio_delta_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dfs_timeout=60s__300steps__02052026 | dfs_timeout=60s__300steps__28042026 | unclean | 6 | 0 | 2.0 | 0.0 | -1.0 | 1.0 |
| dfs_timeout=180s__300steps__03052026 | dfs_timeout=180s__300steps__28042026 | unclean | 6 | 0 | 2.0 | 0.0 | 1.0 | -1.0 |
| dfs_timeout=180s__300steps__03052026 | dfs_timeout=180s__300steps__03052026 | unclean | 6 | 0 | 2.0 | 0.0 | 1.0 | -1.0 |
| dfs_timeout=180s__300steps__03052026 | dfs_timeout=180s__300steps__28042026 | unclean | 6 | 0 | 2.0 | 0.0 | -1.0 | 1.0 |
| dfs_timeout=180s__300steps__03052026 | dfs_timeout=180s__300steps__03052026 | unclean | 6 | 0 | 2.0 | 0.0 | -1.0 | 1.0 |
| dfs_timeout=180s__300steps__03052026 | dfs_timeout=180s__300steps__03052026 | unclean | 3 | 0 | 1.8 | 0.19999999999999998 | -0.9 | 0.7 |
| dfs_timeout=60s__300steps__02052026 | dfs_timeout=60s__300steps__28042026 | unclean | 3 | 0 | 1.8 | 0.15 | -0.9 | 0.75 |
| dfs_timeout=180s__300steps__03052026 | dfs_timeout=180s__300steps__03052026 | unclean | 3 | 0 | 1.8 | -0.19999999999999998 | 0.9 | -0.7 |
| dfs_timeout=180s__300steps__03052026 | dfs_timeout=180s__300steps__28042026 | unclean | 3 | 0 | 1.8 | -0.19999999999999998 | 0.9 | -0.7 |
| dfs_timeout=60s__300steps__02052026 | dfs_timeout=60s__300steps__02052026 | cleaned | 3 | 0 | 1.7999999999999998 | 0.2 | 0.7 | -0.9 |

### By Model Constraint Weight

| model_constraint_weight | phase | num_trajectories | gt_rate | metric | dfs_timeout=180s__300steps__28042026 | dfs_timeout=60s__300steps__28042026 | dfs_timeout=180s__300steps__03052026 | dfs_timeout=60s__300steps__02052026 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0 | cleaned | 1 | 0 | false_plans_ratio_mean | 0.3 | 0.2 |  |  |
| 0.0 | cleaned | 1 | 0 | solving_ratio_mean | 0.2 | 0.2 |  |  |
| 0.0 | cleaned | 1 | 0 | unsolvable_ratio_mean | 0.5 | 0.6 |  |  |
| 0.0 | cleaned | 1 | 10 | false_plans_ratio_mean | 0.0 | 0.0 |  |  |
| 0.0 | cleaned | 1 | 10 | solving_ratio_mean | 1.0 | 1.0 |  |  |
| 0.0 | cleaned | 1 | 10 | unsolvable_ratio_mean | 0.0 | 0.0 |  |  |
| 0.0 | cleaned | 1 | 25 | false_plans_ratio_mean | 0.0 | 0.0 |  |  |
| 0.0 | cleaned | 1 | 25 | solving_ratio_mean | 1.0 | 1.0 |  |  |
| 0.0 | cleaned | 1 | 25 | unsolvable_ratio_mean | 0.0 | 0.0 |  |  |
| 0.0 | cleaned | 1 | 50 | false_plans_ratio_mean | 0.0 | 0.0 |  |  |
| 0.0 | cleaned | 1 | 50 | solving_ratio_mean | 1.0 | 1.0 |  |  |
| 0.0 | cleaned | 1 | 50 | unsolvable_ratio_mean | 0.0 | 0.0 |  |  |
| 0.0 | cleaned | 1 | 75 | false_plans_ratio_mean | 0.0 | 0.0 |  |  |
| 0.0 | cleaned | 1 | 75 | solving_ratio_mean | 1.0 | 1.0 |  |  |
| 0.0 | cleaned | 1 | 75 | unsolvable_ratio_mean | 0.0 | 0.0 |  |  |
| 0.0 | cleaned | 1 | 100 | false_plans_ratio_mean | 0.0 | 0.0 |  |  |
| 0.0 | cleaned | 1 | 100 | solving_ratio_mean | 1.0 | 1.0 |  |  |
| 0.0 | cleaned | 1 | 100 | unsolvable_ratio_mean | 0.0 | 0.0 |  |  |
| 0.0 | cleaned | 2 | 0 | false_plans_ratio_mean | 0.4 | 0.2 |  |  |
| 0.0 | cleaned | 2 | 0 | solving_ratio_mean | 0.2 | 0.2 |  |  |
| 0.0 | cleaned | 2 | 0 | unsolvable_ratio_mean | 0.4 | 0.6 |  |  |
| 0.0 | cleaned | 2 | 10 | false_plans_ratio_mean | 0.0 | 0.0 |  |  |
| 0.0 | cleaned | 2 | 10 | solving_ratio_mean | 1.0 | 1.0 |  |  |
| 0.0 | cleaned | 2 | 10 | unsolvable_ratio_mean | 0.0 | 0.0 |  |  |
| 0.0 | cleaned | 2 | 25 | false_plans_ratio_mean | 0.0 | 0.0 |  |  |
| 0.0 | cleaned | 2 | 25 | solving_ratio_mean | 1.0 | 1.0 |  |  |
| 0.0 | cleaned | 2 | 25 | unsolvable_ratio_mean | 0.0 | 0.0 |  |  |
| 0.0 | cleaned | 2 | 50 | false_plans_ratio_mean | 0.0 | 0.0 |  |  |
| 0.0 | cleaned | 2 | 50 | solving_ratio_mean | 1.0 | 1.0 |  |  |
| 0.0 | cleaned | 2 | 50 | unsolvable_ratio_mean | 0.0 | 0.0 |  |  |
| 0.0 | cleaned | 2 | 75 | false_plans_ratio_mean | 0.0 | 0.0 |  |  |
| 0.0 | cleaned | 2 | 75 | solving_ratio_mean | 1.0 | 1.0 |  |  |
| 0.0 | cleaned | 2 | 75 | unsolvable_ratio_mean | 0.0 | 0.0 |  |  |
| 0.0 | cleaned | 2 | 100 | false_plans_ratio_mean | 0.0 | 0.0 |  |  |
| 0.0 | cleaned | 2 | 100 | solving_ratio_mean | 1.0 | 1.0 |  |  |
| 0.0 | cleaned | 2 | 100 | unsolvable_ratio_mean | 0.0 | 0.0 |  |  |
| 0.0 | cleaned | 3 | 0 | false_plans_ratio_mean | 0.6 | 0.4 |  |  |
| 0.0 | cleaned | 3 | 0 | solving_ratio_mean | 0.2 | 0.2 |  |  |
| 0.0 | cleaned | 3 | 0 | unsolvable_ratio_mean | 0.2 | 0.4 |  |  |
| 0.0 | cleaned | 3 | 10 | false_plans_ratio_mean | 0.0 | 0.0 |  |  |

_... showing first 40 of 576 rows._

#### Top Absolute Delta Rows (Weight Grouping)

| baseline_experiment | experiment_id | phase | num_trajectories | gt_rate | abs_delta_score | solving_ratio_delta_vs_baseline | false_plans_ratio_delta_vs_baseline | unsolvable_ratio_delta_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dfs_timeout=180s__300steps__28042026 | dfs_timeout=180s__300steps__28042026 | unclean | 6 | 0 | 2.0 | 0.0 | 1.0 | -1.0 |
| dfs_timeout=180s__300steps__28042026 | dfs_timeout=180s__300steps__28042026 | unclean | 6 | 0 | 2.0 | 0.0 | -1.0 | 1.0 |
| dfs_timeout=60s__300steps__02052026 | dfs_timeout=180s__300steps__03052026 | unclean | 6 | 0 | 2.0 | 0.0 | -1.0 | 1.0 |
| dfs_timeout=180s__300steps__28042026 | dfs_timeout=60s__300steps__28042026 | unclean | 6 | 0 | 2.0 | 0.0 | 1.0 | -1.0 |
| dfs_timeout=180s__300steps__28042026 | dfs_timeout=60s__300steps__28042026 | unclean | 6 | 0 | 2.0 | 0.0 | -1.0 | 1.0 |
| dfs_timeout=60s__300steps__02052026 | dfs_timeout=180s__300steps__03052026 | unclean | 3 | 0 | 1.8 | 0.19999999999999998 | -0.9 | 0.7 |
| dfs_timeout=180s__300steps__28042026 | dfs_timeout=60s__300steps__28042026 | unclean | 3 | 0 | 1.8 | 0.15 | -0.9 | 0.75 |
| dfs_timeout=60s__300steps__02052026 | dfs_timeout=60s__300steps__02052026 | cleaned | 3 | 0 | 1.7999999999999998 | -0.2 | -0.7 | 0.9 |
| dfs_timeout=60s__300steps__02052026 | dfs_timeout=60s__300steps__02052026 | cleaned | 3 | 0 | 1.7999999999999998 | 0.2 | 0.7 | -0.9 |
| dfs_timeout=60s__300steps__02052026 | dfs_timeout=180s__300steps__03052026 | unclean | 3 | 0 | 1.6 | -0.1 | 0.9 | -0.6 |
