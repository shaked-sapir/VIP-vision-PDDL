Find all fold directories under the given experiment config directory that have at least one model (CFM or fallback) with solving_ratio > 0.

**Experiment directory**: $ARGUMENTS

## Steps

1. Run the following command from the project root with the venv activated:

```
source venv11/bin/activate && python -m benchmark.evaluation.fold_filter "$ARGUMENTS"
```

2. Present the output clearly, grouped into two sections:
   - **Legitimate CFMs** (solution_index >= 0): folds where a real conflict-free model achieved solving_ratio > 0
   - **Fallback only** (solution_index == -1): folds where only the partial fallback model achieved solving_ratio > 0

   Each row includes `solving_under_300s=N`: the number of CFMs with solving_ratio > 0 whose discovery time (`wall_time_so_far` in `conflict_free_solutions_log.json`) is ≤ 300 seconds.

3. If no folds are found, say so explicitly.

4. At the end, print a one-line summary: total qualifying folds, how many via CFM, how many via fallback only.
