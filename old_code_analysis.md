# Old Code Analysis - amlgym_testing.py

## Summary
The refactoring updated the main experiment loop and helper functions, but **reporting/plotting code still uses the old paradigm**.

---

## Old Paradigm vs New Paradigm

| Old Paradigm | New Paradigm |
|--------------|--------------|
| `traj_size` (trajectory size: 1, 3, 5, 7, etc.) | `num_trajectories` (number of full trajectories: 1, 2, 3, 4, 5) |
| `gt_n` (inject GT every n steps) | `gt_rate` (percentage: 0, 10, 25, 50) |
| Truncated trajectories | Full trajectories |

---

## Functions That Need Updating

### 1. `generate_excel_report()` (lines 243-475)
**Problem**: Groups results by `traj_size`, creates sheets named `size=1`, `size=3`, etc.

**Issues**:
- Line 258, 279: `grouped_unclean/cleaned = df.groupby(["domain", "algorithm", "traj_size"])`
- Line 269, 290: Uses `traj_size` column
- Line 304: `size_key = f"{int(row['traj_size'])}__{'unclean' if phase == 'unclean' else ''}"`
- Line 331, 333: Sheet names like `size=1`, `size=3`

**Needs**:
- Replace `traj_size` with `num_trajectories`
- Update sheet names: `numtrajs=1`, `numtrajs=2`, etc.
- Consider also grouping by `gt_rate` (create separate sheets or sub-sections)

---

### 2. `generate_plots()` (lines 477-588)
**Problem**: Plots metrics vs trajectory size

**Issues**:
- Line 481: `def plot_metric_vs_size(df, metric_key, metric_title, save_path, phase_label, domain_label)`
- Line 490: `sub = df[df["algorithm"] == algo].sort_values("traj_size")`
- Line 491: `x = sub["traj_size"]`
- Line 529, 560: Groups by `traj_size`
- Line 540, 571: Uses `traj_size` column
- Line 546, 549, 552: File names like `solving_ratio_vs_traj_size__unclean_({domain_upper}).png`
- Line 577, 580, 583: File names like `solving_ratio_vs_traj_size_({domain_upper}).png`

**Needs**:
- Rename function to `plot_metric_vs_num_trajectories()`
- Replace x-axis from `traj_size` to `num_trajectories`
- Update plot file names: `solving_ratio_vs_num_trajectories.png`
- Consider adding separate plots for different `gt_rate` values

---

### 3. `plot_metric_vs_traj_size_by_n()` (lines 750-854)
**Problem**: Entire function is based on old paradigm

**Issues**:
- Line 750: Function name references `traj_size` and `n`
- Line 752: "How does trajectory size affect metrics for each GT density (n)?"
- Line 759: `results_df: DataFrame with columns: algorithm, traj_size, gt_n, fold, {metric_name}`
- Line 785: `baseline_df = df[df['gt_n'].isna()].copy()`
- Line 788: `n_values = sorted([n for n in df['gt_n'].unique() if pd.notna(n)])`
- Line 818: `n_df = df[df['gt_n'] == n]`
- Line 822-826: Groups by `traj_size`, iterates over trajectory sizes
- Line 829: Label: `f'{algo} (GT every {n})'`
- Line 850: Filename: `{domain_name}_{metric_name}_vs_traj_size_by_n.png`

**Needs**:
- Rename to `plot_metric_vs_num_trajectories_by_gt_rate()`
- Replace all `traj_size` → `num_trajectories`
- Replace all `gt_n` → `gt_rate`
- Update labels: `f'{algo} (GT rate {gt_rate}%)'`
- Update axis labels: "Number of Trajectories" instead of "Trajectory Size"
- Update legend: "GT Rate 10%" instead of "GT every 10 steps"

---

### 4. `plot_metric_vs_n_by_traj_size()` (lines 856-939)
**Problem**: Plots metric vs n for different trajectory sizes

**Issues**:
- Line 856: Function name references `n` and `traj_size`
- Line 860: "Creates one figure with 1x3 subplots (one per trajectory size: 5, 10, 30)"
- Line 864: `results_df: DataFrame with columns: algorithm, traj_size, gt_n, fold, {metric_name}`
- Line 880, 882, 885: Filters by `gt_n`
- Line 890: `available_sizes = [s for s in representative_sizes if s in df['traj_size'].unique()]`
- Line 901: `for idx, traj_size in enumerate(available_sizes)`
- Line 904: `size_df = df[df['traj_size'] == traj_size]`
- Line 910: `grouped = algo_df.groupby('gt_n')[metric_name].agg(['mean', 'std'])`
- Line 923: Title: `f'Trajectory Size = {traj_size}'`
- Line 935: Filename: `{domain_name}_{metric_name}_vs_n_by_traj_size.png`

**Needs**:
- Rename to `plot_metric_vs_gt_rate_by_num_trajectories()`
- Replace all `traj_size` → `num_trajectories`
- Replace all `gt_n` → `gt_rate`
- Update representative values: Instead of sizes [5, 10, 30], use num_trajectories [1, 3, 5]
- Update subplot titles: `f'Number of Trajectories = {num_trajectories}'`
- Update x-axis: "GT Rate (%)" instead of "GT Injection Interval (n)"

---

### 5. `generate_gt_injection_plots()` (lines 941-983)
**Problem**: Calls the above two functions and expects `gt_n` column

**Issues**:
- Line 951: Comment refers to "trajectory size" and "n"
- Line 958: `if 'gt_n' not in df.columns:`
- Line 979: `plot_metric_vs_traj_size_by_n(df, metric, plots_dir, domain_name)`
- Line 982: `plot_metric_vs_n_by_traj_size(df, metric, plots_dir, domain_name)`

**Needs**:
- Update to check for `gt_rate` column instead of `gt_n`
- Call updated plotting functions
- Update comments to reflect new paradigm

---

### 6. Main Loop References (lines 700-722)
**Problem**: Print statements and variable names still reference old paradigm

**Issues**:
- Line 703: `print(f"GENERATING AGGREGATED REPORT FOR TRAJECTORY SIZE = {traj_size}")`
- Line 710: `completed_sizes = sorted(set(r['traj_size'] for r in unclean_results))`
- Line 722: `print(f"✓ Plots updated with results up to size={traj_size}")`

**Needs**:
- Line 703: Change to `num_trajectories`
- Line 710: Change to `num_trajectories`
- Line 722: Change to `num_trajectories`

**NOTE**: This code also appears to be in the wrong location - it references `traj_size` variable which no longer exists after our refactoring!

---

## Data Structure Impact

The refactoring changed what data is stored in results dictionaries. The new `run_single_fold()` in `experiment_helpers.py` should return results with:
- `num_trajectories` field (instead of `traj_size`)
- `gt_rate` field (instead of `gt_n`)

**Check**: Does `experiment_helpers.run_single_fold()` properly set these fields in the returned dictionaries?

---

## Recommended Action Plan

1. **Update experiment_helpers.py** - Ensure results dicts have `num_trajectories` and `gt_rate`
2. **Update generate_excel_report()** - Group by `num_trajectories`, consider `gt_rate` grouping
3. **Update generate_plots()** - Plot vs `num_trajectories` instead of `traj_size`
4. **Rewrite plot_metric_vs_traj_size_by_n()** → `plot_metric_vs_num_trajectories_by_gt_rate()`
5. **Rewrite plot_metric_vs_n_by_traj_size()** → `plot_metric_vs_gt_rate_by_num_trajectories()`
6. **Update generate_gt_injection_plots()** - Call updated functions
7. **Fix main loop print statements** - Lines 703, 710, 722

---

## Critical Issue in Main Loop (Lines 700-722)

These lines reference `traj_size` which **no longer exists** after our refactoring!

```python
print(f"GENERATING AGGREGATED REPORT FOR TRAJECTORY SIZE = {traj_size}")  # ERROR: traj_size undefined
completed_sizes = sorted(set(r['traj_size'] for r in unclean_results))  # ERROR if 'traj_size' not in results
```

This code will **crash** when run! It's leftover from the old loop structure and needs to be either:
- Deleted (if reports should only be generated at the very end)
- Updated to use `num_trajectories` and moved to the correct location in the new loop structure
