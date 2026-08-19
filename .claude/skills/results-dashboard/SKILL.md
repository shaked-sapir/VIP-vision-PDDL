---
name: results-dashboard
description: Build, refresh and open the VIP results dashboard and its companion Excel exports from finished experiment directories. Covers benchmark/evaluation/cfm/build_dashboard.py, dashboard_config.yaml (results_root, domains, metrics, error_band, algorithms/modes, exclude_algorithms, simulation prefix, image experiment_dir), the --regen-plots / --refresh-stats / --domains / --embed flags, the CFM_quality_shared trend PNGs and the _grid_fluent_stats.json cache, combine_dashboard_reports and combined_experiment_reports.xlsx, export_dashboard_shared_metrics, cfm_quality_table, cfm_domain_aggregate, fold_filter and per-experiment fully-detailed-report.xlsx. Use when the user asks to build/regenerate/open the dashboard or results_dashboard.html, to add a domain or an algorithm to it, to rebuild after a backfill or a new run, or to debug an empty heatmap cell, a missing baseline series, a blank trend plot or stale numbers.
---

# Results dashboard

One HTML page over every finished experiment: a Simulation tab (a mask × noise
grid per domain) and an Image tab (one experiment per domain), sharing a metric
nav, heatmaps and learning curves.

Everything is driven by one config. Nothing is auto-discovered except the
simulation grid cells, which are globbed under a per-domain prefix.

Run everything from the project root with the venv active:

```
source venv11/bin/activate
```

---

## 1. Preconditions

The dashboard only reads finished artifacts; it never learns or evaluates. Per
fold instance (`<experiment>/testing/fold<F>_numtrajs<N>_gtrate<G>/`) it wants:

| File | Feeds |
|---|---|
| `all_solutions_metrics.json` | every metric except `fluent_patch_count` |
| `conflict_free_solutions_log.json` | `fluent_patch_count`, CFM discovery times |
| `fold_result.json` | baseline series (ROSAME, ROSAME-I, …) and `CDPS_ANCHORED` |
| `original_observations/*.masking_info` | the masked/flipped corruption table |
| `cdps_anchored/` | the anchored CFM set, if that arm ran |

Missing files do not raise. The cell degrades to an empty tile, which is why a
half-written experiment looks like a rendering bug rather than a data gap.
A cell's `status` is computed from the max solution index:

- `missing` — no CFMs at all → **empty tile**
- `single` — exactly one CFM → tile reads *single model (no conflicts)*
- `ok` — two or more CFMs → trend plot

---

## 2. The config is the only thing you edit

`benchmark/evaluation/cfm/dashboard_config.yaml`, all under the top-level
`results_dashboard:` key. Paths are relative to the project root.

| Key | Effect |
|---|---|
| `results_root` | root holding one subfolder per domain. Default `benchmark/running_results` |
| `output_html` | where the page is written. Default `<results_root>/results_dashboard.html` |
| `domains` | which domains appear, **in display order** |
| `error_band` | `std` \| `ci95` \| `minmax`. An unknown value raises at startup |
| `metrics` | list of `{key, label}`. `key` must match the `<key>_trend.png` filenames |
| `algorithms` | `{key, modes}` registry; `key` must equal the `algorithm` field in `fold_result.json` |
| `exclude_algorithms` | series present in the data but hidden from curves, legend and Δ tables |
| `simulation.prefix.<domain>` | experiment-name prefix used to glob that domain's grid cells |
| `image.experiment_dir.<domain>` | explicit path to that domain's single image experiment |

**To change X, touch Y:**

- *add a domain* → append to `domains`, then add its `simulation.prefix` and/or
  `image.experiment_dir` entry. A domain listed with neither simply renders nothing.
- *point at a new grid run* → `simulation.prefix.<domain>`
- *point at a new image run* → `image.experiment_dir.<domain>`
- *add/remove a plotted metric* → `metrics`
- *a new baseline arm shows in the wrong tab* → give it an `algorithms` entry
- *a baseline should not be plotted at all* → `exclude_algorithms`

### Three behaviours that cause most confusion

1. **Grid cells are globbed, not listed.** For each domain the builder scans
   `<results_root>/<domain>/` for directories whose name both starts with the
   configured prefix and matches `mask=<m>__noise=<n>`. Rename a run, or change
   the prefix, and its cells vanish **silently** — no error, the domain just
   stops appearing in the Simulation tab. If a domain has no `prefix` entry at
   all, *every* `mask=…__noise=…` directory under it matches.
2. **`algorithms[].modes` and `exclude_algorithms` are different tools.**
   `modes` restricts a series to the `simulation` or `image` tab; a baseline
   with no entry defaults to both and renders wherever it has data. Only
   `exclude_algorithms` removes it outright. CDPS, its oracle and
   `CDPS_ANCHORED` are ours, appear in both tabs, and need no entry.
3. **`error_band` is baked into the PNGs.** Changing it does nothing until the
   plots are regenerated with `--regen-plots`.

### The grid is discovered, not declared

Nothing fixes the grid's size, and no number in this document should be taken
as the grid. The mask and noise axes are the **sorted union of the values
parsed out of the discovered directory names**, taken across all domains;
`dashboard_config.yaml` never states them. The grid you get is whatever
`benchmark/run_config.yaml` was run with, under `simulation.grid.masking_ps`
and `simulation.grid.noising_ps` — two lists of floats in [0, 1], of any
length. *Illustration only, not the current or required values:*

```yaml
simulation:
  grid:
    masking_ps: [0.0, 0.01, 0.1]      # example — read the real lists from run_config.yaml
    noising_ps: [0.0, 0.1, 0.2]       # example — length and values are both free
```

So **adding a `p_mask` or `p_noise` value and rerunning needs no dashboard
config change at all** — as long as the new cells land under the same
`simulation.prefix.<domain>`, the axis grows and the columns appear by
themselves. Three consequences, whatever the lengths happen to be:

- **The rendered grid is the full cross product**, |masks| × |noises|, not the
  set of runs that exist. Adding one value to each axis grows it
  multiplicatively, so the number of tiles asked for outruns the number of new
  runs.
- **Combinations never run render as `–`**, identical to a combination that ran
  and failed. Before reading a sparse new column as a result, confirm the cells
  exist on disk. Adding an axis value for one domain only will draw that column
  for *every* domain, mostly empty.
- **The corruption table reads `masked` from the first noise column**, falling
  back to the second (`NOISES[0]` then `NOISES[1]`). Inserting a new *lowest*
  noise value re-sorts that axis, so if the new value was not run for some
  domain, that domain's `masked` figure can blank out even though its other
  cells are intact. Cosmetic, but it looks like data loss.

After growing the grid, rebuild with `--regen-plots --refresh-stats`: the new
cells have no trend PNGs and no cache entry.

---

## 3. Build it

```
python -m benchmark.evaluation.cfm.build_dashboard
```

That is the fast path: it reuses the existing trend PNGs and the per-domain
`_grid_fluent_stats.json` cache, and rebuilds only the HTML.

| Flag | Use it when | Cost |
|---|---|---|
| *(none)* | only the page layout, config metadata or algorithm registry changed | fast |
| `--regen-plots` | metric data changed (new folds, a rerun, a backfill), or `error_band` changed | ~2× the fast path; rewrites `evaluation_results/CFM_quality_shared/<key>_trend.png` per cell |
| `--refresh-stats` | observations changed — new folds, or `original_observations/` was rewritten | recomputes `<results_root>/<domain>/_grid_fluent_stats.json`; parses `.masking_info` and grounds predicates, so the most expensive flag |
| `--domains a b` | iterating on one or two domains | proportionally faster — **but see below** |
| `--embed` | you need one shareable file (email, upload) | much larger file, **different filename** |
| `--config PATH` | trying an alternative config without editing the tracked one | — |

One measurement, to give the ratios rather than a budget: at 5 domains × 9
simulation cells each, plus 5 image experiments, a bare build took **28 s** for
a 1.1 MB page, `--regen-plots` **59 s**, and `--regen-plots --domains gripper`
(9 cells) **11 s**. Cost scales with the *total cell count*, so re-time it
rather than extrapolating if the grid or the domain list has grown.

**`--domains` is a filter, not an incremental update.** It narrows the config's
domain list, and the page is then rendered from that narrowed list — so the
output HTML contains *only* those domains and **overwrites the full one**
(1.1 MB → 0.3 MB for a single domain). Always finish an iteration loop with an
unfiltered build before sharing or reading the page.

Typical full refresh after a batch of runs finishes:

```
python -m benchmark.evaluation.cfm.build_dashboard --regen-plots --refresh-stats
```

The stats cache also self-invalidates if any cached entry is missing its
`flipped` value, so an interrupted earlier run will not be trusted.

---

## 4. Then open it

After a **successful** build, resolve the output path. `build_dashboard` has no
`--output` (that belongs to `combine_dashboard_reports`), so there are only two
cases:

- `--embed` → `<output_html stem>_standalone.html`, by default
  `benchmark/running_results/results_dashboard_standalone.html`
- otherwise → `output_html` from the config, by default
  `benchmark/running_results/results_dashboard.html`

The build prints the resolved path and size on its last line — prefer reading
that over reconstructing it. Then:

```
open benchmark/running_results/results_dashboard.html
```

Report the path and the printed size. **Skip the open entirely if the build
failed**, so a stale page from a previous run is never presented as the new one.

---

## 5. Excel exports

These are separate tools; the HTML build never invokes them.

**Per-experiment detailed report** — one workbook for one experiment (run
params, per-fold metrics, correlation, summary). Written to
`<experiment_dir>/evaluation_results/fully-detailed-report.xlsx`.

```
python -m benchmark.evaluation.experiment_report <experiment_dir>
```

**Every dashboard experiment in one workbook** — two sheets, `simulated` and
`imaged`, discovered through the same `dashboard_config.yaml`. Columns are
`domain, experiment_name, p_mask, p_noise, gt_rate` plus the raw CFM columns.

```
python -m benchmark.evaluation.cfm.combine_dashboard_reports --skip-regenerate
```

Without `--skip-regenerate` it **regenerates the per-experiment
`fully-detailed-report.xlsx` for every experiment it finds**, which is the slow
path by a wide margin. Pass it unless the underlying results actually changed.
Default output `<results_root>/combined_experiment_reports.xlsx`; override with
`--output`.

**Cross-domain shared-metrics workbook** — stacks every algorithm row (CDPS,
ROSAME, each `PISAM_MILP_*`) side by side using only schema base fields.

```
python -m benchmark.evaluation.export_dashboard_shared_metrics [--modes simulation|image]
```

Writes `benchmark/evaluation/raw_data/all_domains_shared_metrics.xlsx` plus a
per-domain workbook each; `--out-dir` moves them.

**Pivot-ready CFM-quality table** for a single experiment:

```
python -m benchmark.evaluation.cfm.cfm_quality_table <experiment_root>
```

---

## 6. Per-experiment analyses

Independent of the dashboard, but usually wanted alongside it.

**CFM quality trend** — do later CFMs improve on earlier ones?

```
python -m benchmark.evaluation.cfm.cfm_quality_analysis <experiment_dir>
```

Writes `<experiment_dir>/evaluation_results/CFM_quality/` (own-x plots; the
dashboard uses its own lean `CFM_quality_shared/` copies instead). Report the
instance count, how many instances have ≥ 2 CFMs, the CFM-count distribution
and the PNGs produced. If no instance has ≥ 2 CFMs, say so explicitly.

**Which folds solved anything**

```
python -m benchmark.evaluation.fold_filter <experiment_dir> [--time-threshold 300]
```

Lists folds with at least one model at `solving_ratio > 0`. Present it in two
groups — **Legitimate CFMs** (`solution_index >= 0`) and **Fallback only**
(`solution_index == -1`) — keep the `solving_under_300s=N` column (CFMs found
within `wall_time_so_far <= threshold`), and close with a one-line total:
qualifying folds, how many via CFM, how many via fallback only.

**Merge every CFM into consensus domains**

```
python -m benchmark.evaluation.cfm.cfm_domain_aggregate <experiment_dir> [--output-dir DIR]
```

Scans `testing/**/conflict_free_models/conflict_free_model_*/model.pddl` and
writes six PDDL files to `<experiment_dir>/aggregated_domains/`:
`union_pre_intersect_eff.pddl`, `intersect_all.pddl`, and `vote_0p25` /
`vote_0p50` / `vote_0p75` / `vote_0p90` (literals present in ≥ 25/50/75/90 % of
CFMs), plus `aggregation_summary.json` with the per-action literal breakdown.
Report the number of models aggregated and all output paths; if no CFM models
are found, say so explicitly.

---

## 7. Rebuilding after a backfill or a new arm

`backfill_baseline.py` / `backfill_cdps.py` add algorithm rows to existing cells
without regenerating data. Afterwards:

| Artifact | Stale? | Fix |
|---|---|---|
| `results_dashboard.html` | yes | rebuild |
| `CFM_quality_shared/*_trend.png` | yes, if the backfilled arm is CDPS-family | `--regen-plots` |
| `_grid_fluent_stats.json` | only if `original_observations/` changed | `--refresh-stats` |
| `fully-detailed-report.xlsx` | yes | drop `--skip-regenerate`, or delete the file |

A **new algorithm key** additionally needs a decision in the config: give it an
`algorithms` entry with the right `modes`, or it will render in both tabs
wherever it happens to have data. Its key must match the `algorithm` field in
`fold_result.json` character for character.

---

## 8. Troubleshooting

| Symptom | Likely cause |
|---|---|
| Most domains vanished at once, both tabs | the last build passed `--domains`, which rewrites the page with only those. Rebuild unfiltered |
| One domain missing from the Simulation tab | `simulation.prefix.<domain>` no longer matches the directory names — the glob failed silently. `ls benchmark/running_results/<domain>/` and compare |
| A new p_mask / p_noise column is nearly all `–` | the axis value exists for one domain but was not run for the others; the grid is a cross product, so the column is drawn for everyone |
| A domain's `masked` figure went blank after adding a noise value | the corruption table reads it from the lowest noise column; the re-sorted axis now starts at a value that domain never ran |
| One empty tile in an otherwise full grid | that cell has no CFMs (`status: missing`) — the fold failed or timed out; check its `conflict_free_solutions_log.json` |
| Tile reads *single model (no conflicts)* | one CFM only, so there is no trend to plot. Not an error |
| A baseline is absent from the curves | in `exclude_algorithms`; or its `algorithms[].modes` excludes this tab; or it genuinely has no rows in `fold_result.json` — check in that order |
| The control bar says *no baseline rows found* | the cells were never backfilled with baseline arms |
| A metric column is blank everywhere | the `metrics[].key` does not match the generated `<key>_trend.png` name |
| `ValueError: unknown error_band` | `error_band` is not one of `std` / `ci95` / `minmax` |
| Numbers unchanged after a rerun | the trend PNGs and/or the stats cache were reused — add `--regen-plots --refresh-stats` |
| Error bands look wrong after editing `error_band` | same: bands are baked into the PNGs |
| Huge HTML | `--embed` inlines every chart as base64. Use the plain build for local viewing |

---

## 9. Invariants

- Never hand-edit `results_dashboard.html`. It is fully generated; the next
  build overwrites it. Fix the config or the data.
- Everything on this page is a **ground-truth-based, offline** metric. Never
  quote it as evidence for online model selection — that is
  `observations_reconstruction_score` in `src/plan_denoising/evaluator.py`.
- The plotted `fluent_patch_count` is the **raw** count — `len(fluent_patches)`
  as written to `conflict_free_solutions_log.json` — so it double-charges
  self-cancelling toggle pairs. The corrected `net_fluent_patch_count` sits in
  the same log entries but is not what this page shows. Do not use the
  dashboard to compare repair magnitude across arms; read the net field
  directly (`src/plan_denoising/patch_accounting.py`).
- The builder is read-only with respect to experiments: it writes only the
  HTML, the `CFM_quality_shared/` PNGs and the `_grid_fluent_stats.json` cache.
  If a rebuild appears to change a result, the result changed underneath it.
