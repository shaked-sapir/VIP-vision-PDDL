# Large-corpora sweep on the BGU SLURM cluster

Separate from `scripts/cluster/`, which drives the small-data sweep off
`benchmark/run_config.yaml`. Nothing here touches those files, and nothing there
reads these.

Outputs land where `benchmark_runner` would put them locally —
`benchmark/running_results/{domain}/{run_name}__mask={p}__noise={q}/` — so
`collect_results`, `experiment_report` and the dashboards work unchanged.

## Files

| File | Role |
|---|---|
| `make_manifest.py` | `run_config_large.yaml` → `manifest.csv`, one row per `(domain, mask, noise)` cell |
| `run_cell.sbatch` | Array template; one task = one cell, running the whole L sweep and every arm |
| `submit.sh` | Computes the array range and submits. `DRY_RUN=1` previews |

## One cell per job, L inside the job

A row is a `(domain, mask, noise)` cell, **not** a `(cell, L)` pair. With 5
domains × 3 masks × 3 noises that is **45 jobs**, each running
`num_trajectories: [10, 50, 100, 500, 2000]` and all five arms.

That is deliberate. A wall-clock learning budget is hardware-dependent, so
every training size and every arm of a cell must land on the same node and the
same CPU model to be comparable. Splitting L across jobs would scatter them.

It also means S_test is computed once per fold and reused across all five L
values (it is cached in `fold{N}_gtrate{R}_shared/`), instead of once per
`(fold, L)`.

## Why this drives `benchmark_runner`, not `experiment_runner`

The arms need the `pisam_milp:` block and its `gt_anchoring` ablation, which only
`benchmark_runner` expands; `experiment_runner`'s CLI cannot express them. The
sbatch therefore passes a single cell through the selectors:

```bash
python -m benchmark.benchmark_runner --config benchmark/run_config_large.yaml \
    --domains "$DOMAIN" --only-mask "$P_MASK" --only-noise "$P_NOISE"
```

`--only-mask` / `--only-noise` default to the whole axis, so omitting them is
exactly the behaviour that existed before they were added. An unknown value
raises rather than silently running the full sweep.

The config stays the single source of truth for the arms, the L list, the
patience rule and the subset size.

## Arms

Five, from one config:

| arm | source |
|---|---|
| `PISAM_MILP_LOOP` (gt=init_only) | `pisam_milp_loop` + `ablations.gt_anchoring` |
| `PISAM_MILP_LOOP` (gt=none) | same block, second ablation value |
| `ROSAME_24` | `BASELINE_REGISTRY` |
| `ROSAME_MILP_24` | `BASELINE_REGISTRY` |
| `ROSAME_MILP_24_TAG` | `BASELINE_REGISTRY` |

## Resources, and where the numbers come from

Measured on the local L≤500 grid (blocksworld, 5 folds, 4 arms):

| L | per fold, all arms |
|---|---|
| 10 | 0.2 min |
| 50 | 1.1 min |
| 100 | 2.1 min |
| 500 | 12.2 min |

The two ROSAME arms scale linearly at ~0.72 s/trajectory (7.4 → 35.9 → 71.0 →
359.7 s at L=10/50/100/500), so L=2000 projects to ~24 min per arm. With a fifth
arm and all five L values, a fold lands near 2.5 h; folds run in parallel, so a
cell is bounded by its slowest fold.

- `--time 1-00:00:00` — 24 h, roughly 10× the projection.
- `--mem 64G` — the five L=2000 fold workers held ~1 GB combined when measured.
  This is deliberate headroom, not a fitted number: the local run died because
  the machine ran out of RAM, and a cell that swaps produces timings that mean
  nothing.
- `--cpus-per-task 12` — SLURM counts logical CPUs and `ThreadsPerCore=2` here,
  so 12 gives 6 physical cores to 5 fold processes plus the parent.
- `--constraint cpu256` — the AMD EPYC 7763 nodes. The unconstrained pool spans a
  1.9× single-thread range, which makes a wall-clock budget non-comparable
  across tasks. Never `cpu128`; that is the slow EPYC.

## Procedure

Corpora are generated **locally**, not here. This sweep only consumes them, and
`run_cell.sbatch` fails fast when a `data_dir` has no `gt_trajectories/`.

```bash
# locally: generate each domain's corpus, then point the config at it
#   (benchmark/run_config_large.yaml -> domains:), and
python scripts/cluster_large/make_manifest.py
git add -A && git commit -m "large sweep manifest" && git push

# on the cluster, from the project root
git pull
conda deactivate                        # the job script activates the env itself
export VIP_ENV=my_env                   # optional; default "vip_venv11"
DRY_RUN=1 scripts/cluster_large/submit.sh    # preview
scripts/cluster_large/submit.sh              # submit
```

Monitor with `squeue --me`; logs land in `logs/vip-large-<arrayid>_<task>.out`.

## Retrying a failed cell

Array tasks are independent and `resume: true` is set in the config, so
re-submitting skips folds that already wrote a `fold_result.json`:

```bash
sbatch --array=7,19,31 scripts/cluster_large/run_cell.sbatch
```
