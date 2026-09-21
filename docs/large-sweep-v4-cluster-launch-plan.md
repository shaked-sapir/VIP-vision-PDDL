# Large-corpora sweep v4 on the BGU cluster: launch plan

Simulation only. `benchmark/run_config_large.yaml` run as fold-level jobs from
`scripts/cluster_large/`, on the extended 6 x 5 grid, with `nolam` and `offlam`
added, both `gt_anchoring` variants of the MILP loop live, and the ROSAME arms
trained to loss convergence.

Status: **steps 1 and 3 done (2026-09-21); nothing submitted.** v3 was deleted from the
cluster and the quota is unblocked; v1 and v2 remain there. The large config carries the
6 x 5 grid and both new arms, `sweep_fold.sbatch` has a 3-day wall and the packing block
(checked in bash against a fake fold tree), and `fold_manifest.csv` has the 750 rows,
150 per domain in config order. Decided: a higher wall limit, and **no** sampling of the
test problems. Revised
2026-09-21 after the convergence work was merged (`420f7a2f1`, `9040e047c`,
`f630f0938`). *Measured* means read that day from the laptop's result trees or
from the cluster over SSH.

---

## 1. What changed since the first version of this plan

| | before | now |
|---|---|---|
| ROSAME stop rule | agreement = 1.0 (MILP arms), hardwired 100 epochs (`rosame_24`) | training-loss plateau (window 10, patience 3, 0.2 %), `epochs: 2000` as a ceiling, and the 1 h fold budget **enforced** |
| `snapshot_interval` | `1` in the large config: one model file per training step | **removed from the config.** Each arm writes its whole per-epoch loss series to one file, `<fold>/rosame_training/<arm>.json`. It was never 5. Setting it (say 100) is now only for studying how the emitted model evolves |
| large config `run_name` | `..__v3` | already `large-corpora__L-sweep__v4` |
| solving metrics | a model with a no-effect operator scored 0 in every bucket | evaluation plans on a copy without no-effect operators (`planning_copy.py`); rows gain `planning_error_ratio` and `planning_dropped_operators` |
| open decision "convergence fix first?" | open | **settled**: it is in the code v4 will run |

Still to do in the config: the 6 x 5 grid and the two new arms (section 4,
step 3). The convergence constants are **provisional**: the fix's own
calibration probe (its section 6) and pilot gate (section 7) have not been run.

---

## 2. State of the cluster (*measured*)

- **Quota still exhausted.** `touch ~/.qtest` fails with `Disk quota exceeded`
  for a 0-byte file. `/home` itself has 172 TB free and 1 % of inodes used, so
  it is the per-user limit. Its numbers are unknown (`quota -s` prints nothing;
  the filer runs no `rquotad`). A 0-byte failure suggests an inode limit, but a
  hard block limit can refuse file creation too, so **both readings stay open
  until HPC support gives the numbers.** The same deletions relieve either.
- Nothing runs, and `git pull` cannot write, until space is freed.
- Corpora: all five present, 2,500 GT problems and 2,500 training dirs each.
  **No corpus extension is needed for new grid values**: masking and noise are
  injected at run time from `gt_trajectories/`, seeded per fold.
- Env: `nolam 1.0.0`, `offlam 1.0.1`. The laptop, source of every small-data
  OffLAM row, has `offlam 1.0.0`. **The two are the same learner for our usage,
  so no pin is needed** (*measured*, by diffing the two installed copies and
  running both): 1.0.1 adds `return_traces` / `infer_actions` to `learn()`, with
  `greedy = not infer_actions`, which at the default is the `True` that 1.0.0
  hard-coded; it bundles the VAL grounder and fixes its path, which is reached
  only when a trace is missing an action, never in setting 1; it fixes
  `Observation.__str__`, used only when a trace is printed; and it guards a
  `None` constants table. Both versions learned literal-for-literal identical
  models on the frozen traces of five folds of the extended small grid, one per
  domain, at masking 0.1 to 0.4.
- Checkout: branch `run-experiments-large-traces` at `abbc5ae9d`, which has
  neither the new arms nor the convergence code.

---

## 3. How much to free, and what

What is on the cluster (fold results counted over SSH; file counts from the
laptop copy where one exists, estimated otherwise):

| tree | fold results on cluster | files | real bytes | on the laptop |
|---|---|---|---|---|
| v3 `anytime_snapshots/` only | — | **6.63 M** *(measured locally)* | 8.1 GB | yes, all 6.6 M files (they cost the laptop 8 GB too) |
| v3, everything else | 438 | 0.39 M | 1.0 GB | 420 of 438 folds, full artifacts |
| v2 | 1,125 | ~3.2 M *(est.: 14 k per fold x 225 folds)* | ~10 GB | **fold results only** (1,170 files); models, loop logs and observations exist on the cluster only |
| v1 (`large-corpora__L-sweep`) | 268 | ~0.7 M *(est.)* | ~2 GB | none; the OUT_OF_MEMORY attempt |
| probes, pilots, small sweeps, corpora, conda envs | — | ~0.5 to 1 M *(est.)* | ~10 to 15 GB | — |

The filer charges whole blocks per file, so its accounting is roughly 4x the
real bytes for trees of small files: v3's snapshots alone are about 30 GB as
the quota sees them.

Three levels, in order of how much they return:

1. **Minimum: v3's snapshot dirs.** 6.6 M files. This alone puts the account
   back where it was before v3 started, a state in which the whole v2 sweep ran
   to completion. v3's fold results stay.
2. **Recommended: all of v1, v2 and v3.** About 11 M files. None of the three
   is comparable with v4 (v1/v2 ran the MILP arms uncapped; all three predate
   the current ROSAME training dynamics). What is lost that the laptop does not
   have: v2's models and logs, 18 v3 folds, all of v1. The dashboard's large tab
   reads v2's fold results, which the laptop has.
3. Probes and pilots (`epprobe__*`, `probe__*`, `mtprobe__*`, `pilot-cpu256`)
   are small; leave them unless the numbers from support say otherwise.

After level 2 the account holds roughly 0.5 to 1 M files; v4 as designed below
adds at most about 0.9 M at its peak (4.2).

Deleting millions of files is slow metadata work on a shared login node. Run it
under `nice`, or as a job with `--output=/dev/null` (a job cannot write its log
into a full home). The deletions are yours to run or to authorise.

---

## 4. Steps, in order

### Step 1. Free the quota (section 3), then confirm `touch ~/.qtest && rm ~/.qtest`.

### Step 2. Ask HPC support for the block and inode limits of `shaksa` on `/home`.

### Step 3. Config and template (laptop, branch `adding-leonardo-algorithms`)

`benchmark/run_config_large.yaml` (run name, convergence block and snapshot
removal are already in):

```yaml
shared:
  algorithms: [pisam_milp_loop, rosame_24, rosame_milp_24, rosame_milp_24_tag, nolam, offlam]
  baseline_regime_gate: strict
  nolam_noise: oracle
  nolam_allow_neg_precs: false
  nolam_seed: 0
  # pisam_milp.ablations.gt_anchoring: [init_only, none]  <- already there: both loop arms run live
simulation:
  grid:
    masking_ps: [0.0, 0.01, 0.1, 0.2, 0.3, 0.4]
    noising_ps: [0.0, 0.1, 0.2, 0.3, 0.4]
```

`scripts/cluster_large/sweep_fold.sbatch`: raise `--time` (section 5) and add a
packing block after `benchmark_runner` returns. The job owns its fold, so this
is safe, and resume is untouched because `fold_result.json` stays:

```bash
EXP="benchmark/running_results/${DOMAIN}/${RUN_NAME}__mask=${P_MASK}__noise=${P_NOISE}"
for inst in "$EXP"/testing/fold${FOLD}_numtrajs*_gtrate*; do
    [ -f "$inst/fold_result.json" ] || continue            # finished instances only
    rm -rf "$inst/temp_rosame_workspace" "$inst"/*_workspace/traces
    if [ -d "$inst/original_observations" ]; then
        tar czf "$inst/original_observations.tar.gz" -C "$inst" original_observations \
            && rm -rf "$inst/original_observations"
    fi
done
```

Then `python scripts/cluster_large/make_manifest.py --per-fold`, commit, push.

### Step 4. Sync the cluster

```bash
cd ~/projects/VIP-vision-PDDL
git fetch && git checkout adding-leonardo-algorithms && git pull
source activate vip_venv11
python -c "import nolam, offlam; print('ok')"
python -m benchmark.benchmark_runner --config benchmark/run_config_large.yaml --dry-run \
    --domains blocksworld --only-mask 0.0 --only-noise 0.4    # expect: [gated out: OffLAM]
conda deactivate
```

### Step 5. Convergence calibration probe (the fix's section 6)

One job per domain, rule off, `snapshot_interval: 1`, fixed 500 epochs at
L = 10, 100, 2000, one fold. It yields the noise band that sets
`min_improvement` and `window`, and **seconds per epoch at L = 2000**, which
decides whether the rule can fire inside the hour there at all. About 15,000
files in total; delete them once read.

### Step 6. Pilot gate: a handful of fold jobs, not 750

One domain, fold 0, the fix's gate plus the corners that are new in v4:

| cell | what it answers |
|---|---|
| mask 0.0, noise 0.0 | every ROSAME row reports `converged`; the L = 2000 arms finish inside the hour (the fix's gate) |
| mask 0.0, noise 0.4 | NOLAM at L = 2000; evaluation cost at the highest noise |
| mask 0.4, noise 0.0 | OffLAM at L = 2000: finishes inside 3,600 s or times out, and its memory |
| mask 0.4, noise 0.4 | worst case for wall time |

Read from each: elapsed, `cgroup_peak_mb` (the template logs it; `sacct` memory
is disabled on this cluster), every arm's stop reason per L,
`planning_error_ratio`, and the packed fold's file count.

### Step 7. Submit, one domain at a time

Set `--mem` and `--time` from the pilot. 150 fold jobs per domain:

```bash
python scripts/cluster_large/make_manifest.py --per-fold --domains blocksworld
sbatch --array=0-149%45 scripts/cluster_large/sweep_fold.sbatch scripts/cluster_large/fold_manifest.csv
```

Failed or wall-killed indices are re-submitted as they are; `resume: true`
skips finished fold instances, so a killed job loses at most the training size
it was on.

### Step 8. Pull, verify, purge, next domain

```bash
ssh bgu "cd ~/projects/VIP-vision-PDDL/benchmark/running_results/blocksworld && tar czf - large-corpora__L-sweep__v4__*" \
  | tar xzf - -C benchmark/running_results/blocksworld/
```

Verify by counting and parsing `fold_result.json` on both sides (750 per
domain), then delete that domain's v4 tree on the cluster and submit the next.

### Step 9. Dashboard

`simulation.large.prefix.<domain>: large-corpora__L-sweep__v4`, then
`build_dashboard --regen-plots`. `--refresh-stats` reads `.masking_info` from
`original_observations/`, which are tarballs after packing; unpack the cells
whose corruption table is wanted.

---

## 5. Storage for v4

A fold job runs one fold of one cell over all five training sizes, 2,660
trajectories. *Measured* rates (v3): `original_observations/` 2 files and
3.1 KB per trajectory; `temp_rosame_workspace/` 3 files per trajectory, still
written by the current ROSAME runners; a loop-arm dir 40 to 120 files; the
fold-shared test states 1 file of 1.3 MB; a Lamanna trace 1 file of ~10 KB per
trajectory in its gated cells; `rosame_training/` 3 files per instance.

| per fold job | files | real bytes |
|---|---|---|
| current config, no packing | ~14,000 to 19,000 | ~45 to 100 MB |
| **current config + packing, at rest** | **~1,000** | **~5 MB** |
| transient peak while the job runs | ~17,000 | ~100 MB |
| (for reference: v3's config) | ~125,000 | ~150 MB |

| whole sweep, 750 fold jobs | files | real bytes |
|---|---|---|
| current config, no packing | ~11 to 14 M | ~35 to 75 GB |
| **current config + packing** | **~0.75 M** | **~4 GB** |
| same, one domain on the cluster at a time | ~0.15 M at rest + ~0.77 M transient at 45 jobs | ~0.8 GB + ~4.5 GB |

Removing the snapshots was necessary and is done; it is not sufficient. Without
packing, v4 would still write more files than the 6.6 M that broke the quota,
because the observations and ROSAME's workspace copies cost 5 files per
trajectory. With packing and domain staging the peak is about 0.9 M files.

What packing costs: a later `backfill_*` pass, or the dashboard's corruption
table, needs the observations unpacked first. They are kept, not deleted, so
both stay possible.

Laptop (*measured*: 28 GB free, of which v3's local snapshots hold 8 GB that
can go): the packed v4 tree, ~4 GB and 0.75 M files, fits. The unpacked form
would not.

---

## 6. Memory

24G per fold job, as the template has it. Behind it: a five-fold v2 cell peaked
at 7.2 to 14.6 GB with the MILP arms uncapped, so 1.5 to 3 GB per fold, and v4
keeps `mip_traces: 4`. NOLAM is a counting pass. OffLAM holding 2,000 traces is
unmeasured; it runs as a child inside the job's cgroup, so a blow-up kills that
child and becomes an `error` row, not a lost fold. Set the array's `--mem` from
the pilot's `cgroup_peak_mb` with 2x headroom: 45 concurrent jobs at 24G is
about 1 TB requested and can pend on memory while cores idle.

---

## 7. Time: the real risk

Learning is now bounded: every learner is capped at the 3,600 s fold budget, so
one training size costs at most about 6 h (three ROSAME arms, two loop arms,
OffLAM in its cells) and the two large sizes will often sit near that cap. A
fold job's learning is therefore roughly 10 to 14 h in the worst case.

**Evaluation is not bounded, and it dominates.** Each fold scores every model
on **500 test problems** (*measured* from v3's `fold_info.json`; k-fold over
2,500), and a fold job scores 35 models (7 arms x 5 sizes): 17,500 planner calls
at up to 60 s each. In v3 at noise up to 0.2 the planner never timed out
(*measured* timeout ratio 0.000) and cells still took 5 to 10 h, with two
hitting the 24 h wall. At noise 0.3 and 0.4 models are worse; every 10 % of
problems that run to the 60 s cap adds about 50 min per model, up to a day per
fold job. There is no knob today that caps or samples the test problems.

Consequences for the design:

- `--time 3-00:00:00` as the starting value (partition maximum is 7 days),
  corrected from the pilot's worst corner. Resume makes a wall kill cheap.
- If the pilot shows evaluation running to days, the choice is between a lower
  `planning_timeout_seconds` for v4 (it is a fresh run, so nothing on disk
  constrains it) and a sampled test set, which needs a small code change. That
  is a decision for after the pilot, with its numbers in hand.
- Order of magnitude if the worst case does not materialise: 750 jobs at 12 to
  20 h, throttle 45, about 8 to 14 days of wall time; a day and a half to three
  days per domain.

---

## 8. Decisions that are yours

1. What to delete on the cluster (section 3). Recommended: all of v1, v2, v3.
2. Whether to delete v3's 6.6 M snapshot files on the laptop as well (8 GB).
3. After the pilot: `--mem`, `--time`, and whether evaluation needs bounding.
