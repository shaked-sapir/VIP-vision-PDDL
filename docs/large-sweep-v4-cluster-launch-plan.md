# Large-corpora sweep v4 on the BGU cluster: launch plan

Simulation only. The large-corpora configuration (`benchmark/run_config_large.yaml`,
fold-level jobs from `scripts/cluster_large/`), on the extended 6 x 5 grid, with
`nolam` and `offlam` added and both `gt_anchoring` variants of the MILP loop run
live, as they already are in that config.

Status: **plan only.** Nothing here has been changed, submitted or deleted.
Facts marked *measured* were read on 2026-09-20 from the laptop's result trees
or from the cluster over SSH.

---

## 1. State today

**The cluster quota is still exhausted.** *Measured:* `touch ~/.qtest` fails with
`Disk quota exceeded`, for a 0-byte file. The filesystem itself is healthy
(172 TB free, 1 % of inodes used), so the limit is the per-user one. Its numbers
are unknown: `quota -s` prints nothing because the filer does not run `rquotad`.
Until space is freed nothing can run, and `git pull` cannot write either.

**What filled it.** The v3 sweep (`large-corpora__L-sweep__v3`) died at its L=500
tier with 438 folds done. `snapshot_interval: 1` makes the ROSAME arms write one
model file per training step:

| *measured*, one fold instance | files | of which `anytime_snapshots/` |
|---|---|---|
| L = 10 | 579 | ~500 |
| L = 100 | 10,700 | 10,082 |
| L = 500 | 50,000 | 47,475 (`ROSAME_24` alone: 50,001) |

An earlier session counted 6.6 million snapshot files. A 0-byte file failing
points at an **inode (file-count) limit**, not a size limit: the bytes involved
are small (50 MB per L=500 instance).

**What is on the cluster, and whether the laptop has it** (*measured*,
`fold_result.json` counts):

| tree | cluster | laptop | note |
|---|---|---|---|
| `large-corpora__L-sweep` (v1) | 268 | 0 | the uncapped-MILP attempt that hit OUT_OF_MEMORY; superseded |
| `..__v2` | 1,125 | 1,125 | complete locally |
| `..__v3` | 438 | 420 | 18 folds not pulled; snapshot dirs still on the cluster |
| probes, pilots, `simulation-cluster-run*` | present | — | small |

Every one of v1, v2 and v3 predates the current symbolic ROSAME training
dynamics (`docs/rosame-training-convergence-fix.md`: they predate `4997d76fa`
and `c5e43f8c0`), so none is comparable with a new run. v4 is a fresh
`run_name`, not a resume.

**Corpora: no extension needed.** Masking and noise are injected at run time by
`SimulatedDataSource` from each corpus's `gt_trajectories/`, seeded per fold, so
a new grid value needs no new data. *Measured:* all five corpora are on the
cluster with 2,500 GT problems and 2,500 training dirs each
(`blocksworld/large_corpus_L500_n2500`, and `large_corpus_n2500` for hanoi,
npuzzle, depot, gripper).

**Environment drift.** The cluster env has `nolam 1.0.0` and **`offlam 1.0.1`**;
the laptop, which produced every small-data OffLAM row, has `offlam 1.0.0`. The
cluster checkout is on `run-experiments-large-traces` at `abbc5ae9d`, which has
neither arm; they are on `adding-leonardo-algorithms` (pushed).

---

## 2. Decisions that are yours

1. **ROSAME convergence fix first, or not.** It is still "plan only". Launching
   v4 now runs `rosame_24` on a fixed 100 epochs and the two `rosame_milp_24*`
   arms until agreement reaches 1.0. If the fix lands afterwards, the ROSAME
   rows of v4 are re-run (a `backfill_baseline` over frozen observations, not a
   whole new sweep, provided the observations are kept; see 4.3).
2. **What to delete on the cluster** (section 3, step 1). Deletions are yours to
   run or to authorise; I have made none.
3. **Pin `offlam==1.0.0` on the cluster**, so large and small rows come from one
   version. Recommended.
4. **Loss curves.** `snapshot_interval` off loses per-epoch ROSAME loss; the
   agreement-per-round series the convergence panel uses comes from
   `milp_rounds` in `fold_result.json` and is unaffected. If loss curves are
   wanted, `snapshot_interval: 1000` costs about 50 files per arm, not 50,000.

---

## 3. Steps, in order

### Step 1. Free the quota (cluster)

Nothing else works before this. In order of inodes recovered:

```bash
cd ~/projects/VIP-vision-PDDL/benchmark/running_results
# a) v3's snapshots: the 6.6M files. The fold results beside them stay.
find . -path '*large-corpora__L-sweep__v3*' -type d -name anytime_snapshots -prune -exec rm -rf {} +
# b) superseded trees, once you are satisfied with what the laptop holds
rm -rf */large-corpora__L-sweep__mask=*          # v1, never pulled, OOM-broken
rm -rf */large-corpora__L-sweep__v2__mask=*      # 1125/1125 on the laptop
rm -rf */large-corpora__L-sweep__v3__mask=*      # 420/438 on the laptop; pull the 18 first if wanted
```

Unlinking millions of files is slow and is metadata load on a shared login node;
run it under `nice`, or as a job with `--output=/dev/null` (a job cannot write
its log into a full home). Then confirm: `touch ~/.qtest && rm ~/.qtest`.

### Step 2. Get the real quota numbers

Ask HPC support for the block and inode limits of `shaksa` on `/home`. Every
estimate in section 4 is a count of files; without the limit it is a comparison
against "what broke last time", not against a number.

### Step 3. Code and config (laptop, branch `adding-leonardo-algorithms`)

`benchmark/run_config_large.yaml`:

```yaml
run_name: large-corpora__L-sweep__v4
shared:
  algorithms: [pisam_milp_loop, rosame_24, rosame_milp_24, rosame_milp_24_tag, nolam, offlam]
  # snapshot_interval: removed (or 1000 if loss curves are wanted)
  baseline_regime_gate: strict
  nolam_noise: oracle
  nolam_allow_neg_precs: false
  nolam_seed: 0
  # pisam_milp.ablations.gt_anchoring: [init_only, none]   <- already there: both loop arms run live
simulation:
  grid:
    masking_ps: [0.0, 0.01, 0.1, 0.2, 0.3, 0.4]
    noising_ps: [0.0, 0.1, 0.2, 0.3, 0.4]
```

`scripts/cluster_large/sweep_fold.sbatch`:

- `--time 2-00:00:00` (section 5).
- A packing block after `benchmark_runner` returns. The job owns its fold, so
  this is safe, and resume is untouched because `fold_result.json` stays:

```bash
EXP="benchmark/running_results/${DOMAIN}/${RUN_NAME}__mask=${P_MASK}__noise=${P_NOISE}"
for inst in "$EXP"/testing/fold${FOLD}_numtrajs*_gtrate*; do
    [ -f "$inst/fold_result.json" ] || continue            # only finished instances
    rm -rf "$inst/temp_rosame_workspace" "$inst"/*_workspace/traces
    if [ -d "$inst/original_observations" ]; then
        tar czf "$inst/original_observations.tar.gz" -C "$inst" original_observations \
            && rm -rf "$inst/original_observations"
    fi
done
```

  (`$DOMAIN` must be the results-dir name, which for these five domains equals
  the config key; check npuzzle's display name before trusting it.)

Then `python scripts/cluster_large/make_manifest.py --per-fold` (750 rows),
commit, push.

### Step 4. Sync the cluster

```bash
cd ~/projects/VIP-vision-PDDL
git fetch && git checkout adding-leonardo-algorithms && git pull
source activate vip_venv11 && pip install offlam==1.0.0      # decision 3
python -c "import nolam, offlam; print('ok')"
python -m benchmark.benchmark_runner --config benchmark/run_config_large.yaml --dry-run \
    --domains blocksworld --only-mask 0.0 --only-noise 0.4    # expect: [gated out: OffLAM]
conda deactivate
```

### Step 5. Pilot: three fold jobs, not 750

One domain, fold 0, the three corners that exercise what is new:

| cell | why |
|---|---|
| mask 0.0, noise 0.4 | NOLAM at L=2000; slowest evaluation regime |
| mask 0.4, noise 0.0 | OffLAM at L=2000: does it finish inside 3600 s, and how much memory |
| mask 0.4, noise 0.4 | worst case for the loop arms and for wall time |

Read from each: elapsed, `cgroup_peak_mb` (the template logs it; `sacct` memory
is disabled on this cluster), OffLAM's `terminated_by` per L, and the file count
of the packed fold dir (`find <fold dirs> -type f | wc -l`).

### Step 6. Submit

Set `--mem` from the pilot's peak with 2x headroom and `--time` from its
elapsed. Then, staged by domain so the cluster never holds more than one
domain's results:

```bash
python scripts/cluster_large/make_manifest.py --per-fold --domains blocksworld
sbatch --array=0-149%45 scripts/cluster_large/sweep_fold.sbatch scripts/cluster_large/fold_manifest.csv
```

150 fold jobs per domain (30 cells x 5 folds). Failed indices are re-submitted
as they are; `resume: true` skips finished instances.

### Step 7. Pull, verify, purge, next domain

```bash
ssh bgu "cd ~/projects/VIP-vision-PDDL/benchmark/running_results/blocksworld && tar czf - large-corpora__L-sweep__v4__*" \
  | tar xzf - -C benchmark/running_results/blocksworld/
```

Verify by counting `fold_result.json` on both sides (750 per domain) and parsing
each, never by the transfer's exit banner. Only then delete the domain's v4 tree
on the cluster and submit the next domain.

### Step 8. Dashboard

`dashboard_config.yaml`: `simulation.large.prefix.<domain>: large-corpora__L-sweep__v4`,
then `build_dashboard --regen-plots`. `--refresh-stats` reads `.masking_info`
from `original_observations/`, which are tarballs after step 3; unpack them for
the cells whose corruption table is wanted, or leave that table to the
small-data sweep.

---

## 4. Storage

### 4.1 What one fold job writes

A fold job runs one fold of one cell over all five training sizes: 2,660
trajectories in total (10 + 50 + 100 + 500 + 2,000). *Measured* rates from v3:
`original_observations/` is 2 files and about 3.1 KB per trajectory,
`temp_rosame_workspace/` 3 files per trajectory, a loop-arm dir 40 to 120 files
and 0.15 MB, the fold-shared test states 1 file of 1.3 MB. A Lamanna trace is 1
file of about 10 KB per trajectory (3.5x its source), in the gated cells only.

| per fold job | files | bytes |
|---|---|---|
| as the config stands (`snapshot_interval: 1`) | ~125,000 | ~150 MB |
| snapshots off, nothing else | ~14,000 to 19,000 | ~45 to 100 MB |
| snapshots off + step 3's packing (final, at rest) | **~1,000** | **~5 MB** |
| same, transient peak while the job runs | ~17,000 | ~100 MB |

### 4.2 The whole sweep: 150 cells x 5 folds = 750 fold jobs

| | files | bytes | `du` on the cluster (block padding, ~4x) |
|---|---|---|---|
| as the config stands | ~94 million | ~110 GB | ~450 GB |
| snapshots off only | ~11 to 14 million | ~35 to 75 GB | ~150 to 300 GB |
| **snapshots off + packing** | **~0.75 million** | **~4 GB** | **~15 GB** |
| same, one domain at a time (step 6) | ~0.15 million | ~0.8 GB | ~3 GB |
| transient, 45 jobs running | ~0.77 million | ~4.5 GB | — |

For scale: what exhausted the quota was about 6.6 million snapshot files on top
of the older trees. "Snapshots off" alone would still write more files than
that, which is why the packing step is not optional. With packing and domain
staging, the cluster holds about 0.15 M files at rest plus 0.77 M transient,
under a fifth of what it held when it broke.

### 4.3 What packing costs

`original_observations/` become one tarball per fold instance. Anything that
reads them afterwards needs them unpacked first: a `backfill_baseline` /
`backfill_cdps` pass (for example re-running the ROSAME arms after the
convergence fix), and the dashboard's corruption table. They are kept, not
deleted, precisely so those remain possible.

### 4.4 The laptop

*Measured:* 28 GB free. The packed v4 tree is about 4 GB and 0.75 M files, which
fits. The unpacked form (11+ M files) would not, which is what took the laptop to
99 % during the v3 pull.

---

## 5. Memory and time

**Memory.** `sweep_fold.sbatch` requests 24G for one fold worker. The v2
measurement behind it: a five-fold cell peaked at 7.2 to 14.6 GB, so 1.5 to 3 GB
per fold, and that was with the MILP arms uncapped; v4 keeps `mip_traces: 4`.
NOLAM is a counting pass and small. OffLAM holding 2,000 traces is the unmeasured
one, and it runs in a child process inside the job's cgroup, so an OffLAM blow-up
is an OOM kill of the job's largest process, reported as an `error` row, not a
lost fold. Keep 24G for the pilot; set the array's value from the pilot's
`cgroup_peak_mb`. 45 concurrent jobs at 24G is about 1 TB requested, which can
pend on memory while cores idle, so lowering it after the pilot also shortens the
queue.

**Time.** v2 cells took 5 to 10 h at noise up to 0.2, dominated by evaluation,
and two hit the 24 h wall (one `gt=none` evaluation took 8.4 h). v4 adds noise
0.3 and 0.4, where the small-data sweep's cells ran 2 to 4x longer, and OffLAM
can spend its full 3,600 s at each of the larger sizes in the noise-0 cells.
Hence `--time 2-00:00:00`; the partition maximum is 7 days. Order of magnitude
for the whole sweep: 750 jobs at about 10 h is 7,500 job-hours; at a throttle of
45 that is about a week of wall time, per domain about a day and a half.
