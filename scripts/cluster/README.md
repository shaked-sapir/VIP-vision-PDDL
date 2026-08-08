# Running the simulated benchmark on the BGU SLURM cluster

Everything here turns `benchmark/run_config.yaml` (the single source of truth)
into SLURM array jobs. Outputs land exactly where `benchmark_runner` would put
them — `benchmark/running_results/{domain}/{run_name}__mask={p}__noise={q}/` —
so `collect_results`, `experiment_report`, and the dashboards work unchanged.

## Files

| File | Role |
|---|---|
| `make_manifest.py` | Reads `run_config.yaml`, writes `manifest.csv` (+ `run_flags.sh` with the shared CLI flags). `--per-fold` writes `manifest_folds.csv` instead. |
| `run_cell.sbatch` | Array template, **cell level**: one task = one `(domain, p_mask, p_noise, num_trajs)`; its 5 folds run in parallel in-process (6 CPUs). |
| `run_fold.sbatch` | Array template, **fold level**: one task = a single fold instance (2 CPUs). Uses `experiment_runner --folds`. |
| `submit.sh` | Computes the `--array` range from the manifest and submits. `DRY_RUN=1` to preview. |

## Typical workflow

```bash
# on your laptop: edit benchmark/run_config.yaml, then
python scripts/cluster/make_manifest.py     # or --per-fold, or --domains blocksworld
git add -A && git commit -m "sweep manifest" && git push

# on the cluster (login node, from the project root)
git pull
conda deactivate                            # job scripts activate the env themselves
export VIP_ENV=my_env                       # optional override (default: "vip_venv11")
scripts/cluster/submit.sh                   # or: submit.sh scripts/cluster/manifest_folds.csv 30
```

Monitor with `squeue --me`; logs are in `logs/vip-cell-<arrayid>_<task>.out`.
After the array finishes, aggregate as usual (`collect_results` /
`benchmark.evaluation.experiment_report`), optionally as a dependent job:

```bash
sbatch --dependency=afterany:<array_job_id> --wrap "python -m benchmark.evaluation.experiment_report <experiment_dir>"
```

## Retrying failures

Every task runs with `--resume`, and each fold instance's `fold_result.json`
is its completion marker — so re-submitting is idempotent and only recomputes
what's missing:

```bash
sacct -j <array_job_id> --format=JobID,JobName,State,Elapsed,MaxRSS | grep -Ev 'COMPLETED|batch|extern'
sbatch --array=7,23,51 scripts/cluster/run_cell.sbatch scripts/cluster/manifest.csv
```

## Cell-level vs fold-level

Start with **cell-level** (`manifest.csv`): ~fewer, mid-sized tasks; the
in-process fold pool does what it was designed for. Switch to **fold-level**
(`--per-fold`) only if individual folds keep timing out unevenly and you want
finer-grained retries. Don't mix both granularities for the same run_name at
the same time — they'd write to the same experiment directories concurrently.

## Notes & gotchas

- **Submit from the project root** — the templates `cd "$SLURM_SUBMIT_DIR"`.
- **Regenerate the manifest after every `run_config.yaml` change** —
  `run_flags.sh` snapshots the shared params; a stale one will make `--resume`
  strict-abort on a `run_params.json` conflict (which is the safety net doing
  its job).
- Simulation is CPU-only: templates request `--gpus=0`, so you never wait in
  the GPU queue.
- SLURM `--time` in the templates is set above the runner's own internal
  timeouts (`learning_timeout x 2` per fold), so partial-result handling in
  Python fires before SLURM's hard kill. If you raise the timeouts in the run
  config, raise `--time` accordingly.
- `run_params.json` is written per experiment dir by whichever task gets there
  first; concurrent tasks of the same cell write identical content, and with
  `--resume` it's skip-if-exists — benign.
- Keep `events_tracing: false` for sweeps (the trace HTML is heavy).
