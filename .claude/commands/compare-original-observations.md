---
name: compare-original-observations
description: >
  Compare a fold's frozen `original_observations/` trajectories against the source GT100
  training trajectories, state by state, reporting the symmetric difference per state.
  Use when: "how corrupted are this fold's observations", "diff the observations against
  ground truth", "which fluents were masked or flipped in fold X", or when investigating
  whether a fold's degraded input explains a bad learned model.
---

Compare a fold's `original_observations/` against the source GT100 training trajectories, state by state.

**Input**: experiment directory and fold name, separated by a space — e.g.  
`benchmark/running_results/blocksworld/SIM__TO=300__... fold0_numtrajs3_gtrate0`

Alternatively, pass the fold directory path directly.

## What it does

For each `original_observation_{problem}.trajectory`:

1. Finds `{problem}_gtrate100.trajectory` (from `run_params.json` simulated GT pool, or training data)
2. Compares **states only** (not operators), **excluding state 0**
3. Reports symmetric difference per state: count + which fluents differ

## Steps

1. Parse `$ARGUMENTS` into experiment dir and fold name (or treat as fold path if it contains `original_observations/`).

2. Run from project root with venv activated:

```
source venv11/bin/activate && python -m benchmark.evaluation.compare_original_observations <experiment_dir> --fold <fold_name>
```

3. Present the printed table and any warnings (missing GT100, state-count mismatch).

4. If no `original_observations/` exist in the fold, say so explicitly.
