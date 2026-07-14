# Plan: Produce GT + Noisy Trajectories Correctly in One `data_generator` Run

## Purpose

Make a single `data_generator` run emit, per problem, **two independent
trajectory artifacts**:

- a **noisy (observed)** trajectory — the classifier's reading of the rendered
  frames, and
- a **clean (ground-truth)** trajectory — the true states,

where the clean one is produced **deterministically from an authoritative
source and never derived from the classifier**. This works uniformly for
PDDLGym (predefined + generate) and external (depot/gripper) domains and removes
the need for the separate `generate_gt_trajectories` post-step.

---

## Root cause being fixed

Today GT and noisy share one `{problem}_trajectory.json`.
`_run_generation_inference` (data_generator.py:92) runs classification, then
`_translate_problem_and_rebuild_json` (data_generator.py:113–125) `unlink()`s
the true GT JSON and rebuilds it **from the classifier `.trajectory`**.
`generate_gt_trajectories` (data_generator.py:337, 508) later transcribes the
surviving — now classifier-derived — JSON into `gt_trajectories/`. Net effect:
the real states captured from the simulator are destroyed, and `gt_trajectories/`
holds classifier output relabelled "GT".

Empirically this corrupts hanoi badly (168 conflicts/cell, timeouts at
mask=0/noise=0) and npuzzle mildly (~3 conflicts/cell); blocks/depot/gripper
survive only because their classifier happened to be accurate.

---

## Decisions (locked)

1. **GT source: capture-at-source only** — no symbolic replay through the
   reference model.
2. **Layout: unchanged for now** — noisy stays at `training/trajectories/`, GT at
   `gt_trajectories/`. (Rename of noisy → `training/noisy_trajectories/` is a
   deferred follow-up; see "Deferred".)

---

## Target artifact layout

Per problem, two parallel trees. Same `objects` and same `ground_action`
sequence in both; **only the state literals differ** (classifier vs true).

```
<experiment>/
  training/trajectories/<problem>/          # NOISY artifact
      <problem>.trajectory                  # classifier states + true actions
      <problem>_trajectory.json             # classifier states + true actions (JSON)
      <problem>.masking_info
      state_000000.png ...                  # rendered frames
  gt_trajectories/<problem>/                # CLEAN artifact
      <problem>.trajectory                  # true states + true actions
      <problem>_trajectory.json             # true states + true actions (JSON)
```

### Which `_trajectory.json` is which (direct answer)

| File | Artifact | `current_state` / `next_state` literals | `ground_action` | Role |
|---|---|---|---|---|
| `training/trajectories/<p>/<p>_trajectory.json` | **noisy** | classifier's predicted states | true action sequence | the observation the learner sees / degrades (masking + noising applied to this) |
| `gt_trajectories/<p>/<p>_trajectory.json` | **clean** | true captured states (gym `env.step` / external planner) | true action sequence | ground-truth reference for GT-rate injection and evaluation |

The two JSONs are identical in structure, object set, and action labels. They
diverge only in the state `literals`. Actions are treated as known throughout,
so they are true in **both** files; only *states* are noisy in the noisy file.

---

## GT capture — one generator, per-mode source

New shared module `benchmark/experiment_running_helpers/gt_builder.py`:

- `capture_gt(mode, handler, problem_dir, source_dir=None) -> List[step]`
- `write_gt(steps, gt_dir)` — reuses `build_trajectory_file` /
  `convert_trajectory_to_json` to emit both `.trajectory` and `_trajectory.json`.
- `validate_gt(steps)` — lightweight, model-free checks (see "Verification").

Authoritative source per mode:

- **PDDLGym (generate + predefined):** the `GT_trajectory` that
  `_execute_trajectory` (pddlgym_trajectory_handler.py:174–189) already builds
  from `env.step` literals. Gym is the deterministic ground-truth simulator.
  Persist these states to `gt_trajectories/` **before** classification runs.
- **External (depot/gripper):** the source planner trajectory in
  `source_problem_dir`. Copy it to `gt_trajectories/` **before**
  `_cleanup_external_source_files` (data_generator.py:478) touches the dir.

Both GT state sets are passed through the same `translate_problem_pddl` used for
the noisy path so object and predicate schemas match — translation is applied to
the GT **states directly**, never by round-tripping a `.trajectory`.

---

## Changes in `data_generator.py`

1. **Generate path** (`_run_generation_inference`, :92): capture + write GT to
   `gt_trajectories/` first, then classify → write noisy to
   `training/trajectories/`. **Remove** the `json_file.unlink()` + rebuild in
   `_translate_problem_and_rebuild_json` (:123–125) — schema-translate the GT
   states in place instead of reconstructing the JSON from the classifier
   `.trajectory`.
2. **Predefined path** (`generate_trajectories`, PDDLGym branch, ~:480–496):
   same ordering — persist gym GT before inference.
3. **External branch** (:467–478): copy source planner GT to `gt_trajectories/`
   before `_cleanup_external_source_files`.
4. **Remove** the trailing `generate_gt_trajectories(...)` calls (:337, :508).

---

## GT-rate injection & evaluation wiring

`inject_gt_states_by_percentage` (pddl_trajectory.py:396) reads GT states from
the `json_trajectory_path` it is given (:410–411). It must be pointed at the
**`gt_trajectories/<p>/<p>_trajectory.json`** (clean), not the noisy JSON in
`training/trajectories/`. Audit its call sites and pass the clean path.

`select_simulated_gt_trajectories` / `build_gt_trajectory_lookup`
(simulated_data_utils.py:53–77) already resolve from `gt_trajectories/`, so the
simulated path needs no code change — only correct contents.

---

## `generate_gt_trajectories.py` → backfill tool

Repurpose as a thin CLI over `gt_builder`, no longer part of the normal run.

**Backfill consequence of capture-at-source:** for already-corrupt datasets
(hanoi, npuzzle) the true states were deleted, so backfill cannot transcribe
them. It must **re-derive by re-stepping**: re-instantiate the gym env per
problem, replay the preserved `ground_action` sequence from the problem `.pddl`
init, and capture `env.step` states — no re-imaging, no re-classification.
External backfill pulls GT from the original source dataset dir if still present.

---

## Determinism

Gym problem/plan generation is seeded (`--seed`, already present); record the
seed in the run manifest so GT capture and any re-step backfill reproduce
identical states. The GT path has no LLM ⇒ fully deterministic. The noisy path
remains classifier-driven.

---

## Verification step (built into the run)

Model-free guard (no symbolic replay, per decision):

- every GT state is fully specified and every step carries a `ground_action`;
- no fluent conflicts across identical lifted transitions (a clean, fully
  observable GT must be conflict-free);
- optional per-domain invariant hook (e.g. hanoi: one support per disc, legal
  stacking, clear ⇔ uncovered).

Emit a per-run pass/fail summary; fail a problem loudly rather than shipping
silent corruption. Regression check: after the fix, mask=0/noise=0 learning on
hanoi and npuzzle must yield 0 conflicts, 1 CFM, solving_ratio 1.0;
blocks/depot/gripper must stay clean.

---

## Task breakdown (ordered)

1. **`gt_builder` module** — `capture_gt` / `write_gt` / `validate_gt`; unit-test
   `write_gt` round-trips `.trajectory` ↔ `_trajectory.json`.
2. **PDDLGym GT capture** — expose `_execute_trajectory`'s `GT_trajectory` and
   write it to `gt_trajectories/` before inference in both generate and
   predefined paths.
3. **External GT capture** — copy source planner trajectory to `gt_trajectories/`
   before cleanup.
4. **Stop the overwrite** — delete `unlink()`+rebuild in
   `_translate_problem_and_rebuild_json`; translate GT states directly.
5. **Drop the post-step** — remove `generate_gt_trajectories(...)` calls; keep
   the script as a backfill CLI.
6. **Wire GT-rate injection** — point `inject_gt_states_by_percentage` at the
   clean `gt_trajectories/` JSON; audit call sites.
7. **Verification** — implement `validate_gt` + per-run summary.
8. **Backfill** — re-step gym on hanoi + npuzzle to regenerate correct GT.
9. **Regression** — rerun mask=0/noise=0 across all five domains; confirm 0
   conflicts / solving 1.0 everywhere.

---

## Deferred

Rename noisy `training/trajectories/` → `training/noisy_trajectories/` for
explicit semantics. Reader-audit checklist for when we do it:
`experiment_runner.py:68`, `normalize.py:187`,
`compare_original_observations.py:91` (leave the hardcoded `__main__` demo paths
in `llm_*_fluent_classifier.py` alone).

---

## Risks

- Keeping the shared `trajectories` name means the noisy artifact still owns a
  `_trajectory.json`; ensure the GT writer targets `gt_trajectories/`
  exclusively and nothing in the run rebuilds
  `training/trajectories/.../_trajectory.json` from GT — the two trees live in
  parallel.
- Any remaining consumer that reads `training/trajectories/.../_trajectory.json`
  expecting GT (rather than the noisy observation) must be repointed at
  `gt_trajectories/` — the GT-injection call site is the known one; grep for
  others.
- Schema translation of GT states must handle static predicates and typing;
  cover in tests.
