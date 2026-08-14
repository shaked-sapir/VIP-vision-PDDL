# Deferred Refactor TODOs

Future work we **may** want to do later — none of this is committed or urgent.
These are the pieces of the 2026-07-17 code review (`docs/code-review-17-07-2026.md`)
that were intentionally left for later after the safe/high-value parts were done.

Status of the review's P1/P2 structural items:
- **2.6** (CDPSConfig / kill the NOISY_PISAM config-bag) — **done** (Stage 1).
- **2.7a** (delete commented `_dedup_patches`, fix `run()` return-type lie, add `SearchResult` dataclass) — **done**.
- **2.8a** (make `upenv_compat` cwd-independent via `tempfile`, drop `predictive_metrics`' `os.chdir`) — **done** (safe subset).
- **2.9** (kill the save→read round-trip, de-inject `evaluate_model`/`save_learning_metrics`, fix the `terminated_by`/`conflict_free_model_count` None bug) — **done**.
- Profiler/`timing_report.json` full removal — **done**.

---

## Future TODO — 2.7b: break up `conflict_search.py` further

`src/plan_denoising/conflict_search.py` is still a ~1,380-line module (it was
~1,090 when this was written; it has grown, not shrunk). The
`SearchResult` dataclass (2.7a) covered the return value, but two more extractions
were deferred because they are real internal surgery on `self.`-coupled methods:

- **`SearchResultWriter`** — pull the persistence/serialization methods out of the
  search class: `_serialize_model_constraints`, `_serialize_fluent_patches`,
  `_save_patch_details`, `_save_conflict_free_model`, `_save_final_model`,
  `_write_node_expansion_times`, `_write_conflict_free_solutions_log`.
- **`ConflictGrouper`** — pull the grouping/scoring heuristics out:
  `_conflict_priority`, `_group_key`, `_group_conflicts`, `_choose_conflict_group`,
  `_group_representative`, and the five `_group_score_*` methods.

Do this on its own branch and lean on the tests (`test_conflict_search.py`,
`test_node_choosing_strategy.py`, `test_model_patch_toggle.py`), since they exercise
these paths.

## Future TODO — 2.8b: split `run_single_fold`

`benchmark/experiment_running_helpers/run_fold.py::run_single_fold` is a ~300-line
function inside one `try/finally`. Split into `_prepare_fold_data` / `_run_cdps` /
`_finalize_fold` (or similar). The care point is the large set of locals threaded
between phases — move them behind a small context object or explicit returns and
verify no data-flow drift. Pure readability refactor, higher churn than risk.

## Future TODO — 2.8 (upstream): fully remove `run_fold`'s `os.chdir`

`run_single_fold` still does `os.chdir(fold_work_dir)`. After 2.8a this is **only**
needed because AMLGym's `problem_solving` writes a plan file named `./tmp` to the
current working directory (`amlgym/metrics/_solving.py:85,92`), so the per-fold chdir
keeps parallel fold processes from racing on it. To drop the chdir entirely we'd need
AMLGym's `_solving.py` to write that plan to a `tempfile` instead of `./tmp` — an
**upstream change in the AMLGym package**, not this repo. Until then the chdir stays
(now documented with a comment at the call site).
