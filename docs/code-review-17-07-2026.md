# Code Review — VIP-vision-PDDL

**Date:** 2026-07-17
**Scope:** (1) revision of the "repo consultant — experiment planning" session's work — commit `28b749ecc` ("started refactoring of experiment running towards standalone algorithms") plus the uncommitted working-tree changes on `ablation-study-vs-rosame`; (2) a full audit of the codebase for refactoring opportunities, DRY violations, separation-of-concerns problems, and bad practices. All findings were verified against the actual code (file:line references throughout).

---

## Part 1 — Review of the other session's work

### What it did

The commit + working tree together implement the "standalone algorithms" refactor:

- New `benchmark/algorithms.py`: one flat namespace (`cdps` + `BASELINE_REGISTRY` keys), `--algorithms cdps rosame` replaces `--baselines`, any subset runnable standalone.
- Killed the two-phase (unclean/cleaned) flow: `run_single_fold` now runs baselines once on the degraded data plus one CDPS pass; per-phase result rows are gone.
- Killed the `mode` (masked/fullyobs) axis: SAM/PISAM/NOISY_SAM/ROSAME/PO_ROSAME adapters deleted, one ternary ROSAME runner that degenerates to binary when nothing is masked, `supports_mode()` removed from `BaselineRunner`.
- New result pipeline: `result_schema.py` (base fields + nested `algorithm_specific`) + `collect_results.py` reading per-cell `fold_result.json` markers; results CSVs and the 854-line `reporting.py` deleted.
- `SimulatedDataSource.prepare()` now persists degraded observations to disk so CDPS and baselines consume byte-identical data.
- Deleted `analysis_experiment_compare/` (~11 modules + ~12k lines of committed CSV outputs) and stale helpers; updated CLAUDE.md.

### Assessment

**The direction is right and most of it is well executed.** Killing the phase/mode matrix removes a real combinatorial mess (the old `learn_sam_pisam` had 4 code paths; now 1). The `result_schema` + `fold_result.json` single-source-of-truth design is cleaner than the old triple CSV + Excel write path, and nesting `algorithm_specific` keeps the base schema honest. Deleting `analysis_experiment_compare/` with its committed CSVs was overdue. The docstrings on the new modules are unusually good.

**But the refactor is ~85% finished, and the remaining 15% is exactly the kind of residue that rots.** Concrete issues, ranked:

1. **It introduced a double-write of `original_observations/`** (new bug). `SimulatedDataSource.prepare()` writes `output_dir/original_observations/original_observation_{problem}.{trajectory,masking_info}` (`data_source.py:215-218`), and then `_learn_pisam_with_profiling` unconditionally re-saves the same observations to the same dir with the same prefix (`learning_helpers.py:144-153` — the guard `if fold_work_dir is not None and prepared_trajectories and masked_observations:` does not check `pre_built_observations is None`). In simulated runs every file is written twice (verified: both derive identical stems). Fix: make persisting the data source's job only; gate the `learning_helpers` save on `pre_built_observations is None`.

2. **Dead assignment + lying comment in `run_fold.py:353-355`.** `denoiser_algo_name = 'NOISY_PISAM'` ("used for the metrics file lookup only") is immediately overwritten by `learn_sam_pisam`'s return value, which is now hardcoded `'PISAM'`. It only works because `load_learning_metrics` ignores both its `phase` and `algorithm_name` parameters entirely (`statistics.py:112-126`). Fix: stop returning `algo_name` from `learn_sam_pisam`, reduce `load_learning_metrics` to `(fold_work_dir)`.

3. **Stale phase vocabulary survives the phase removal.** `total_transitions_unclean`, `"[STATS] Unclean phase"` (`run_fold.py:307-312`); the persisted metadata key `cleaned_equals_unclean_pisam` (`trajectory_utils.py:242-264`, `run_fold.py:424`) for what is now "patched equals input"; profiler categories `sam_pisam_trajectory_processing_cleaned` / `learning_process_*_cleaned` (`learning_helpers.py:127,137,217`); module name `cleaned_trajectories.py` whose docstring still cites the deleted `NOISY_SAM`. One rename pass finishes the job.

4. **`check_trajectories_equal` kept its dead two-phase branch.** The `is_patched_observations=False` path (`run_fold.py:33-76`) is unreachable — the only call site passes `True`. Delete the flag and branch.

5. **`_build_main_kwargs` silently drops unknown `shared:` keys** (`benchmark_runner.py:275-303`). Every archived run config in `finished_run_configs/` still has `mode: masked`; it's now discarded with no warning — as would be a typo like `n_fold`. The module docstring example (`benchmark_runner.py:33`) also still shows `mode: masked`. Fix: warn/raise on unknown keys, update the docs.

6. **The refactor's helper untracked file** `benchmark/diagnosis/check_rosame_encoding_parity_DELETE_LATER.py` should be deleted once parity is confirmed (it is referenced by nothing), and `test_refactoring.py` at the root — which imports the long-gone `benchmark.experiment_helpers` and `setup_algorithm_workspace` (deleted in this very working tree) — is now doubly dead.

7. **Small design nit:** baselines receive `timeout_seconds=conflict_search_timeout or 60` (`run_fold.py:114`) — the *conflict-search* timeout is not a meaningful budget for ROSAME, and `60` is a magic fallback. Give baselines their own config knob or pass `learning_timeout_seconds`.

One thing the session did that deserves explicit praise: resume safety. Old experiment dirs have `run_params.json` with `mode`/`baselines` keys, so `run_params_conflicts()` will strict-abort a resume into an incompatible directory rather than mixing schemas. That was thought through.

---

## Part 2 — Codebase audit

### P0 — Fix before trusting anything (correctness / security)

**2.1 Live API keys are hardcoded in five files.** The same real OpenAI key (`sk-proj-CgoDxLAn…`) and Gemini key (`AIzaSyANQZ…`) appear in the `__main__` blocks of `llm_maze_fluent_classifier.py:127,139`, `llm_blocks_fluent_classifier.py:128,140`, `llm_hiking_fluent_classifier.py:101,113`, `llm_hanoi_fluent_classifier.py:150,162`, `llm_npuzzle_fluent_classifier.py:146,158`. They are in git history. **Revoke both keys today**, load from `config.yaml`/env, and delete the copy-pasted demo blocks (which also carry copy-paste artifacts — variables named `hanoi_openai` in the blocks/maze/hiking files).

**2.2 `SimulatedDataSource.prepare()` silently ignores `gt_rate`** (`data_source.py:182-232`, verified). GT indices are hardcoded to `{0}` (lines 217, 227). A simulated run configured with `gt_rates: [25]` would produce 0%-GT behavior while every row, directory name, and stat says 25%. Fix: raise on `gt_rate > 0` until implemented, or implement injection.

**2.3 GT-transition counting is a drifting reimplementation.** `statistics.py:59-77` *recomputes* GT indices from `gt_rate` ("following the same logic as `inject_gt_states_by_percentage`") instead of using the actual per-trajectory indices already carried in `prepared_trajectories[i][3]`. Any drift between the two implementations corrupts `total_gt_transitions`. Fix: `sum(len(t[3]) for t in prepared_trajectories)`.

**2.4 Batch timeout escapes the `try` and aborts `main()`** (`experiment_runner.py:380`, verified). `for future in as_completed(futures, timeout=batch_timeout)` raises `TimeoutError` *at the for-statement*, outside the `try:` on the next line — one slow batch discards all collected results. Also `completed_count` increments before `future.result()`, so the error message blames the wrong job. Wrap the loop itself.

**2.5 Silent `except Exception: pass/continue` in the results path.** `collect_results.py:31-33,55-57` makes a corrupt `run_params.json` or `fold_result.json` silently vanish from the results table — an experiment under-reports with no trace. Same pattern in `post_process_gt_metrics.py:36-37,59-60` plus four silent early returns. Your own CLAUDE.md forbids this ("No silent except blocks"). Log the path at minimum.

### P1 — Highest-leverage refactorings

**2.6 One `ConflictSearchConfig` dataclass would kill the worst plumbing in the repo.** The same 9 conflict-search parameters (`fluent_patch_cost, fluent_patch_weight, model_patch_cost, model_constraint_weight, max_search_nodes, search_mode, node_choosing_strategy, conflict_group_strategy, fluent_branch_mode`) are hand-threaded through **seven** layers, key by key: `run_config.yaml` → `_PASSTHROUGH_KEYS` (`benchmark_runner.py:264-272`) → `experiment_runner.main` (**25 params**, re-listed in `run_params`, the banner, and `fold_kwargs`) → `run_single_fold` (**27 params**) → `learn_sam_pisam` (18 params) → attribute-assignment onto `NOISY_PISAM` (`learning_helpers.py:267-277`) → read back off the learner into `ConflictDrivenPatchSearch(...)` (`learning_helpers.py:161-180`) — with the CLI mirroring it an eighth time. Adding one search parameter today touches ~8 sites.

Related: **`NOISY_PISAM` is a dead adapter being used as a config bag.** Its `learn()` method is never called (`NOISY_PISAM.py:49-111`); the only instantiation assigns 10 attributes and reads them back. Its docstring is copy-pasted from the deleted SAM adapter (cites the SAM paper). Worse, its `@dataclass` fields `negative_precondition_policy`, `max_search_nodes`, `timeout_seconds`, `seed` lack type annotations — so they are *class attributes, not fields*. Replace the adapter with a frozen `CDPSConfig` dataclass in `src/pi_sam/plan_denoising/` (string→enum coercion in `__post_init__`, a `to_dict()` for run_params/reports) and pass it end-to-end.

**2.7 `conflict_search.py` is a 1,144-line god module.** The class mixes the search loop (`run()` alone is ~350 lines, `conflict_search.py:403-753`), cost computation, JSON serialization, model/report persistence, timing percentile math, and five group-scoring heuristics. Plus: `run()` returns a **7-tuple** that callers unpack as `learned_model, _, _, _, _, report, patched_observations` (`learning_helpers.py:191`), and its return annotation/docstring lie (element 5 documented as `patch_count: int`, actually `final_cost: float` — `conflict_search.py:424,749`). And there are 74 lines of commented-out alternative `_dedup_patches` code (`:946-1001`, "Uncomment to revert" — git history is the archive). Extract a `SearchResult` dataclass, a `SearchResultWriter`, and a `ConflictGrouper`; delete the commented block.

**2.8 `run_single_fold` is a 340-line god function with 27 parameters** (`run_fold.py:138-478`): CV split, data prep, test-state generation, metadata, stats, baseline loop, CDPS, metrics save/load, patched-obs saving, multi-solution eval, correlation, timing, resume marker. It also does `os.chdir(fold_work_dir)` **inside a process-pool worker** (`run_fold.py:218`) — process-global state mutation that exists only because `upenv_compat.py:44-51` writes temp PDDL files to the cwd (and `predictive_metrics.py:81-94` adds its own `os.chdir` workaround for the same reason). Split into `_prepare_fold_data` / `_run_cdps` / `_finalize_fold`, and give `upenv_compat` a `tempfile.TemporaryDirectory` so both `chdir`s die.

**2.9 Write-then-immediately-read round-trip:** `run_fold.py:374-376` calls `save_learning_metrics_func(...)` (which writes `learning_metrics.json` *and returns the dict*, `evaluation.py:59`) and then `load_learning_metrics(...)` to re-read the same data from disk, after which it passes through three renamings. Use the return value. Also `evaluate_model_func` / `save_learning_metrics_func` are injected as parameters but only ever bound to the same two module functions — pointless indirection; import them.

### P2 — DRY violations

**2.10 The 18 AMLGym metric names live in three places** that can silently drift: the `null_metrics` literal (`run_fold.py:225-231`), `AMLGYM_METRIC_FIELDS` (`result_schema.py:23-28`), and `evaluate_model`'s return keys (`evaluation.py:145-168`). Make `null_metrics = {k: None for k in AMLGYM_METRIC_FIELDS}` and assert `evaluate_model`'s keys against the schema.

**2.11 `run_params.json` loading is re-implemented five times** with divergent fallback/error behavior: `collect_results.py:25-33`, `compare_original_observations.py:45-53`, `experiment_report.py:656-662`, `cfm_quality_table.py:85-99`, `cfm_quality_analysis.py:69-79`. One `load_run_params(experiment_dir)` next to the schema.

**2.12 Fold-directory naming knowledge is scattered.** The regex `fold(\d+)_numtrajs(\d+)_gtrate(\d+)` appears verbatim in `experiment_report.py:39` and `cfm_quality_table.py:73`; `startswith("fold")` iterdir filters appear 5+ times; `collect_results.py:50` uses a different glob for the same thing; `build_dashboard.py:67-80` and `combine_dashboard_reports.py:35-46` are copy-pasted cell parsers. `resume.py` already owns the naming convention (`fold_instance_dir`) — add `iter_fold_dirs()` / `parse_fold_dir_name()` there.

**2.13 Copy-paste with a written confession:** `simulated_data_utils.py:27-46` is marked "mirrored from run_simulated_experiment.py" and duplicates `simulated_version/run_simulated_experiment.py:50-69` exactly. Import instead of mirror. Similarly, the trajectory-exclusion filter (`'truncated' not in stem…`) is duplicated within `trajectory_utils.py` itself (74-76 vs 163-165).

**2.14 The five LLM fluent classifiers are heavy copy-paste.** Identical `__init__` scaffolds (including `use_uncertain`, which the base class doesn't own — maze assigns it *twice*, lines 25 and 33), identical `set_type_to_objects`, identical `__main__` blocks, and copy-pasted docstrings that lie ("N-puzzle domain" on the *hiking* classifier:13; "Hanoi domain" on maze:53). Lift `use_uncertain` + `set_type_to_objects` into `LLMFluentClassifier`; delete the `__main__` blocks (see 2.1).

**2.15 Duplicate LLM helpers:** `src/llms/facts_extraction.py:27,65,71` reimplements three `LLMFluentClassifier` methods; `src/llms/utils.py:30 encode_image` duplicates `src/utils/visualize.py:65 encode_image_to_base64`. Port the two remaining callers and delete.

**2.16 `diagnosis/visualize_trace.py:1141-1231` hand-rolls a regex parser** for the `.trajectory`/`.masking_info` on-disk format that `src.utils.masking.load_masked_observation` already parses (and `retrace_search.py:47-80` uses correctly on the same directories). Two divergent parsers for one format is a bug factory.

### P2 — Separation of concerns / conventions

**2.17 `print()` is the only logging mechanism** — hundreds of occurrences across the pipeline; `correlation_analysis.py` defines a `logger` and then prints anyway (:157,186,203,282). With process-pool folds the output interleaves unusably. Adopt `logging` with per-fold loggers; `conflict_search.py` shows the pattern — though its `DefaultSearchLogger` (`conflict_search.py:123-130`) attaches a new StreamHandler per instance (constructed per fold → duplicated streams); use `logging.getLogger(__name__)`.

**2.18 Broad excepts that convert bugs into null result rows:** `run_fold.py:443-459` (whole CDPS phase → null row), `baselines/rosame_runner.py:128-130` (any ROSAME encoding bug → silent null row). At minimum `logger.exception` so real failures are distinguishable from expected ones.

**2.19 `sys.path` hacks and import-time side effects:** module-level `sys.path.insert` in `benchmark_runner.py:64`, `data_generator.py:35`, `generate_gt_trajectories.py:31`, diagnosis scripts; `po_rosame_runner.py:12-13` mutates `sys.path` inside an import-except; `test_states_generator.py:43` mutates a `unified_planning` global at import; `UPState.MAX_ANCESTORS = None` set library-wide in two places. Standardize on `python -m benchmark....` (already the documented style) and scope UP config.

**2.20 Domain logic in the generic base:** `pddlgym_trajectory_handler.py:83-84` hardcodes a maze-specific render branch (`if self.domain_name == "PDDLEnvMaze-v0"`) in the shared base class, violating your own layering rule. Add a `_render()` hook.

**2.21 Magic strings where the project mandates enums:** `state_type="prev"/"next"`, `branch_type="fluent_fix"/"model_fix"/…`, `terminated_by="timeout_exceeded"/…` (`conflict_search.py:442-570,607-649,913-933`; `noisy_learner_mixin.py:167-270`). Three small enums.

**2.22 `NoisyLearnerMixin.handle_effects` is a 120-line function** of four near-identical match-and-emit-conflict loops separated by section comments (`noisy_learner_mixin.py:358-477`). Extract one `_emit_conflicts(...)` helper, call it four times.

**2.23 Dead code chains in `src/`** (grep-verified): `src/pi_sam/pisam_experiment_runner.py`, `src/vip_experiments/BasicSamExperimentRunnner.py` (typo included) → `src/domains/hanoi/algorithm.py` → the deterministic hanoi/blocks handlers reachable only through it; both experiment-runner modules import a top-level `experiments` package that is *gitignored* (`.gitignore:15-16`), so they cannot even import on a clean clone. `LLMFluentClassifier.create_examples_for_few_shot` (`llm_fluent_classifier.py:67-98`) is unused *and* would raise `TypeError` if called (passes a list where an int index is expected). Delete.

### P3 — Repo hygiene

**2.24 Root directory.** 12 root `test_*.py` files, none of which are pytest-safe: `test_refactoring.py` (imports deleted modules, `sys.exit` at module level), `test_gt_injection.py` / `test_problem8_frame_axioms.py` (hardcoded `/Users/shakedsapir/…` paths, logic at import time, `signal.alarm(30)`), LLM scripts that would make live API calls under pytest collection, and four generators for abandoned domains (doors, rearrangement, tsp, slidetile — none in `_DOMAIN_REGISTRY`). Plus one-off outputs (`temp_problem3_generation/`, `slidetile_test_sequence/`, `tsp_test_sequence/`, `artifact_trajectory_confusion.html`), many stale xlsx/png artifacts, and stale summary .md files (`old_code_analysis.md` analyzes a file that no longer exists; `QUICKSTART_CROSS_VALIDATION.md` is actually a pasted LLM answer about frame axioms, not a quickstart; `refactoring_cleanup_summary.md`/`uniform_interface_fix_summary.md` describe deleted modules). Target: root keeps `README.md`, `CLAUDE.md`, `config.example.yaml`, `requirements.txt`; add `docs/` and `scripts/`; delete the dead scripts and outputs. `README.md` itself is a 6-line stub ending "(( I need to write down this explanation better ))".

**2.25 Tests.** No `pytest.ini`/`conftest.py`/CI; the only clean unit suites hide inside `src/` (`test_node_choosing_strategy.py`, `test_model_patch_toggle.py`); `test_noisy_pisam_learning.py:74` hardcodes `absulute_path_prefix = Path("/Users/shakedsapir/…")` (typo included) and `test_conflict_search.py` depends on gitignored data — neither runs on a clean clone. Zero tests for: `run_fold.py`, `data_source.py`, `collect_results.py`/`result_schema.py`, `algorithms.py`, masking/noising strategies, all of `src/utils/`. The masking/noising strategies and `result_schema`/`collect_results` are pure and cheap to test — start there, with tiny checked-in fixture trajectories.

**2.26 Config.** `config.example.yaml` is badly out of sync with what the code reads: `image_llm_backend_factory.py:69-72` expects nested `openai:`/`google:` vendor blocks with `{visual_components|conflict_verification}_model.model_name/temperature`, the example has flat keys and no `google:` at all — a fresh clone following the example crashes. Its `domains:` lists `blocks`/`hanoi` while the registry keys are `blocksworld`/`npuzzle`/`hanoi`/`hiking`/`maze`/`depot`/`gripper`, and the `generation:` sub-block is undocumented. `dashboard_config.yaml:2` references two scripts that don't exist. Regenerate the example from the real `config.yaml`.

**2.27 requirements.txt.** Only 3 of 13 entries pinned (thesis reproducibility!); `anytree`/`imageio` unused; **missing** at least: `openai`, `google-genai`, `amlgym`, `unified_planning`, `tarski`, `scikit-learn`, `plotly`, `Pillow`, `openpyxl`. `.gitignore` ignores literal `venv` but not the documented `venv11`; add `venv*/`, `.pytest_cache/`, `.env`, and root-artifact rules; resolve the contradiction of gitignored-but-imported `src/experiments`.

---

## Suggested order of attack

1. **Today:** revoke the API keys (2.1); fix `gt_rate` handling and GT-transition counting (2.2, 2.3) before running more simulated experiments; fix the `as_completed` timeout (2.4).
2. **Finish the standalone-algorithms refactor** (Part 1, items 1-6): double-save, dead `denoiser_algo_name`, phase-vocabulary rename, dead branch, unknown-key warning, delete the two dead scripts. Small, mechanical, closes the branch cleanly.
3. **`CDPSConfig` dataclass** (2.6) — the single highest-leverage change; it deletes the `NOISY_PISAM` config bag and shrinks four signatures at once.
4. **`SearchResult` dataclass + `conflict_search.py` split** (2.7), then `run_single_fold` split (2.8, 2.9).
5. **DRY pass** (2.10-2.16) and logging (2.17-2.18) opportunistically, as you touch each file.
6. **Hygiene weekend** (2.24-2.27): root cleanup, tests consolidation, config example, requirements pinning.
