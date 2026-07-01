# VIP-vision-PDDL — Repo Analysis Report

**Date:** 2026-07-01  
**Scope:** Full codebase scan — `src/`, `benchmark/`, root-level files

---

## 1. Dependency Issues

### 1.1 Circular Dependency: `utils` ↔ `object_detection`
- `utils/visualize.py` imports `BoundedObject` from `src/object_detection/bounded_object`
- Multiple object detectors import from `src/utils/visualize`
- Works at runtime but is fragile. **Fix:** move `BoundedObject` to `src/typings/` (it's a data class, not detection logic).

### 1.2 Boundary Violation: `src/` imports `experiments/`
Three files in `src/` import from an external `experiments` package:
- `lab_simulator.py` → `experiments.experiments_consts`
- `pi_sam/pisam_experiment_runner.py` → `experiments.basic_experiment_runner`, `experiments.experiments_consts`
- `vip_experiments/BasicSamExperimentRunnner.py` → `experiments.*`

This couples the library to experiment infrastructure. **Fix:** inject experiment config via constructor params or move these files to `benchmark/`.

### 1.3 Hardcoded Absolute Paths (11+ files)
Paths like `/Users/shakedsapir/...` appear in `__main__` blocks and test code across fluent classifiers, object detectors, and simulator scripts. Breaks on any other machine. **Fix:** use `config.yaml` or relative paths.

---

## 2. Code Smells

### 2.1 Duplicated Code
- **Color-to-object mappings** duplicated between `llm_blocks_object_detector.py` and `llm_blocks_fluent_classifier.py`. Should be a shared constant.
- **`_manipulate_trajectory_json` pattern** copy-pasted across `llm_hanoi_trajectory_handler.py` and `llm_npuzzle_trajectory_handler.py` with identical try/except wrappers.
- **Copy-pasted docstrings with wrong domain names:** `LLMHikingFluentClassifier` says "N-puzzle domain"; `LLMHikingObjectDetector` says "Slide (8-puzzle)" with disc/peg examples.
- **`benchmark/amlgym_models/ROSAME.py` vs `PO_ROSAME.py`** — near-identical code with `_record_timing` duplicated verbatim. A third variant exists in `benchmark/baselines/rosame_runner.py`.

### 2.2 God Functions
- `simulator_cli.py::main()` — 124 lines: config loading, path resolution, masking dispatch, two execution branches
- `lab_simulator.py::run_cross_validation()` — ~150+ lines
- `image_trajectory_handler.py::create_trajectory_from_gym()` and `construct_trajectory_from_images()` — both >50 lines with mixed I/O and state management

### 2.3 Dead Code
- `llm_npuzzle_fluent_classifier.py` lines 41-65 — large commented-out block
- `blocks_contour_fluent_classifier.py` lines 120-183 — inline test harness with hardcoded paths in production code
- `benchmark/data/n_puzzle_typed/fix_trajectories.py` — one-off script, never imported
- `benchmark/run_reorder_experiment.py` — never imported by other code

### 2.4 Magic Strings/Numbers
- Retry count `max_retries=3` hardcoded in `llm_fluent_classifier.py::extract_facts_once()`
- Pixel thresholds 5, 50, 100 in `hanoi_fluent_classifier.py` without named constants
- Grid range 30 hardcoded in both `llm_hiking_fluent_classifier.py` and `llm_hiking_object_detector.py`

### 2.5 Silent/Dangerous Exception Handling
- `image_trajectory_handler.py` line 165 — `except Exception:` silently resets state and retries in a while loop → **infinite loop risk**
- Trajectory handlers catch all exceptions and only `print()` a warning, continuing with potentially corrupted data

### 2.6 Inconsistent Interfaces
- `detect()` return types diverge: `color_object_detector.py` returns `List[BoundedObject]` while `llm_object_detector.py` returns `Dict[str, List[str]]`. The abstract base doesn't enforce a return type.
- `OpenAIImageLLMBackend` inherits from both `ImageLLMBackend` (Protocol) and `ABC` — the ABC is redundant.
- `self.use_uncertain` set twice in `llm_maze_fluent_classifier.py` (lines 25 and 33)

---

## 3. Structural Issues

### 3.1 Test Files Scattered
12 `test_*.py` files at the repo root instead of in `tests/`. The `tests/` directory only has 2 files. All root-level tests should be consolidated into `tests/`.

### 3.2 Oversized Files in benchmark/
- `benchmark/diagnosis/trace_spurious_effects.py` — 1,446 lines
- `benchmark/data_generator.py` — 1,068 lines
- `benchmark/experiment_running_helpers/reporting.py` — 854 lines
- `benchmark/amlgym_testing.py` — 826 lines

### 3.3 `vip_experiments/` Module
Unclear purpose — `BasicSamExperimentRunnner.py` (note the typo in the filename) imports from `experiments.*` which doesn't exist in the repo. Likely dead or belongs in `benchmark/`.

---

## 4. Prioritized Recommendations

| Priority | Issue | Effort | Impact |
|---|---|---|---|
| **P0** | Fix infinite-loop risk in `image_trajectory_handler.py` exception handler | Low | High (correctness) |
| **P0** | Fix `detect()` return type inconsistency — enforce contract in base class | Medium | High (reliability) |
| **P1** | Break `utils` ↔ `object_detection` circular dep (move `BoundedObject` to `typings/`) | Low | Medium |
| **P1** | Extract duplicated color mappings, trajectory manipulation patterns to shared locations | Low | Medium |
| **P1** | Move or delete `vip_experiments/` — it's broken/dead code | Low | Medium (cleanliness) |
| **P2** | Remove `experiments/` imports from `src/` — inject config instead | Medium | Medium |
| **P2** | Move 12 root-level test files into `tests/` | Low | Low (organization) |
| **P2** | Split god functions (simulator_cli main, run_cross_validation) | Medium | Medium |
| **P3** | Replace hardcoded paths with config references | Low | Low (portability) |
| **P3** | Clean up dead code (commented blocks, unused scripts) | Low | Low |
| **P3** | Replace magic numbers with named constants | Low | Low |
| **P3** | Fix copy-pasted wrong-domain docstrings | Low | Low |
