# CLAUDE.md — Project Context & Coding Guidelines

## Project Overview

**VIP-vision-PDDL** learns PDDL action models from unsupervised visual traces (video/image sequences of agents performing tasks). The pipeline converts video frames into PDDL trajectories via object detection and fluent classification, then feeds those trajectories into a **PI-SAM** learner with optional **Conflict-Directed Patch Search (CDPS)** to resolve conflicting observations and construct a partial action model.

The system handles **partial observability** (masked/uncertain fluents from noisy classifiers) and **noisy observations** (conflicting trajectories) via PI-SAM masking/noising plus CDPS conflict patching in `src/pi_sam/plan_denoising/`.

---

## Module Map

Every module under `src/` has a clear owner responsibility. Do not cross these boundaries.

| Directory | Responsibility |
|---|---|
| `src/object_detection/` | Detect objects in a single image frame. Input: image. Output: list of `BoundedObject`. |
| `src/fluent_classification/` | Classify PDDL fluents from a single image frame. Input: image. Output: `Dict[str, PredicateTruthValue]`. LLM path uses `ImageLLMBackend` protocol + `ImageLLMBackendFactory` (OpenAI/Gemini). |
| `src/trajectory_handlers/` | Image→trajectory pipeline per domain. Split into inference base, PDDLGym source, external source, LLM mixin layers, and `pddlgym_problem_generator.py` (ROSAME-style problem generation). |
| `src/pi_sam/` | Learning logic: PI-SAM, Noisy-PI-SAM, masking/noising strategies, conflict patching. |
| `src/pi_sam/masking/` | Masking strategies (random, percentage, uncertain). Isolated from learning logic. |
| `src/pi_sam/noising/` | Noising strategies (random, percentage) for flipping unmasked predicate polarity. |
| `src/pi_sam/noisy_pisam/` | Noise-aware learning variant via mixin composition. |
| `src/pi_sam/plan_denoising/` | Conflict-Directed Patch Search (CDPS): `conflict_search.py`, `conflict_search_config.py` (`CDPSConfig`), `frontier.py` (search mode / node / conflict-group / fluent-branch strategies). GT-aware via `gt_states_by_obs` (init state always GT). Also `evaluator.py` — `observations_reconstruction_score`, a **ground-truth-free** model score (effect mismatches + inapplicability events, rollout and one-step, against the frozen original observations). Use it for online selection; **never** `benchmark/experiment_running_helpers/evaluation.py:evaluate_model`, which needs GT and is for offline reporting only. |
| `src/pi_sam/plan_denoising/milp_version/` | The same repair problem solved as one CP-SAT program instead of by search: `encoder.py`, `encoding_config.py` (`MilpEncodingConfig`, presets `upstream()`/`tag()`/`cdps_dialect()`), `converter.py` (pddl_plus → `planning_structs`, `GtAnchoring`, `RepeatedArgsInstance`), `config.py` (`CdpsMilpConfig`), `trajectory_extraction.py` (solved MILP → re-masked T′), `single_round.py` (the `cdps_milp_single_round` driver), `vendor/` (upstream code, see `vendor/UPSTREAM.md`). Shared with the `rosame_milp*` baselines — hence `src/`, not `benchmark/`. |
| `src/domains/` | Domain PDDL files and per-domain problem folders (`problems/problemN/`). One subdir per domain. |
| `src/action_model/` | Parsers between PDDL ↔ gym ↔ SAM formats. |
| `src/llms/` | LLM integration: prompts, constants, ground-truth files, precision/recall evaluation. |
| `src/utils/` | **Shared utilities only.** No domain-specific logic here. |
| `src/typings/` | Shared type aliases and TypedDicts. No logic. |
| `benchmark/` | Experiment runners, data generators, evaluation scripts. Not part of the library. |
| `benchmark/experiment_running_helpers/` | Fold execution glue: `run_fold.py` (spine), `data_source.py` (image vs simulated), `learning_helpers.py` (CDPS), `trajectory_utils.py` (fold prep / anchored GT), `resume.py` (fold I/O + `run_params` checks), `gt_builder.py` (GT export/validation), `collect_results.py` + `result_schema.py` (per-cell JSON → tables). |
| `benchmark/algorithm_adapters/` | ROSAME PO encoding (`po_rosame_runner.py`) and `rosame_milp/` (the ROSAME↔MILP bridge + milp loop; the encoder itself lives in `src/pi_sam/plan_denoising/milp_version/`). Used by `baselines/`. CDPS calls `ConflictDrivenPatchSearch` directly from `learning_helpers.py`. |
| `benchmark/baselines/` | Pluggable competitor runners (`BaselineRunner` ABC) registered in `BASELINE_REGISTRY` (`rosame`, `rosame_i`, `rosame_milp*`). |
| `benchmark/diagnosis/` | CDPS search-trace tooling: `trace_serialization.py`, `visualize_trace.py`, `retrace_search.py`. |
| `benchmark/simulated_version/` | Simulated-experiment utilities: `run_simulated_experiment.py`, `noise_injection.py`, `noise_evaluation.py`. |

---

## Base Class Contracts

When adding a new detector, classifier, or trajectory handler, **always extend the base class**. Never duplicate the interface.

### Object Detector
```python
# src/object_detection/base_object_detector.py
class ObjectDetector(ABC):
    @abstractmethod
    def detect(self, image: Union[cv2.typing.MatLike, Path, str], *args, **kwargs): ...
```
- One method. Input is flexible (Mat, Path, str). Return type is domain-defined but conventionally a list of `BoundedObject`.
- Domain-specific detectors live in `src/object_detection/` and are named `{domain}_object_detector.py` or `llm_{domain}_object_detector.py`.
- LLM detectors extend `llm_object_detector.py`.

### Fluent Classifier
```python
# src/fluent_classification/base_fluent_classifier.py
class FluentClassifier(ABC):
    def classify(self, image_path: Path | str) -> Dict[str, PredicateTruthValue]: ...
```
- Returns a dict of predicate string → `PredicateTruthValue` (TRUE / FALSE / UNCERTAIN).
- `UNCERTAIN` is meaningful — it drives masking downstream. Never silently drop it.
- Domain-specific classifiers live in `src/fluent_classification/` and are named `{domain}_fluent_classifier.py` or `llm_{domain}_fluent_classifier.py`.
- LLM classifiers extend `llm_fluent_classifier.py`; vendor backends implement `ImageLLMBackend` (`openai_image_llm_backend.py`, `gemini_image_llm_backend.py`).

### Image Trajectory Handler (hierarchy)

Trajectory handlers are split into layers. **Do not put PDDLGym or external-source logic in the base class.**

```python
# src/trajectory_handlers/image_trajectory_handler.py
class ImageTrajectoryHandler(ABC):
    @abstractmethod
    def init_visual_components(self, *args, **kwargs) -> None: ...
    @abstractmethod
    def run_pipeline(self, problem_name: str, images_path: Path, **kwargs) -> List[dict]: ...
```
- Base class owns the **inference pipeline only**: classify images → `.trajectory` + `.masking_info`.
- Inherited helpers: `construct_trajectory_from_images`, `image_trajectory_pipeline`, `create_masking_info`, `create_trajectory_and_masks`.
- Override `_rename_ground_action` for domain-specific action name transforms.
- **Entry point for callers is `run_pipeline`**, not the old gym-only methods directly.

```python
# src/trajectory_handlers/pddlgym_trajectory_handler.py
class PDDLGymImageTrajectoryHandler(ImageTrajectoryHandler):
    # Adds gym rendering, stepping, GT trajectory JSON generation.
    # run_pipeline: create_trajectory_from_gym → create_trajectory_and_masks
```

```python
# src/trajectory_handlers/external_trajectory_handler.py
class ExternalImageTrajectoryHandler(ImageTrajectoryHandler):
    # Reads pre-existing state_*.png + .trajectory files from disk.
    # run_pipeline: ensure_trajectory_json → extract actions → create_trajectory_and_masks
```

```python
# src/trajectory_handlers/llm_visual_components_mixin.py
class LLMVisualComponentsMixin:
    detector_class: Type[ObjectDetector]
    classifier_class: Type[FluentClassifier]

    def __init__(self, *, pddl_domain_file: Path, vendor: str = "openai", **kwargs): ...
    # Cooperative init — parses domain via DomainParser(partial_parsing=True).
    # init_visual_components: _pre_init_hook → detect objects → create classifier.
    # Override _pre_init_hook for domain prep (e.g. ensure trajectory JSON).
```

```python
# Concrete LLM combos (pass-through classes in dedicated files — set class attrs + hooks in domain file):
# src/trajectory_handlers/llm_image_trajectory_handler.py
class LLMImageTrajectoryHandler(LLMVisualComponentsMixin, PDDLGymImageTrajectoryHandler): ...
# src/trajectory_handlers/llm_external_trajectory_handler.py
class LLMExternalImageTrajectoryHandler(LLMVisualComponentsMixin, ExternalImageTrajectoryHandler): ...
```

**Choosing a base when adding a domain:**
- Images generated by PDDLGym → subclass `PDDLGymImageTrajectoryHandler` (deterministic) or `LLMImageTrajectoryHandler` (LLM).
- Images come from external files → subclass `ExternalImageTrajectoryHandler` or `LLMExternalImageTrajectoryHandler`.
- Domain handlers are named `llm_{domain}_trajectory_handler.py` (LLM) or `{domain}_image_trajectory_handler.py` (deterministic).

### Masking Strategy
```python
# src/pi_sam/masking/masking_strategies.py
class MaskingStrategy(ABC):
    @abstractmethod
    def mask(self, predicates: set[GroundedPredicate], *args, **kwargs)
        -> Tuple[set[GroundedPredicate], set[GroundedPredicate]]: ...
```
- Returns `(masked, unmasked)`. Always a tuple — don't change the contract.

### Noising Strategy
```python
# src/pi_sam/noising/noising_strategies.py
class NoisingStrategy(ABC):
    @abstractmethod
    def noise(self, predicates: Set[GroundedPredicate], *args, **kwargs) -> Set[GroundedPredicate]: ...
```
- Returns the subset of predicates whose polarity should be flipped. Does **not** mutate — caller uses `flip_fluent_in_state` from `utils/pddl_state.py`.
- Pass only unmasked predicates; the strategy has no masking awareness.

---

## Modularity & Code Reuse Rules

These rules are non-negotiable. Before writing any new code:

### 1. Search `src/utils/` first
Every utility function that is not domain-specific belongs in `src/utils/`. Before writing a helper, check:

| File | What it provides |
|---|---|
| `utils/containers.py` | `to_list`, `serialize`, `group_objects_by_key`, `sort_objects_numerically`, `shrink_whitespaces` |
| `utils/pddl_state.py` | `get_state_grounded_predicates`, `get_state_unmasked_predicates`, `get_state_masked_predicates`, `find_predicate_negation`, `state_positive_set`, `compare_states`, `compare_observations`, `observations_equal`, `copy_state`, `copy_observation`, `copy_observation_linked`, `flip_fluent_in_state`, `ground_observation_completely`, `ground_all_predicates_in_state`, `ground_all_states_in_observation`, `get_all_possible_groundings`, `get_all_possible_groundings_for_domain` |
| `utils/pddl_gym.py` | `set_problem_by_name`, `ground_action`, `parse_gym_to_pddl_literal`, `parse_gym_to_pddl_ground_action`, `multi_replace_predicate`, `translate_pddlgym_state_to_image_predicates`, `extract_objects_from_pddlgym_state`, `translate_problem_pddl_text` |
| `utils/pddl_trajectory.py` | `build_trajectory_file`, `observation_to_trajectory_file`, `ensure_trajectory_json`, `extract_actions_from_trajectory_json`, `propagate_frame_axioms_in_trajectory`, `propagate_frame_axioms_in_memory`, `propagate_frame_axioms_selective`, `inject_gt_states_by_percentage` |
| `utils/trajectory_json_converter.py` | `convert_trajectory_to_json` — `.trajectory` + `.pddl` → `_trajectory.json` |
| `utils/masking.py` | `mask_state`, `mask_observation`, `mask_observations`, `save_masking_info`, `load_masking_info`, `load_masked_observation` |
| `utils/visualize.py` | `draw_objects`, `to_int_rgb`, `find_exact_rgb_color_mask`, `load_image`, `encode_image_to_base64` |
| `utils/time.py` | `create_experiment_timestamp` |
| `utils/config.py` | `load_config` with project-relative path resolution |

`utils/pddl.py` is a re-export hub for backward compatibility — prefer importing from the specific module directly.

### 2. Do not duplicate per-domain logic
If two domain implementations share logic, extract it:
- Shared logic between domains → `src/utils/`
- Shared logic within a module family (e.g., all LLM classifiers) → a base class or mixin in that module

### 3. Compose, don't inherit deeply
The codebase uses **mixin composition** deliberately (e.g., `NoisyPisamLearner(NoisyLearnerMixin, PISAMLearner)`). Prefer this over deep inheritance chains. A new variant = a new mixin or a new concrete class combining existing pieces.

### 4. One responsibility per function
- Functions must do one thing. If a function both loads data **and** transforms it, split it.
- Keep functions short enough to read in one screen. If a function exceeds ~40 lines, consider splitting.

### 5. Don't add new module-level state
Avoid global variables and module-level mutable state. Pass config/dependencies explicitly via `__init__` parameters.

---

## Coding Standards

- **Type hints on all function signatures** — including return types.
- **Docstrings on all public methods** — one-line minimum, use Google style for complex ones.
- **No silent `except` blocks** — always log or re-raise. Never `except: pass`.
- **Prefer `Path` over `str` for file paths** — use `pathlib.Path` throughout; convert at I/O boundaries only.
- **Enums over magic strings** — `PredicateTruthValue`, `MaskingType`, `NegativePreconditionPolicy` are the pattern. New categorical values → new `Enum`.
- **Config via `config.yaml`** — no hardcoded paths or magic numbers in source code. Load via `src/utils/config.py`.
- **No global state** — avoid module-level mutable variables.

---

## Adding a New Domain

Checklist when adding support for a new planning domain:

1. `src/domains/{domain}/` — PDDL domain file + `problems/problemN/` folders (each with `{problem}.pddl`; external domains also ship `state_*.png` + GT `.trajectory`)
2. `src/object_detection/{domain}_object_detector.py` or `llm_{domain}_object_detector.py`
3. `src/fluent_classification/{domain}_fluent_classifier.py` or `llm_{domain}_fluent_classifier.py`
4. `src/trajectory_handlers/` — pick the right handler base:
   - PDDLGym images: `{domain}_image_trajectory_handler.py` → `PDDLGymImageTrajectoryHandler`, or `llm_{domain}_trajectory_handler.py` → `LLMImageTrajectoryHandler`
   - External images: `llm_{domain}_trajectory_handler.py` → `LLMExternalImageTrajectoryHandler`
   - Set `detector_class` / `classifier_class` for LLM handlers; override `_rename_ground_action`, `_manipulate_trajectory_json`, or `_pre_init_hook` as needed
5. `src/llms/domains/{domain}/` — constants, prompts, ground-truth files (if LLM-based)
6. `config.yaml` — add domain entry with problem paths and parameters
7. `benchmark/data/{domain}/` — trajectory data directory
8. Add an entry to `_DOMAIN_REGISTRY` in `benchmark/data_generator.py` and a matching key in `config.yaml`

Do **not** duplicate inference, masking, or trajectory-file logic — inherit from the handler hierarchy. Only implement `init_visual_components` (or LLM class attrs) and domain-specific hooks.

---

## Key Architectural Patterns

- **Template Method** — base classes define the pipeline; subclasses fill in domain-specific steps via hooks.
- **Handler Layering** — `ImageTrajectoryHandler` (inference) ← `PDDLGymImageTrajectoryHandler` / `ExternalImageTrajectoryHandler` (data source) ← domain concrete class. LLM domains add `LLMVisualComponentsMixin` via multiple inheritance.
- **Mixin Composition** — noise handling (`NoisyLearnerMixin`) and LLM wiring (`LLMVisualComponentsMixin`) are mixins, not deep subclass chains.
- **Strategy** — masking via `MaskingStrategy`; noising via `NoisingStrategy`. Algorithm selection via `benchmark/algorithms.py` (`cdps` / `cdps_anchored` + baseline keys from `benchmark/baselines/`).
- **DataSource** — `ImageDataSource` vs `SimulatedDataSource` in `benchmark/experiment_running_helpers/data_source.py` decouple observation preparation from fold execution.
- **CDPSConfig** — immutable dataclass in `conflict_search_config.py` bundles search *shape* (`search_mode`, `node_choosing_strategy`, `conflict_group_strategy`, `fluent_branch_mode`, patch costs/weights, `negative_precondition_policy`, `seed`); runtime timeout and `gt_states_by_obs` are passed separately to `ConflictDrivenPatchSearch.run`.
- **GT-aware CDPS** — search always respects `gt_states_by_obs` (init state is GT; never fluent-patch GT states; reject GT-refuted model constraints). The anchored variant (`cdps_anchored`) differs in **data prep** (also inject final state as GT via `trajectory_utils` / `inject_gt_states_by_percentage(..., anchor_final=True)`), not search shape.
- **Result schema** — per-cell results live in `testing/.../fold_result.json`; `collect_results.py` + `result_schema.py` flatten them for reports (no per-experiment CSVs).
- **Resume** — `resume.py` skips completed folds and aborts on `run_params` mismatch when `resume: true` in `run_config.yaml`.
- **Backfill** — `backfill_baseline.py` and `backfill_cdps.py` retrofit algorithm rows into existing cells using frozen `original_observations/` (shared helpers in `backfill_common.py`), without regenerating data.
- **Cluster orchestration** — `scripts/cluster/` expands `run_config.yaml` into SLURM array jobs (`make_manifest.py` → `run_cell.sbatch` / `run_fold.sbatch` + `submit.sh`).
- **Factory** — LLM vendor/model selection via `ImageLLMBackendFactory.create(vendor, model_type)`; config loaded from `config.yaml`.
- **Closed-World Assumption** — when a fluent is absent from a state, it is assumed false. `UNCERTAIN` breaks this assumption and triggers masking.
- **Frame-Axiom Propagation** — unmasked fluents are propagated across states using frame axioms before trajectory files are written (`utils/pddl_trajectory.py`).

---

## Experiments & Benchmarking

- Entry points:
  - `benchmark/benchmark_runner.py` — config-driven batch runner (`benchmark/run_config.yaml`); delegates each cell to `experiment_runner`
  - `benchmark/experiment_runner.py` — single experiment (one domain, one data source); `--algorithms cdps rosame` (default)
  - `benchmark/data_generator.py` — multi-problem trajectory generation via `_DOMAIN_REGISTRY`
  - `benchmark/generate_gt_trajectories.py` — GT backfill/validation CLI (`gt_builder.py`)
  - `benchmark/simulated_version/run_simulated_experiment.py` — standalone simulated runs
  - `benchmark/backfill_baseline.py` — retrofit baseline results into existing cells (`original_observations/` → learn → evaluate → merge into `fold_result.json`); supports `--workers N` for parallel cells
  - `benchmark/backfill_cdps.py` — retrofit `cdps_anchored` (or related CDPS rows) into existing cells via the same frozen-observation path (`backfill_common.py`)
  - `scripts/cluster/` — SLURM cluster launch: `make_manifest.py` + `submit.sh` + cell/fold sbatch scripts
- Algorithms: `benchmark/algorithms.py` — `cdps` (init GT), `cdps_anchored` (init+final GT via data prep) and `cdps_milp_single_round` (MILP denoiser + PI-SAM), plus baseline keys from `benchmark/baselines/BASELINE_REGISTRY` (`rosame`, `rosame_i`, `rosame_milp`, `rosame_milp_tag`, `rosame_milp_base`). `resolve_algorithms` → `(run_cdps, run_cdps_anchored, run_cdps_milp, baseline_runners)`. Legacy run configs may use `baselines: [rosame]` (implies CDPS too). MILP encoding rule-sets are bundled in `src/pi_sam/plan_denoising/milp_version/encoding_config.py` (`MilpEncodingConfig`, presets `upstream()`/`tag()`/`cdps_dialect()`); a new MILP variant = a new preset + a thin runner/driver + one registry entry.
- MILP denoiser (`cdps_milp_single_round`): configured by the `cdps_milp:` block of `run_config.yaml` → `CdpsMilpConfig` (validated even when the arm is off), run from `learning_helpers.learn_cdps_milp_single_round()` → `milp_version/single_round.run_single_round()`, dispatched by `run_fold.run_cdps_phase(..., milp_config=...)` so both denoisers share one evaluation path. Row labels are arm-suffixed (`cdps_milp_algorithm_name`); artifacts land under `<fold>/cdps_milp_single_round/` in CDPS's shape plus `milp_repair_log.json` / `milp_masked_completion.json`. With `eq16: off` the reported `repair_cost` is a lower bound on any CDPS conflict-free model's cost on the same fold — the standing sanity check.
- Learning path: `learning_helpers.learn_cdps()` → `CDPSConfig` + `ConflictDrivenPatchSearch.run(..., gt_states_by_obs=..., timeout_seconds=...)` on masked observations (no `NOISY_PISAM` adapter; SAM-only paths removed). Optional `--events-tracing` / `events_tracing` writes `search_trace.json` per fold via `benchmark/diagnosis/trace_serialization.py`.
- Data lives under `benchmark/data/{domain}/`; experiment outputs under `benchmark/running_results/{domain}/{experiment_name}/`; finished batch manifests under `benchmark/finished_run_configs/<run_name>/`
- Evaluation: `benchmark/evaluation/experiment_report.py` (`fully-detailed-report.xlsx`), `benchmark/evaluation/cfm/cfm_quality_analysis.py`, `cfm_quality_table.py`, `cfm_domain_aggregate.py`, `build_dashboard.py`, `combine_dashboard_reports.py` (reads `dashboard_config.yaml`), `predictive_metrics.py` + `upenv_compat.py` (predictive-power eval), `fold_filter.py`, `trajectory_fluent_confusion.py`, `compare_original_observations.py`, `correlation_analysis.py`, `anytime/` (post-hoc anytime performance profiles: `checkpoints.py` reader → `score.py` GT-free scorer → `curves.py` → `run_anytime.py` CLI)
- Configuration: `config.yaml` at project root; batch runs via `benchmark/run_config.yaml` (`shared` search/resume/algorithm knobs; `simulation` mask/noise grid; `domains` list)
- Activate environment: `source venv11/bin/activate`

**Supported domains (config keys):** `blocksworld`, `hanoi`, `hiking`, `maze`, `npuzzle` (PDDLGym / generated); `depot`, `gripper` (external images). Deterministic blocksworld inference uses `BlocksContourFluentClassifier` + `ColorObjectDetector` via `blocks_image_trajectory_handler.py`.

---

## Git Workflow — Feature Branches

When asked to implement a feature in a new branch:

1. **Default: create a new branch directly** using `git checkout -b <branch-name>` in the working directory and implement the feature there.
2. Only use `git worktree add` if the user **explicitly** asks for a worktree.
3. When done, inform the user of the branch name so they can review or merge.

---

## What Not To Do

- Do not import from `benchmark/` into `src/` — benchmark depends on src, not the reverse.
- Do not add LLM prompt strings outside of `src/llms/domains/{domain}/`.
- Do not write trajectory files manually — use `utils/pddl_trajectory.py`.
- Do not skip the base class when adding a new detector/classifier — the trajectory handler expects the interface.
