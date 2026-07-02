# CLAUDE.md — Project Context & Coding Guidelines

## Project Overview

**VIP-vision-PDDL** learns PDDL action models from unsupervised visual traces (video/image sequences of agents performing tasks). The pipeline converts video frames into PDDL trajectories via object detection and fluent classification, then feeds those trajectories into a SAM-based learner to construct or refine a partial action model.

The system handles **partial observability** (masked/uncertain fluents from noisy classifiers) and **noisy observations** (conflicting trajectories) via the PI-SAM and Noisy-PI-SAM learning variants.

---

## Module Map

Every module under `src/` has a clear owner responsibility. Do not cross these boundaries.

| Directory | Responsibility |
|---|---|
| `src/object_detection/` | Detect objects in a single image frame. Input: image. Output: list of `BoundedObject`. |
| `src/fluent_classification/` | Classify PDDL fluents from a single image frame. Input: image. Output: `Dict[str, PredicateTruthValue]`. |
| `src/trajectory_handlers/` | Image→trajectory pipeline per domain. Split into inference base, PDDLGym source, external source, and LLM mixin layers. |
| `src/pi_sam/` | Learning logic: PI-SAM, Noisy-PI-SAM, masking/noising strategies, conflict patching. |
| `src/pi_sam/masking/` | Masking strategies (random, percentage, uncertain). Isolated from learning logic. |
| `src/pi_sam/noising/` | Noising strategies (random, percentage) for flipping unmasked predicate polarity. |
| `src/pi_sam/noisy_pisam/` | Noise-aware learning variant via mixin composition. |
| `src/pi_sam/plan_denoising/` | Denoising trajectories at the plan level before learning. |
| `src/domains/` | Domain PDDL files and domain-specific experiment code. One subdir per domain. |
| `src/action_model/` | Parsers between PDDL ↔ gym ↔ SAM formats. |
| `src/llms/` | LLM integration: prompts, constants, ground-truth files, precision/recall evaluation. |
| `src/utils/` | **Shared utilities only.** No domain-specific logic here. |
| `src/typings/` | Shared type aliases and TypedDicts. No logic. |
| `benchmark/` | Experiment runners, data generators, evaluation scripts, pluggable baselines. Not part of the library. |
| `benchmark/baselines/` | Pluggable baseline algorithm runners (e.g. ROSAME) registered for `benchmark_runner.py`. |

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
- Domain-specific detectors live in `src/object_detection/` and are named `{domain}_object_detector.py`.

### Fluent Classifier
```python
# src/fluent_classification/base_fluent_classifier.py
class FluentClassifier(ABC):
    def classify(self, image_path: Path | str) -> Dict[str, PredicateTruthValue]: ...
```
- Returns a dict of predicate string → `PredicateTruthValue` (TRUE / FALSE / UNCERTAIN).
- `UNCERTAIN` is meaningful — it drives masking downstream. Never silently drop it.
- Domain-specific classifiers live in `src/fluent_classification/` and are named `{domain}_fluent_classifier.py` or `llm_{domain}_fluent_classifier.py`.

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
    # Wires LLM detector + classifier; override _pre_init_hook for domain prep.
```

```python
# Concrete LLM combos (empty pass-through classes — set class attrs + hooks in domain file):
class LLMImageTrajectoryHandler(LLMVisualComponentsMixin, PDDLGymImageTrajectoryHandler): ...
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
| `utils/pddl_state.py` | `get_state_grounded_predicates`, `compare_states`, `compare_observations`, `copy_state`, `copy_observation`, `flip_fluent_in_state`, `ground_observation_completely`, `get_all_possible_groundings` |
| `utils/pddl_gym.py` | `set_problem_by_name`, `ground_action`, `parse_gym_to_pddl_literal`, `parse_gym_to_pddl_ground_action`, `translate_pddlgym_state_to_image_predicates`, `extract_objects_from_pddlgym_state` |
| `utils/pddl_trajectory.py` | `build_trajectory_file`, `observation_to_trajectory_file`, `ensure_trajectory_json`, `extract_actions_from_trajectory_json`, `propagate_frame_axioms_in_trajectory`, `propagate_frame_axioms_in_memory`, `propagate_frame_axioms_selective`, `inject_gt_states_by_percentage` |
| `utils/trajectory_json_converter.py` | `convert_trajectory_to_json` — `.trajectory` + `.pddl` → `_trajectory.json` |
| `utils/masking.py` | `mask_state`, `mask_observation`, `mask_observations`, `save_masking_info`, `load_masking_info`, `load_masked_observation` |
| `utils/visualize.py` | `draw_objects`, `find_exact_rgb_color_mask`, `load_image`, `encode_image_to_base64` |
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

1. `src/domains/{domain}/` — PDDL domain + problem files
2. `src/object_detection/{domain}_object_detector.py` or `llm_{domain}_object_detector.py`
3. `src/fluent_classification/{domain}_fluent_classifier.py` or `llm_{domain}_fluent_classifier.py`
4. `src/trajectory_handlers/` — pick the right handler base:
   - PDDLGym images: `{domain}_image_trajectory_handler.py` → `PDDLGymImageTrajectoryHandler`, or `llm_{domain}_trajectory_handler.py` → `LLMImageTrajectoryHandler`
   - External images: `llm_{domain}_trajectory_handler.py` → `LLMExternalImageTrajectoryHandler`
   - Set `detector_class` / `classifier_class` for LLM handlers; override `_rename_ground_action`, `_manipulate_trajectory_json`, or `_pre_init_hook` as needed
5. `src/llms/domains/{domain}/` — constants, prompts, ground-truth files (if LLM-based)
6. `config.yaml` — add domain entry with problem paths and parameters
7. `benchmark/data/{domain}/` — trajectory data directory
8. Update domain registry/switch in `simulator_cli.py`, `simulator.py`, or `lab_simulator.py`

Do **not** duplicate inference, masking, or trajectory-file logic — inherit from the handler hierarchy. Only implement `init_visual_components` (or LLM class attrs) and domain-specific hooks.

---

## Key Architectural Patterns

- **Template Method** — base classes define the pipeline; subclasses fill in domain-specific steps via hooks.
- **Handler Layering** — `ImageTrajectoryHandler` (inference) ← `PDDLGymImageTrajectoryHandler` / `ExternalImageTrajectoryHandler` (data source) ← domain concrete class. LLM domains add `LLMVisualComponentsMixin` via multiple inheritance.
- **Mixin Composition** — noise handling (`NoisyLearnerMixin`) and LLM wiring (`LLMVisualComponentsMixin`) are mixins, not deep subclass chains.
- **Strategy** — masking via `MaskingStrategy`; noising via `NoisingStrategy`. Baseline algorithms via `benchmark/baselines/` registry (`get_baselines`).
- **Closed-World Assumption** — when a fluent is absent from a state, it is assumed false. `UNCERTAIN` breaks this assumption and triggers masking.
- **Frame-Axiom Propagation** — unmasked fluents are propagated across states using frame axioms before trajectory files are written (`utils/pddl_trajectory.py`).

---

## Experiments & Benchmarking

- Entry points: `benchmark/benchmark_runner.py`, `benchmark/data_generator.py`, `src/simulator_cli.py`
- Baselines: `benchmark/baselines/` — register runners in `BASELINE_REGISTRY`, select via `--baselines` in `benchmark_runner.py`
- Data lives under `benchmark/data/{domain}/`
- Evaluation: `benchmark/evaluation/cfm_quality_analysis.py`, `benchmark/evaluation/cfm_domain_aggregate.py`
- Configuration: `config.yaml` at project root
- Activate environment: `source venv11/bin/activate`

**Supported domains:** blocksworld, hanoi, hiking, maze, n_puzzle (PDDLGym); depot, gripper (external images).

---

## Git Workflow — Feature Branches

When asked to implement a feature in a new branch:

1. **Default: create a new branch directly** using `git checkout -b <branch-name>` in the working directory and implement the feature there.
2. Only use `git worktree add` if the user **explicitly** asks for a worktree.
3. When done, inform the user of the branch name so they can review or merge.

---

## What Not To Do

- Do not import from `benchmark/` into `src/` — benchmark depends on src, not the reverse.
- Do not add LLM prompt strings outside of `src/llms/{domain}/`.
- Do not write trajectory files manually — use `utils/pddl_trajectory.py`.
- Do not skip the base class when adding a new detector/classifier — the trajectory handler expects the interface.
