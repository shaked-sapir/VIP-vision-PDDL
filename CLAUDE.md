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
| `src/trajectory_handlers/` | Orchestrate the full image→trajectory pipeline per domain. Composes an `ObjectDetector` + `FluentClassifier`. |
| `src/pi_sam/` | Learning logic: PI-SAM, Noisy-PI-SAM, masking strategies, conflict patching. |
| `src/pi_sam/masking/` | Masking strategies (random, percentage, uncertain). Isolated from learning logic. |
| `src/pi_sam/noisy_pisam/` | Noise-aware learning variant via mixin composition. |
| `src/pi_sam/plan_denoising/` | Denoising trajectories at the plan level before learning. |
| `src/domains/` | Domain-specific gym environments (pddlgym wrappers). One subdir per domain. |
| `src/action_model/` | Parsers between PDDL ↔ gym ↔ SAM formats. |
| `src/llms/` | LLM integration: prompts, constants, ground-truth files, precision/recall evaluation. |
| `src/utils/` | **Shared utilities only.** No domain-specific logic here. |
| `src/typings/` | Shared type aliases and TypedDicts. No logic. |
| `benchmark/` | Experiment runners, data generators, evaluation scripts. Not part of the library. |

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

### Image Trajectory Handler
```python
# src/trajectory_handlers/image_trajectory_handler.py
class ImageTrajectoryHandler(ABC):
    @abstractmethod
    def init_visual_components(self, *args, **kwargs) -> None: ...
```
- Holds an `ObjectDetector` and `FluentClassifier` as instance attributes.
- Subclasses only need to implement `init_visual_components` to wire in the correct detector/classifier.
- Do not re-implement `construct_trajectory_from_images` or `image_trajectory_pipeline` — those are inherited.

### Masking Strategy
```python
# src/pi_sam/masking/masking_strategies.py
class MaskingStrategy(ABC):
    @abstractmethod
    def mask(self, predicates: set[GroundedPredicate], *args, **kwargs)
        -> Tuple[set[GroundedPredicate], set[GroundedPredicate]]: ...
```
- Returns `(masked, unmasked)`. Always a tuple — don't change the contract.

---

## Modularity & Code Reuse Rules

These rules are non-negotiable. Before writing any new code:

### 1. Search `src/utils/` first
Every utility function that is not domain-specific belongs in `src/utils/`. Before writing a helper, check:

| File | What it provides |
|---|---|
| `utils/containers.py` | `to_list`, `serialize`, `group_objects_by_key`, `sort_objects_numerically`, `shrink_whitespaces` |
| `utils/pddl_state.py` | `get_state_grounded_predicates`, `compare_states`, `copy_observation`, `get_all_possible_groundings` |
| `utils/pddl_gym.py` | `set_problem_by_name`, `ground_action`, `parse_gym_to_pddl_literal`, `translate_pddlgym_state_to_image_predicates` |
| `utils/pddl_trajectory.py` | `build_trajectory_file`, `propagate_frame_axioms_in_trajectory`, `inject_gt_states_by_percentage` |
| `utils/masking.py` | `mask_state`, `mask_observation`, `save_masking_info`, `load_masking_info` |
| `utils/visualize.py` | `draw_objects`, `find_exact_rgb_color_mask`, `encode_image_to_base64` |
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

1. `src/domains/{domain}/` — gym environment wrapper
2. `src/object_detection/{domain}_object_detector.py` — extends `ObjectDetector`
3. `src/fluent_classification/{domain}_fluent_classifier.py` — extends `FluentClassifier`
4. `src/trajectory_handlers/{domain}_trajectory_handler.py` — extends `ImageTrajectoryHandler`, implements `init_visual_components`
5. `src/llms/{domain}/` — constants, prompts, ground-truth files (if LLM-based)
6. `config.yaml` — add domain entry with problem paths and parameters
7. `benchmark/data/{domain}/` — trajectory data directory
8. Update any domain registry/switch in `simulator.py` or `lab_simulator.py`

Do **not** duplicate trajectory pipeline logic — `ImageTrajectoryHandler` handles it.

---

## Key Architectural Patterns

- **Template Method** — base classes define the pipeline; subclasses fill in the domain-specific steps.
- **Mixin Composition** — noise handling is a mixin (`NoisyLearnerMixin`) layered onto learning classes, not a deep subclass.
- **Strategy** — masking behavior is swappable via `MaskingStrategy` subclasses.
- **Closed-World Assumption** — when a fluent is absent from a state, it is assumed false. `UNCERTAIN` breaks this assumption and triggers masking.
- **Frame-Axiom Propagation** — unmasked fluents are propagated across states using frame axioms before trajectory files are written (`utils/pddl_trajectory.py`).

---

## Experiments & Benchmarking

- Entry points: `benchmark/amlgym_testing.py`, `benchmark/run_reorder_experiment.py`
- Data lives under `benchmark/data/{domain}/`
- Evaluation: `benchmark/evaluation/cfm_quality_analysis.py`
- Configuration: `config.yaml` at project root
- Activate environment: `source venv11/bin/activate`

---

## Git Workflow — Feature Branches

When asked to implement a feature in a new branch:

1. **Never switch the current working directory's branch.** The user operates parallel Cloud instances on the original branch and must not be disrupted.
2. Use `git worktree add` to check out the new branch in a sibling directory:
   ```bash
   git worktree add ../VIP-vision-PDDL-<feature-name> <new-branch-name>
   ```
3. Implement the feature entirely inside the worktree directory (`../VIP-vision-PDDL-<feature-name>/`).
4. Never run `git checkout` or `git switch` in the main working directory as part of a feature branch workflow.
5. When done, inform the user of the worktree path and branch name so they can review or merge.

---

## What Not To Do

- Do not import from `benchmark/` into `src/` — benchmark depends on src, not the reverse.
- Do not add LLM prompt strings outside of `src/llms/{domain}/`.
- Do not write trajectory files manually — use `utils/pddl_trajectory.py`.
- Do not skip the base class when adding a new detector/classifier — the trajectory handler expects the interface.
