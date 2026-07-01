# Trajectory Handler Refactoring — Design Document

## Problem Summary

The `src/trajectory_handlers/` module has 10 files, 7 of which are LLM-based domain handlers that share ~60-70% identical code. The duplication:

- **`__init__`**: 7 handlers store the same fields (`api_key`, `vendor`, `object_detector_model`, etc.). Only npuzzle and maze omit the model/temperature params (they delegate to the factory).
- **`init_visual_components`**: Same 3-step pattern — create detector via factory, call `detect()`, create classifier with results. Only depot/gripper add `_ensure_trajectory_json` before.
- **`create_masking_info`**: Byte-for-byte identical across all 7 handlers.
- **`create_trajectory_and_masks`**: Byte-for-byte identical across all 7 handlers.
- **`_ensure_trajectory_json`**: Duplicated verbatim between depot and gripper.
- **Wrong docstrings**: Hiking says "blocksworld", maze says "Hanoi", npuzzle says "Hanoi".

The only genuinely unique logic per domain is:
- Which detector/classifier classes to instantiate
- `_rename_ground_action` (3 of 7 have meaningful overrides)
- `_manipulate_trajectory_json` (4 of 7 have overrides; hanoi and npuzzle are complex)
- `get_image_path_by_index` (depot only — unpadded filenames)

---

## Proposed Architecture

### Layer 1: `ImageTrajectoryHandler` (base — exists, needs changes)

Move `create_masking_info` and `create_trajectory_and_masks` **into the base class**. They are universal — every LLM handler defines them identically. This requires the base class to hold a `domain` attribute (already present in all subclasses).

```python
# image_trajectory_handler.py — additions to base class

class ImageTrajectoryHandler(ABC):
    domain: Domain  # <-- new required attribute

    def create_masking_info(self, problem_name: str, imaged_trajectory: list[dict], 
                            trajectory_path: Path) -> None:
        """Extract and save masking info from imaged trajectory."""
        trajectory_masking_info = (
            [parse_grounded_predicates(imaged_trajectory[0]['current_state']['unknown'], self.domain)] +
            [parse_grounded_predicates(step['next_state']['unknown'], self.domain)
             for step in imaged_trajectory]
        )
        save_masking_info(trajectory_path, problem_name, trajectory_masking_info)

    def create_trajectory_and_masks(self, problem_name: str, actions: List[str], 
                                     images_path: Path) -> List[dict]:
        """Run image_trajectory_pipeline + save masking info."""
        imaged_trajectory = self.image_trajectory_pipeline(problem_name, actions, images_path)
        self.create_masking_info(problem_name, imaged_trajectory, images_path)
        return imaged_trajectory
```

**Impact**: Deletes ~200 lines of duplicated code across 7 files (+ 1 in benchmark).

### Layer 2: `LLMTrajectoryHandler` (new intermediate base)

A new class between `ImageTrajectoryHandler` and the domain-specific handlers. Absorbs the shared LLM wiring boilerplate.

```python
# src/trajectory_handlers/llm_trajectory_handler.py

class LLMTrajectoryHandler(ImageTrajectoryHandler):
    """Base for all LLM-based trajectory handlers.
    
    Subclasses only need to set:
      - detector_class: Type[LLMObjectDetector subclass]
      - classifier_class: Type[LLMFluentClassifier subclass]
    And optionally override:
      - _rename_ground_action()
      - _manipulate_trajectory_json()
      - _ensure_trajectory_json()  (for domains with pre-existing .trajectory files)
    """

    # Subclass must set these (class-level attributes)
    detector_class: type     # e.g. LLMBlocksObjectDetector
    classifier_class: type   # e.g. LLMBlocksFluentClassifier

    def __init__(self, domain_name: str, pddl_domain_file: Path,
                 vendor: str = "openai"):
        super().__init__(domain_name=domain_name)
        self.vendor = vendor
        self.domain = DomainParser(pddl_domain_file, partial_parsing=True).parse_domain()

    def init_visual_components(self, init_state_image_path: Path) -> None:
        """Standard 3-step LLM init: detect objects, then create classifier."""
        self._pre_init_hook(init_state_image_path)
        
        self.object_detector = self.detector_class(
            llm_backend=ImageLLMBackendFactory.create(
                vendor=self.vendor, model_type="object_detection"),
            init_state_image_path=init_state_image_path
        )
        detected_objects: Dict[str, List[str]] = self.object_detector.detect(
            str(init_state_image_path))

        self.fluent_classifier = self.classifier_class(
            llm_backend=ImageLLMBackendFactory.create(
                vendor=self.vendor, model_type="fluent_classification"),
            type_to_objects=detected_objects,
            init_state_image_path=init_state_image_path
        )

    def _pre_init_hook(self, init_state_image_path: Path) -> None:
        """Override point for pre-init work (e.g. _ensure_trajectory_json)."""
        pass
```

**Design decisions:**
- `api_key`, `model`, `temperature` params removed from `__init__` — the factory already reads these from `config.yaml`. The handlers currently accept them but never use them directly (they pass `vendor` to the factory and the factory loads config). This is dead parameter passing.
- `vendor` is the only LLM config param needed at handler level.
- `_pre_init_hook` replaces the depot/gripper `_ensure_trajectory_json` pattern cleanly.

### Layer 3: Domain-specific handlers (drastically simplified)

Each domain handler becomes a thin configuration class. Here's what each reduces to:

#### Blocks (was 148 lines → ~30 lines)
```python
class LLMBlocksImageTrajectoryHandler(LLMTrajectoryHandler):
    detector_class = LLMBlocksObjectDetector
    classifier_class = LLMBlocksFluentClassifier

    @staticmethod
    def _rename_ground_action(action_str: str) -> str:
        return (action_str.replace('pick-up', 'pick_up')
                .replace('put-down', 'put_down')
                .replace(', robot:robot', ''))

    def _manipulate_trajectory_json(self, gt_trajectory_json: list) -> list:
        # ... blocksworld-specific transforms (unchanged)
```

#### Hiking (was 97 lines → ~8 lines)
```python
class LLMHikingImageTrajectoryHandler(LLMTrajectoryHandler):
    detector_class = LLMHikingObjectDetector
    classifier_class = LLMHikingFluentClassifier
    # No overrides needed — identity _rename_ground_action, no JSON manipulation
```

#### Depot (was 143 lines → ~20 lines)
```python
class LLMDepotImageTrajectoryHandler(LLMTrajectoryHandler):
    detector_class = LLMDepotObjectDetector
    classifier_class = LLMDepotFluentClassifier

    def get_image_path_by_index(self, image_dir: Path, idx: int) -> str:
        return self.get_image_full_path(image_dir, f"state_{idx}.png")

    def _pre_init_hook(self, init_state_image_path: Path) -> None:
        _ensure_trajectory_json(init_state_image_path.parent)
```

#### Gripper (was 135 lines → ~15 lines)
```python
class LLMGripperImageTrajectoryHandler(LLMTrajectoryHandler):
    detector_class = LLMGripperObjectDetector
    classifier_class = LLMGripperFluentClassifier

    def _pre_init_hook(self, init_state_image_path: Path) -> None:
        _ensure_trajectory_json(init_state_image_path.parent)
```

#### Maze (was 115 lines → ~15 lines)
```python
class LLMMazeImageTrajectoryHandler(LLMTrajectoryHandler):
    detector_class = LLMMazeObjectDetector
    classifier_class = LLMMazeFluentClassifier

    @staticmethod
    def _rename_ground_action(action_str: str) -> str:
        return action_str.replace('move-', 'move_')

    def _manipulate_trajectory_json(self, gt_trajectory_json: list) -> list:
        # ... hyphen-to-underscore in action names (unchanged)
```

#### Hanoi and N-Puzzle
Keep their `_rename_ground_action` and `_manipulate_trajectory_json` overrides — these contain genuinely complex domain-specific type translation logic that can't be simplified further without a larger rethink.

### `_ensure_trajectory_json` — extract to shared function

Currently duplicated between depot and gripper. Extract to `src/utils/pddl_trajectory.py` as a standalone function:

```python
# src/utils/pddl_trajectory.py

def ensure_trajectory_json(images_dir: Path) -> None:
    """If _trajectory.json is missing, convert from .trajectory + problem.pddl."""
    if list(images_dir.glob("*_trajectory.json")):
        return
    trajectory_files = list(images_dir.glob("*.trajectory"))
    problem_files = list(images_dir.glob("*.pddl"))
    if trajectory_files and problem_files:
        convert_trajectory_to_json(trajectory_files[0], problem_files[0])
```

---

## About the JSON → .trajectory Question

The `_trajectory.json` is **not** an unnecessary intermediate. It serves two distinct roles:

1. **Ground-truth reference** — LLM classifiers load it for few-shot examples; domain algorithms load it for GT comparison; GT injection reads it to replace predicted states with GT states.
2. **Domain adaptation layer** — `_manipulate_trajectory_json` transforms PDDLGym's untyped/differently-typed representation into the domain's actual type system before writing the JSON. The `.trajectory` file is then derived from this adapted JSON.

Eliminating the JSON would require either:
- Embedding the type adaptation into the gym environment itself (wrong layer), or
- Doing the adaptation at `.trajectory` serialization time (losing the structured JSON for GT reference)

So the two-file approach is sound. The problem is **how** it's implemented, not **that** it exists.

---

## Migration Plan

### Phase 1: Move universal methods to base class (low risk)
1. Add `domain` attribute to `ImageTrajectoryHandler`
2. Move `create_masking_info` + `create_trajectory_and_masks` to base class
3. Delete these methods from all 7 LLM handlers + 1 benchmark handler
4. Run existing tests

### Phase 2: Create `LLMTrajectoryHandler` intermediate (medium risk)
1. Create `src/trajectory_handlers/llm_trajectory_handler.py`
2. Extract `_ensure_trajectory_json` to `src/utils/pddl_trajectory.py`
3. Re-parent all 7 LLM handlers to extend `LLMTrajectoryHandler`
4. Remove boilerplate `__init__` and `init_visual_components` from each
5. Remove unused `api_key`/`model`/`temperature` constructor params
6. Update all callers that pass these params (simulators, data_generator, benchmark)
7. Run existing tests

### Phase 3: Cleanup (low risk)
1. Fix wrong docstrings
2. Remove dead `_rename_ground_action` identity overrides (hiking)
3. Update `__init__.py` exports

---

## Line Count Impact

| File | Before | After |
|---|---|---|
| `image_trajectory_handler.py` (base) | 355 | ~385 (+masking methods) |
| `llm_trajectory_handler.py` (new) | 0 | ~60 |
| `llm_blocks_trajectory_handler.py` | 148 | ~30 |
| `llm_hanoi_trajectory_handler.py` | 218 | ~110 (keeps complex transforms) |
| `llm_hiking_trajectory_handler.py` | 97 | ~8 |
| `llm_maze_trajectory_handler.py` | 115 | ~20 |
| `llm_npuzzle_trajectory_handler.py` | 224 | ~115 (keeps complex transforms) |
| `llm_depot_trajectory_handler.py` | 143 | ~20 |
| `llm_gripper_trajectory_handler.py` | 135 | ~15 |
| **Total** | **1435** | **~763** |

~47% reduction, almost entirely from removing duplication.
