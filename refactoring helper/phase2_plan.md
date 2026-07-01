# Phase 2 Execution Plan — Create LLMImageTrajectoryHandler

## Prerequisites
- Phase 1 must be merged into your current branch first
- Branch: create `refactor/phase2-llm-base-class` from current HEAD

## Overview
Create `LLMImageTrajectoryHandler` intermediate base class that absorbs the shared `__init__` and `init_visual_components` boilerplate from all 7 LLM handlers. Also extract `_ensure_trajectory_json` to utils.

**Out of scope**: The benchmark handler (`benchmark/domains/blocksworld/amlgym_llm_trajectory_handler.py`) uses an older API pattern (passes api_key directly to detectors, not through the factory). Do NOT re-parent it — it stays inheriting from `ImageTrajectoryHandler` directly.

---

## Step 1: Extract `_ensure_trajectory_json` to utils

**File**: `src/utils/pddl_trajectory.py`

Add this function at the end of the file (before the ground-truth injection section, around line 80):

```python
def ensure_trajectory_json(images_dir: Path) -> None:
    """If _trajectory.json is missing, convert from .trajectory + problem.pddl."""
    from src.utils.trajectory_json_converter import convert_trajectory_to_json

    if list(images_dir.glob("*_trajectory.json")):
        return
    trajectory_files = list(images_dir.glob("*.trajectory"))
    problem_files = list(images_dir.glob("*.pddl"))
    if trajectory_files and problem_files:
        convert_trajectory_to_json(
            trajectory_path=trajectory_files[0],
            problem_path=problem_files[0],
        )
```

---

## Step 2: Create `src/trajectory_handlers/llm_image_trajectory_handler.py`

Create this new file:

```python
"""Base class for all LLM-based trajectory handlers.

Subclasses set `detector_class` and `classifier_class` as class attributes,
and optionally override `_rename_ground_action`, `_manipulate_trajectory_json`,
or `_pre_init_hook`.
"""

from pathlib import Path
from typing import Dict, List, Type

from pddl_plus_parser.lisp_parsers import DomainParser

from src.fluent_classification.base_fluent_classifier import FluentClassifier
from src.fluent_classification.image_llm_backend_factory import ImageLLMBackendFactory
from src.object_detection.base_object_detector import ObjectDetector
from src.trajectory_handlers.image_trajectory_handler import ImageTrajectoryHandler


class LLMImageTrajectoryHandler(ImageTrajectoryHandler):
    """Base for all LLM-based trajectory handlers.

    Subclasses must set:
        detector_class: The LLM object detector class for the domain.
        classifier_class: The LLM fluent classifier class for the domain.

    Optionally override:
        _rename_ground_action() — transform action names from gym to domain format.
        _manipulate_trajectory_json() — transform GT trajectory JSON before writing.
        _pre_init_hook() — run before visual component init (e.g. ensure trajectory JSON).
    """

    detector_class: Type[ObjectDetector]
    classifier_class: Type[FluentClassifier]

    def __init__(self, domain_name: str, pddl_domain_file: Path,
                 vendor: str = "openai", **kwargs):
        super().__init__(domain_name=domain_name)
        self.vendor = vendor
        self.domain = DomainParser(pddl_domain_file, partial_parsing=True).parse_domain()

    def _pre_init_hook(self, init_state_image_path: Path) -> None:
        """Override point for work before visual component init."""
        pass

    def init_visual_components(self, init_state_image_path: Path) -> None:
        """Standard LLM init: run pre-hook, detect objects, create classifier."""
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
```

---

## Step 3: Rewrite each domain handler

### 3a. `src/trajectory_handlers/llm_blocks_trajectory_handler.py`

Replace entire file with:

```python
from pathlib import Path

from src.fluent_classification.llm_blocks_fluent_classifier import LLMBlocksFluentClassifier
from src.object_detection.llm_blocks_object_detector import LLMBlocksObjectDetector
from src.trajectory_handlers.llm_image_trajectory_handler import LLMImageTrajectoryHandler


class LLMBlocksImageTrajectoryHandler(LLMImageTrajectoryHandler):
    """LLM-based trajectory handler for the Blocksworld domain."""

    detector_class = LLMBlocksObjectDetector
    classifier_class = LLMBlocksFluentClassifier

    @staticmethod
    def _rename_ground_action(action_str: str) -> str:
        """Rename gym-format actions: pick-up→pick_up, put-down→put_down, remove robot param."""
        return (action_str.replace('pick-up', 'pick_up')
                .replace('put-down', 'put_down')
                .replace(', robot:robot', ''))

    def _manipulate_trajectory_json(self, gt_trajectory_json: list) -> list:
        """Apply blocksworld-specific transformations to trajectory JSON."""
        import re

        for step in gt_trajectory_json:
            for state_key in ['current_state', 'next_state']:
                if state_key in step and 'literals' in step[state_key]:
                    literals = step[state_key]['literals']
                    new_literals = []
                    for lit in literals:
                        if lit == "handempty(robot:robot)":
                            new_literals.append("handempty()")
                        elif lit == "handfull(robot:robot)":
                            continue
                        else:
                            new_literals.append(lit)
                    step[state_key]['literals'] = new_literals

            if 'ground_action' in step:
                action = step['ground_action']
                action = re.sub(r'pick-up\(([^,]+):block,\s*robot:robot\)', r'pick_up(\1:block)', action)
                action = re.sub(r'put-down\(([^,]+):block,\s*robot:robot\)', r'put_down(\1:block)', action)
                action = re.sub(r'stack\(([^,]+):block,\s*([^,]+):block,\s*robot:robot\)', r'stack(\1:block, \2:block)', action)
                action = re.sub(r'unstack\(([^,]+):block,\s*([^,]+):block,\s*robot:robot\)', r'unstack(\1:block, \2:block)', action)
                step['ground_action'] = action

        return gt_trajectory_json
```

### 3b. `src/trajectory_handlers/llm_hiking_trajectory_handler.py`

Replace entire file with:

```python
from src.fluent_classification.llm_hiking_fluent_classifier import LLMHikingFluentClassifier
from src.object_detection.llm_hiking_object_detector import LLMHikingObjectDetector
from src.trajectory_handlers.llm_image_trajectory_handler import LLMImageTrajectoryHandler


class LLMHikingImageTrajectoryHandler(LLMImageTrajectoryHandler):
    """LLM-based trajectory handler for the Hiking domain."""

    detector_class = LLMHikingObjectDetector
    classifier_class = LLMHikingFluentClassifier
```

### 3c. `src/trajectory_handlers/llm_maze_trajectory_handler.py`

Replace entire file with:

```python
from src.fluent_classification.llm_maze_fluent_classifier import LLMMazeFluentClassifier
from src.object_detection.llm_maze_object_detector import LLMMazeObjectDetector
from src.trajectory_handlers.llm_image_trajectory_handler import LLMImageTrajectoryHandler


class LLMMazeImageTrajectoryHandler(LLMImageTrajectoryHandler):
    """LLM-based trajectory handler for the Maze domain."""

    detector_class = LLMMazeObjectDetector
    classifier_class = LLMMazeFluentClassifier

    @staticmethod
    def _rename_ground_action(action_str: str) -> str:
        """Replace hyphens with underscores in action names (amlgym compatibility)."""
        return action_str.replace('move-', 'move_')

    def _manipulate_trajectory_json(self, gt_trajectory_json: list) -> list:
        """Replace hyphens with underscores in action names only (not in parameters)."""
        for step in gt_trajectory_json:
            if 'ground_action' in step and step['ground_action']:
                action = step['ground_action']
                paren_idx = action.find('(')
                if paren_idx > 0:
                    step['ground_action'] = action[:paren_idx].replace('-', '_') + action[paren_idx:]
                else:
                    step['ground_action'] = action.replace('-', '_')
        return gt_trajectory_json
```

### 3d. `src/trajectory_handlers/llm_depot_trajectory_handler.py`

Replace entire file with:

```python
from pathlib import Path

from src.fluent_classification.llm_depot_fluent_classifier import LLMDepotFluentClassifier
from src.object_detection.llm_depot_object_detector import LLMDepotObjectDetector
from src.trajectory_handlers.llm_image_trajectory_handler import LLMImageTrajectoryHandler
from src.utils.pddl_trajectory import ensure_trajectory_json


class LLMDepotImageTrajectoryHandler(LLMImageTrajectoryHandler):
    """LLM-based trajectory handler for the Depot domain."""

    detector_class = LLMDepotObjectDetector
    classifier_class = LLMDepotFluentClassifier

    def get_image_path_by_index(self, image_dir: Path, image_sequential_index: int) -> str:
        """Use unpadded image names (state_0.png instead of state_0000.png)."""
        return self.get_image_full_path(image_dir, f"state_{image_sequential_index}.png")

    def _pre_init_hook(self, init_state_image_path: Path) -> None:
        """Auto-convert .trajectory to _trajectory.json if needed."""
        ensure_trajectory_json(init_state_image_path.parent)
```

### 3e. `src/trajectory_handlers/llm_gripper_trajectory_handler.py`

Replace entire file with:

```python
from pathlib import Path

from src.fluent_classification.llm_gripper_fluent_classifier import LLMGripperFluentClassifier
from src.object_detection.llm_gripper_object_detector import LLMGripperObjectDetector
from src.trajectory_handlers.llm_image_trajectory_handler import LLMImageTrajectoryHandler
from src.utils.pddl_trajectory import ensure_trajectory_json


class LLMGripperImageTrajectoryHandler(LLMImageTrajectoryHandler):
    """LLM-based trajectory handler for the Gripper domain."""

    detector_class = LLMGripperObjectDetector
    classifier_class = LLMGripperFluentClassifier

    def _pre_init_hook(self, init_state_image_path: Path) -> None:
        """Auto-convert .trajectory to _trajectory.json if needed."""
        ensure_trajectory_json(init_state_image_path.parent)
```

### 3f. `src/trajectory_handlers/llm_hanoi_trajectory_handler.py`

Replace entire file with:

```python
import re
from pathlib import Path

from src.fluent_classification.llm_hanoi_fluent_classifier import LLMHanoiFluentClassifier
from src.object_detection.llm_hanoi_object_detector import LLMHanoiObjectDetector
from src.trajectory_handlers.llm_image_trajectory_handler import LLMImageTrajectoryHandler


class LLMHanoiImageTrajectoryHandler(LLMImageTrajectoryHandler):
    """LLM-based trajectory handler for the Hanoi domain."""

    detector_class = LLMHanoiObjectDetector
    classifier_class = LLMHanoiFluentClassifier

    @staticmethod
    def _rename_ground_action(action_str: str) -> str:
        """Rename move(...) to move_peg_peg/move_disc_peg/etc based on argument types."""
        name_end = action_str.index('(')
        args_str = action_str[name_end:]
        args = args_str[1:-1].split(',')
        names = [a.split(':')[0].strip() for a in args]
        c2 = "peg" if names[1].startswith("peg") else "disc"
        c3 = "peg" if names[2].startswith("peg") else "disc"
        return f"move_{c2}_{c3}{args_str}"

    def _manipulate_trajectory_json(self, gt_trajectory_json: list) -> list:
        """Transform hanoi trajectory from pddlgym untyped format to typed format."""

        def transform_object_type(obj: str) -> str:
            if ':default' not in obj:
                return obj
            name = obj.split(':')[0]
            if name.startswith('peg'):
                return f"{name}:peg"
            elif name.startswith('d'):
                return f"{name}:disc"
            return obj

        def contains_peg(literal: str) -> bool:
            match = re.match(r'\w+\((.*)\)', literal)
            if match:
                args = [arg.split(':')[0].strip() for arg in match.group(1).split(',')]
                return any(arg.startswith('peg') for arg in args)
            return False

        def transform_literal(lit: str) -> str:
            if lit.startswith('smaller('):
                suffix = 'peg' if contains_peg(lit) else 'disc'
                lit = lit.replace('smaller(', f'smaller-{suffix}(')
            elif lit.startswith('on('):
                suffix = 'peg' if contains_peg(lit) else 'disc'
                lit = lit.replace('on(', f'on-{suffix}(')
            elif lit.startswith('clear('):
                suffix = 'peg' if contains_peg(lit) else 'disc'
                lit = lit.replace('clear(', f'clear-{suffix}(')
            lit = re.sub(r'(peg\d+):default', r'\1:peg', lit)
            lit = re.sub(r'(d\d+):default', r'\1:disc', lit)
            return lit

        for step in gt_trajectory_json:
            for state_key in ['current_state', 'next_state']:
                if state_key in step and 'literals' in step[state_key]:
                    step[state_key]['literals'] = [transform_literal(lit)
                                                   for lit in step[state_key]['literals']]
                if state_key in step and 'objects' in step[state_key]:
                    step[state_key]['objects'] = [transform_object_type(obj)
                                                  for obj in step[state_key]['objects']]
                if state_key in step and 'goal' in step[state_key]:
                    step[state_key]['goal'] = [transform_literal(lit)
                                               for lit in step[state_key]['goal']]

            if 'ground_action' in step:
                try:
                    action = self._rename_ground_action(step['ground_action'])
                    action = re.sub(r'(peg\d+):default', r'\1:peg', action)
                    action = re.sub(r'(d\d+):default', r'\1:disc', action)
                    step['ground_action'] = action
                except Exception as e:
                    print(f"Warning: Failed to transform action '{step['ground_action']}': {e}")

        return gt_trajectory_json
```

### 3g. `src/trajectory_handlers/llm_npuzzle_trajectory_handler.py`

Replace entire file with:

```python
import re
from pathlib import Path

from src.fluent_classification.llm_npuzzle_fluent_classifier import LLMNpuzzleFluentClassifier
from src.object_detection.llm_npuzzle_object_detector import LLMNpuzzleObjectDetector
from src.trajectory_handlers.llm_image_trajectory_handler import LLMImageTrajectoryHandler


class LLMNpuzzleImageTrajectoryHandler(LLMImageTrajectoryHandler):
    """LLM-based trajectory handler for the N-Puzzle domain."""

    detector_class = LLMNpuzzleObjectDetector
    classifier_class = LLMNpuzzleFluentClassifier

    @staticmethod
    def _rename_ground_action(action_str: str) -> str:
        """Transform move-direction(tile, X, Y, shift) to move(t_T:tile, p_X_Y:position, p_I_J:position)."""
        gym_action_name, args_part = action_str.split("(", 1)
        args_str = args_part.rstrip(")")
        arg_names = [a.split(":", 1)[0].strip() for a in args_str.split(",")]
        tile_raw, gym_from_x_cord, gym_from_y_cord, gym_shift_cord = arg_names

        target_position_from = f"p_{gym_from_x_cord[1]}_{gym_from_y_cord[1]}"
        target_tile = f"{tile_raw[0]}_{tile_raw[1]}"

        if gym_action_name in ["move-down", "move-up"]:
            target_position_to = f"p_{gym_from_x_cord[1]}_{gym_shift_cord[1]}"
        elif gym_action_name in ["move-left", "move-right"]:
            target_position_to = f"p_{gym_shift_cord[1]}_{gym_from_y_cord[1]}"

        return f"move({target_tile}:tile, {target_position_from}:position, {target_position_to}:position)"

    def _manipulate_trajectory_json(self, gt_trajectory_json: list) -> list:
        """Transform npuzzle trajectory from pddlgym untyped format to typed format."""
        all_x_coords = set()
        all_y_coords = set()

        for step in gt_trajectory_json:
            for state_key in ['current_state', 'next_state']:
                if state_key in step and 'objects' in step[state_key]:
                    for obj in step[state_key]['objects']:
                        if obj.startswith('x') and ':default' in obj:
                            all_x_coords.add(int(obj.split(':')[0][1:]))
                        elif obj.startswith('y') and ':default' in obj:
                            all_y_coords.add(int(obj.split(':')[0][1:]))

        max_x = max(all_x_coords) if all_x_coords else 0
        max_y = max(all_y_coords) if all_y_coords else 0

        neighbor_literals = []
        for x in range(1, max_x + 1):
            for y in range(1, max_y + 1):
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = x + dx, y + dy
                    if 1 <= nx <= max_x and 1 <= ny <= max_y:
                        neighbor_literals.append(f"neighbor(p_{x}_{y}:position,p_{nx}_{ny}:position)")

        for step in gt_trajectory_json:
            for state_key in ['current_state', 'next_state']:
                if state_key in step and 'literals' in step[state_key]:
                    new_literals = []
                    for lit in step[state_key]['literals']:
                        at_match = re.match(r'at\(t(\d+):default,x(\d+):default,y(\d+):default\)', lit)
                        if at_match:
                            t, x, y = at_match.groups()
                            new_literals.append(f"at(t_{t}:tile,p_{x}_{y}:position)")
                            continue
                        blank_match = re.match(r'blank\(x(\d+):default,y(\d+):default\)', lit)
                        if blank_match:
                            x, y = blank_match.groups()
                            new_literals.append(f"empty(p_{x}_{y}:position)")
                            continue
                        if (lit.startswith('tile(') or lit.startswith('position(') or
                            lit.startswith('inc(') or lit.startswith('dec(')):
                            continue
                        new_literals.append(lit)
                    new_literals.extend(neighbor_literals)
                    step[state_key]['literals'] = new_literals

                if state_key in step and 'goal' in step[state_key]:
                    new_goal = []
                    for lit in step[state_key]['goal']:
                        at_match = re.match(r'at\(t(\d+):default,x(\d+):default,y(\d+):default\)', lit)
                        if at_match:
                            t, x, y = at_match.groups()
                            new_goal.append(f"at(t_{t}:tile,p_{x}_{y}:position)")
                            continue
                        new_goal.append(lit)
                    step[state_key]['goal'] = new_goal

            if 'ground_action' in step:
                try:
                    step['ground_action'] = self._rename_ground_action(step['ground_action'])
                except Exception as e:
                    print(f"Warning: Failed to transform action '{step['ground_action']}': {e}")

        return gt_trajectory_json
```

---

## Step 4: Update callers that pass dead constructor params

These callers pass `api_key`, `object_detector_model`, `object_detection_temperature`, `fluent_classifier_model`, `fluent_classification_temperature` — all of which are now absorbed by the factory via config.yaml. The new `__init__` accepts `**kwargs` so these won't break immediately, but they should be cleaned up.

### `src/simulator.py` lines 69-76 and 159-166
Change from:
```python
LLMBlocksImageTrajectoryHandler(
    domain_name,
    openai_apikey,
    object_detector_model=...,
    ...
)
```
To:
```python
LLMBlocksImageTrajectoryHandler(
    domain_name=domain_name,
    pddl_domain_file=pddl_domain_file,
)
```

**NOTE**: These callers currently pass `openai_apikey` as the 2nd positional arg where the constructor expects `pddl_domain_file`. This is a pre-existing bug. Fix it by passing `pddl_domain_file` (which already exists in scope in both callsites).

### `src/simulator_cli.py` lines 227 and 232
Same issue — passes `(domain_name, openai_apikey)`. Change to:
```python
LLMBlocksImageTrajectoryHandler(
    domain_name=domain_name,
    pddl_domain_file=pddl_domain_file,
)
```
You'll need to check that `pddl_domain_file` is in scope at line 227. It's constructed from config earlier in the function.

### `src/simulator_cli.py` line 283
Same fix.

### `src/lab_simulator.py` lines 585-611
All three handler constructions (blocks, hanoi, npuzzle). Change each to:
```python
trajectory_handler = LLMXxxImageTrajectoryHandler(
    domain_name=gym_domain_name,
    pddl_domain_file=pddl_domain_file,
)
```
`pddl_domain_file` is already in scope (loaded from config).

### `benchmark/data_generator.py` lines 152-157
Already uses kwargs `domain_name=`, `pddl_domain_file=`, `api_key=`, `vendor=`. After this change, `api_key` becomes a dead kwarg caught by `**kwargs`. Remove it:
```python
trajectory_handler = trajectory_handler_class(
    domain_name=gym_domain_name,
    pddl_domain_file=benchmark_domain_path,
    vendor=vendor,
)
```

### `src/test_npuzzle_object_detector.py` lines 72-80
Same pattern — remove dead params, keep `domain_name` and `pddl_domain_file`.

### `test_maze_llm_classification.py` lines 71-79
Same fix.

---

## Step 5: Update `__init__.py`

In `src/trajectory_handlers/__init__.py`, change:
```python
from .image_trajectory_handler import *
```
to:
```python
from .image_trajectory_handler import ImageTrajectoryHandler
from .llm_image_trajectory_handler import LLMImageTrajectoryHandler
```

---

## Step 6: Commit

```bash
git add -A
git commit -m "refactor: create LLMImageTrajectoryHandler base, collapse domain handler boilerplate"
```

---

## Verification Checklist

1. `grep -r "def __init__" src/trajectory_handlers/llm_*.py` — should only appear in `llm_image_trajectory_handler.py`
2. `grep -r "def init_visual_components" src/trajectory_handlers/llm_*.py` — should only appear in `llm_image_trajectory_handler.py`
3. `grep -r "ImageLLMBackendFactory" src/trajectory_handlers/llm_*.py` — should only appear in `llm_image_trajectory_handler.py`
4. `grep -r "class LLM.*ImageTrajectoryHandler" src/trajectory_handlers/` — all should inherit from `LLMImageTrajectoryHandler` (except the base itself)
5. `grep -rn "_ensure_trajectory_json" src/trajectory_handlers/` — should NOT appear (moved to utils)
6. `grep -rn "ensure_trajectory_json" src/utils/pddl_trajectory.py` — should exist
7. `grep -rn "openai_apikey\|api_key\|object_detector_model\|fluent_classifier_model" src/trajectory_handlers/llm_*.py` — should return nothing
