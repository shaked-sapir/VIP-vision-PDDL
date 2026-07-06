import re
from pathlib import Path

from src.fluent_classification.llm_npuzzle_fluent_classifier import LLMNpuzzleFluentClassifier
from src.object_detection.llm_npuzzle_object_detector import LLMNpuzzleObjectDetector
from src.trajectory_handlers.llm_image_trajectory_handler import LLMImageTrajectoryHandler

# Domain name of the typed eval schema (matches n_puzzle.pddl / eight01x_eval.pddl).
_TYPED_DOMAIN_NAME = "n_puzzle_typed"


class LLMNpuzzleImageTrajectoryHandler(LLMImageTrajectoryHandler):
    """LLM-based trajectory handler for the N-Puzzle domain."""

    detector_class = LLMNpuzzleObjectDetector
    classifier_class = LLMNpuzzleFluentClassifier

    def translate_problem_pddl(self, pddl_path: Path) -> None:
        """Rewrite a generated slidetile problem .pddl into the typed n_puzzle schema.

        The raw problem uses the untyped `strips-sliding-tile` schema (separate
        `x*`/`y*` position objects, `at(t x y)`, `blank`, plus `tile`/`position`/
        `inc`/`dec` axioms). The eval schema `n_puzzle_typed` merges coordinates
        into `p_X_Y` position objects, uses `at(t_T p_X_Y)` / `empty(p_X_Y)`, and
        needs explicit `neighbor` literals. Object identities change, so this is a
        rebuild (not a per-literal rewrite): translate init + goal, derive objects
        and neighbors, and write a fresh typed problem file — preserving this
        problem's own init and goal.
        """
        text = pddl_path.read_text()
        problem_name = self._parse_problem_name(text)
        init_literals = self._translate_init_block(self._extract_block(text, "init"))
        goal_literals = self._translate_goal_literals(self._extract_block(text, "goal"))

        tiles, positions = self._collect_objects(init_literals + goal_literals)
        pddl_path.write_text(self._render_typed_problem(
            problem_name, tiles, positions, init_literals, goal_literals))

    # ── Problem-file translation helpers ─────────────────────────────────

    @staticmethod
    def _parse_problem_name(text: str) -> str:
        """Extract the problem name from a `(define (problem NAME) ...)` header."""
        match = re.search(r'\(problem\s+([\w-]+)\)', text)
        return match.group(1) if match else "problem"

    @staticmethod
    def _extract_block(text: str, keyword: str) -> str:
        """Return the raw body text of the (:init ...) or (:goal ...) block."""
        if keyword == "init":
            match = re.search(r'\(:init\s+(.*?)\)\s*\(:goal', text, re.DOTALL)
        else:
            match = re.search(r'\(:goal\s+(.*)\)\s*\)\s*$', text, re.DOTALL)
        return match.group(1) if match else ""

    @staticmethod
    def _translate_init_block(init_body: str) -> list[str]:
        """Translate init literals `(at t x y)` / `(blank x y)` to the typed schema.

        Reuses the slidetile→typed literal mapping (drops tile/position/inc/dec,
        maps at/blank, derives neighbors) shared with the fluent classifier.
        """
        # Convert `(at t6 x1 y3)` → `at(t6:default,x1:default,y3:default)` so the
        # classifier's shared translator (which expects that literal form) applies.
        raw_literals = []
        for pred_match in re.finditer(r'\(([a-zA-Z][\w-]*(?:\s+\w+)*)\)', init_body):
            parts = pred_match.group(1).split()
            name, args = parts[0], parts[1:]
            if not args:
                raw_literals.append(f"{name}()")
            else:
                raw_literals.append(f"{name}({','.join(f'{a}:default' for a in args)})")

        typed = LLMNpuzzleFluentClassifier._translate_slidetile_literals_to_typed(raw_literals)
        # Strip the `:type` suffixes → bare PDDL literals `at t_6 p_1_3`.
        return [LLMNpuzzleImageTrajectoryHandler._strip_types(lit) for lit in typed]

    @staticmethod
    def _translate_goal_literals(goal_body: str) -> list[str]:
        """Translate goal `(at t x y)` literals to typed `at t_T p_X_Y`."""
        translated = []
        for at_match in re.finditer(r'\(at\s+t(\d+)\s+x(\d+)\s+y(\d+)\)', goal_body):
            t, x, y = at_match.groups()
            translated.append(f"at t_{t} p_{x}_{y}")
        return translated

    @staticmethod
    def _strip_types(literal: str) -> str:
        """`at(t_6:tile,p_1_3:position)` → `at t_6 p_1_3`."""
        match = re.match(r'([a-zA-Z][\w-]*)\((.*)\)', literal)
        if not match:
            return literal
        pred, args_str = match.groups()
        args = [a.split(':')[0].strip() for a in args_str.split(',') if a.strip()]
        return " ".join([pred, *args])

    @staticmethod
    def _collect_objects(literals: list[str]) -> tuple[list[str], list[str]]:
        """Collect the tile (`t_*`) and position (`p_*`) objects appearing in literals."""
        tiles, positions = set(), set()
        for lit in literals:
            for token in lit.split()[1:]:
                if token.startswith("t_"):
                    tiles.add(token)
                elif token.startswith("p_"):
                    positions.add(token)
        return sorted(tiles), sorted(positions)

    @staticmethod
    def _render_typed_problem(problem_name: str, tiles: list[str], positions: list[str],
                              init_literals: list[str], goal_literals: list[str]) -> str:
        """Render a typed n_puzzle problem file.

        Each `- <type>` group is written on its own line so downstream object
        parsers (e.g. trajectory_json_converter._parse_problem_objects, which
        matches one type group per line) pick up both positions and tiles.
        """
        objects = (f"\n\t{' '.join(positions)} - position"
                   f"\n\t{' '.join(tiles)} - tile\n")
        init_block = "\n    ".join(f"({lit})" for lit in init_literals)
        goal_block = " ".join(f"({lit})" for lit in goal_literals)
        return (
            f"(define (problem {problem_name})\n"
            f"  (:domain {_TYPED_DOMAIN_NAME})\n"
            f"  (:objects {objects})\n"
            f"  (:init\n    {init_block}\n    )\n"
            f"  (:goal (and {goal_block}))\n"
            f"  )\n"
        )

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
