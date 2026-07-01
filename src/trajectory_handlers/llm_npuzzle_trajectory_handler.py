import re

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
