from pathlib import Path
from typing import Dict, List

from pddl_plus_parser.lisp_parsers import DomainParser

from src.action_model.gym2SAM_parser import parse_grounded_predicates
from src.fluent_classification.image_llm_backend_factory import ImageLLMBackendFactory
from src.fluent_classification.llm_npuzzle_fluent_classifier import LLMNpuzzleFluentClassifier
from src.object_detection.llm_npuzzle_object_detector import LLMNpuzzleObjectDetector
from src.trajectory_handlers import ImageTrajectoryHandler
from src.utils.masking import save_masking_info


class LLMNpuzzleImageTrajectoryHandler(ImageTrajectoryHandler):
    """
    LLM-based trajectory handler for the Hanoi domain.
    Uses LLMNpuzzleObjectDetector and LLMNpuzzleFluentClassifier.
    """

    def __init__(self,
                 domain_name,
                 pddl_domain_file: Path,
                 api_key: str,
                 vendor: str = "openai",
                 ):
        super().__init__(domain_name=domain_name)
        self.api_key = api_key
        self.vendor = vendor
        self.domain = DomainParser(pddl_domain_file, partial_parsing=True).parse_domain()

    def init_visual_components(self, init_state_image_path: Path) -> None:
        """
        In this class, this method should only be called after initializing a specific
        blocksworld problem, because the object detection module depends on blocksworld colors which
        are determined only at problem initialization time - and they are extracted from the initial state image.
        """

        self.object_detector = LLMNpuzzleObjectDetector(
            llm_backend=ImageLLMBackendFactory.create(
                vendor=self.vendor,
                model_type="object_detection"
            ),
            init_state_image_path=init_state_image_path
        )

        detected_objects_by_type: Dict[str, List[str]] = self.object_detector.detect(str(init_state_image_path))

        self.fluent_classifier = LLMNpuzzleFluentClassifier(
            llm_backend=ImageLLMBackendFactory.create(
                vendor=self.vendor,
                model_type="fluent_classification"
            ),
            type_to_objects=detected_objects_by_type,
            init_state_image_path=init_state_image_path
        )

        print(f"Initialized LLMNpuzzleImageTrajectoryHandler with detected objects: {detected_objects_by_type}")

    @staticmethod
    def _rename_ground_action(action_str: str) -> str:
        """
        in the pddlgym, the "move-X" (tile, X, Y, X\Y new) means to move the BLANK to position X,Y
        by sliding the tile at position X,Y into the blank position.
        :param action_str: action to transform from gym format to our format
        :return:
        """
        # split into: original_name, "(args)"
        gym_action_name, args_part = action_str.split("(", 1)
        args_str = args_part.rstrip(")")

        # extract argument names
        arg_names = [a.split(":", 1)[0].strip() for a in args_str.split(",")]

        # extract coordinates
        tile_raw, gym_from_x_cord, gym_from_y_cord, gym_shift_cord = arg_names

        target_position_from = f"p_{gym_from_x_cord[1]}_{gym_from_y_cord[1]}"
        target_tile = f"{tile_raw[0]}_{tile_raw[1]}"

        if gym_action_name in ["move-down", "move-up"]:
            target_position_to = f"p_{gym_from_x_cord[1]}_{gym_shift_cord[1]}"
        elif gym_action_name in ["move-left", "move-right"]:
            target_position_to = f"p_{gym_shift_cord[1]}_{gym_from_y_cord[1]}"

        return f"move({target_tile}:tile, {target_position_from}:position, {target_position_to}:position)"

    def create_masking_info(self, problem_name: str, imaged_trajectory: list[dict], trajectory_path: Path) -> None:
        trajectory_masking_info = (
                [parse_grounded_predicates(imaged_trajectory[0]['current_state']['unknown'], self.domain)] +
                [parse_grounded_predicates(step['next_state']['unknown'], self.domain)
                 for step in imaged_trajectory]
        )

        # Save to working directory
        save_masking_info(trajectory_path, problem_name, trajectory_masking_info)

    def create_trajectory_and_masks(self, problem_name: str, actions: List[str], images_path: Path) -> List[dict]:
        """
        Creates trajectory and masking info files from images.

        This method:
        1. Initializes visual components (object detection) if not already done
        2. Runs fluent classification on all images
        3. Saves trajectory file (problem_name.trajectory)
        4. Saves masking info file (problem_name.masking_info)

        Returns:
            imaged_trajectory: List of dicts containing predicted states for each step
        """
        imaged_trajectory = super().image_trajectory_pipeline(problem_name, actions, images_path)

        self.create_masking_info(problem_name, imaged_trajectory, images_path)

        return imaged_trajectory

    def _manipulate_trajectory_json(self, gt_trajectory_json: list) -> list:
        """
        Transform npuzzle trajectory from pddlgym format to the typed format.

        Transformations:
        1. at(tT:default,xX:default,yY:default) → at(t_T:tile,p_X_Y:position)
        2. blank(xX:default,yY:default) → empty(p_X_Y:position)
        3. Add neighbor(p_X_Y:position, p_I_J:position) for grid adjacency
        4. move-direction(...) → move(tile, from_pos, to_pos)

        Args:
            gt_trajectory_json: List of trajectory steps in pddlgym format

        Returns:
            Modified trajectory JSON in typed npuzzle format
        """
        import re

        # Extract grid dimensions by finding all x and y coordinates
        all_x_coords = set()
        all_y_coords = set()

        for step in gt_trajectory_json:
            for state_key in ['current_state', 'next_state']:
                if state_key in step and 'objects' in step[state_key]:
                    for obj in step[state_key]['objects']:
                        if obj.startswith('x') and ':default' in obj:
                            x_num = obj.split(':')[0][1:]  # Extract number from 'xN:default'
                            all_x_coords.add(int(x_num))
                        elif obj.startswith('y') and ':default' in obj:
                            y_num = obj.split(':')[0][1:]  # Extract number from 'yN:default'
                            all_y_coords.add(int(y_num))

        max_x = max(all_x_coords) if all_x_coords else 0
        max_y = max(all_y_coords) if all_y_coords else 0

        # Generate all neighbor literals for the grid
        neighbor_literals = []
        for x in range(1, max_x + 1):
            for y in range(1, max_y + 1):
                # Add neighbors in 4 directions if they're within bounds
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = x + dx, y + dy
                    if 1 <= nx <= max_x and 1 <= ny <= max_y:
                        neighbor_literals.append(f"neighbor(p_{x}_{y}:position,p_{nx}_{ny}:position)")

        # Process each step
        for step in gt_trajectory_json:
            # Transform literals in current_state and next_state
            for state_key in ['current_state', 'next_state']:
                if state_key in step and 'literals' in step[state_key]:
                    literals = step[state_key]['literals']
                    new_literals = []

                    for lit in literals:
                        # Transform at(tT:default,xX:default,yY:default) → at(t_T:tile,p_X_Y:position)
                        at_match = re.match(r'at\(t(\d+):default,x(\d+):default,y(\d+):default\)', lit)
                        if at_match:
                            tile_num, x_num, y_num = at_match.groups()
                            new_literals.append(f"at(t_{tile_num}:tile,p_{x_num}_{y_num}:position)")
                            continue

                        # Transform blank(xX:default,yY:default) → empty(p_X_Y:position)
                        blank_match = re.match(r'blank\(x(\d+):default,y(\d+):default\)', lit)
                        if blank_match:
                            x_num, y_num = blank_match.groups()
                            new_literals.append(f"empty(p_{x_num}_{y_num}:position)")
                            continue

                        # Skip tile(tN:default), position(xN:default), position(yN:default), inc, dec predicates
                        if (lit.startswith('tile(') or lit.startswith('position(') or
                            lit.startswith('inc(') or lit.startswith('dec(')):
                            continue

                        # Keep other literals as-is
                        new_literals.append(lit)

                    # Add neighbor literals to the state
                    new_literals.extend(neighbor_literals)
                    step[state_key]['literals'] = new_literals

                # Transform goal literals if present
                if state_key in step and 'goal' in step[state_key]:
                    goal_literals = step[state_key]['goal']
                    new_goal_literals = []

                    for lit in goal_literals:
                        # Transform at(tT:default,xX:default,yY:default) → at(t_T:tile,p_X_Y:position)
                        at_match = re.match(r'at\(t(\d+):default,x(\d+):default,y(\d+):default\)', lit)
                        if at_match:
                            tile_num, x_num, y_num = at_match.groups()
                            new_goal_literals.append(f"at(t_{tile_num}:tile,p_{x_num}_{y_num}:position)")
                            continue

                        # Keep other goal literals as-is
                        new_goal_literals.append(lit)

                    step[state_key]['goal'] = new_goal_literals

            # Transform ground_action using existing method
            if 'ground_action' in step:
                original_action = step['ground_action']
                try:
                    step['ground_action'] = self._rename_ground_action(original_action)
                except Exception as e:
                    print(f"Warning: Failed to transform action '{original_action}': {e}")

        return gt_trajectory_json
