from src.fluent_classification.llm_maze_fluent_classifier import LLMMazeFluentClassifier
from src.object_detection.llm_maze_object_detector import LLMMazeObjectDetector
from src.trajectory_handlers.llm_image_trajectory_handler import LLMImageTrajectoryHandler


class LLMMazeImageTrajectoryHandler(LLMImageTrajectoryHandler):
    """LLM-based trajectory handler for the Maze domain."""

    detector_class = LLMMazeObjectDetector
    classifier_class = LLMMazeFluentClassifier

    @staticmethod
    def _rename_ground_action(action_str: str) -> str:
        """Replace hyphens with underscores in action names (action naming convention)."""
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
