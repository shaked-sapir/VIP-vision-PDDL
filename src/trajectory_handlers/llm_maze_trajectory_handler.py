from src.fluent_classification.llm_maze_fluent_classifier import LLMMazeFluentClassifier
from src.object_detection.llm_maze_object_detector import LLMMazeObjectDetector
from src.trajectory_handlers.llm_external_trajectory_handler import LLMExternalImageTrajectoryHandler


class LLMMazeImageTrajectoryHandler(LLMExternalImageTrajectoryHandler):
    """LLM-based trajectory handler for the Maze domain."""

    detector_class = LLMMazeObjectDetector
    classifier_class = LLMMazeFluentClassifier

    @staticmethod
    def _rename_ground_action(action_str: str) -> str:
        """Replace hyphens with underscores in action names (action naming convention)."""
        return action_str.replace('move-', 'move_')
