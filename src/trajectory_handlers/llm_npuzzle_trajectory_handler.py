from src.fluent_classification.llm_npuzzle_fluent_classifier import LLMNpuzzleFluentClassifier
from src.object_detection.llm_npuzzle_object_detector import LLMNpuzzleObjectDetector
from src.trajectory_handlers.llm_external_trajectory_handler import LLMExternalImageTrajectoryHandler


class LLMNpuzzleImageTrajectoryHandler(LLMExternalImageTrajectoryHandler):
    """LLM-based trajectory handler for the N-Puzzle domain."""

    detector_class = LLMNpuzzleObjectDetector
    classifier_class = LLMNpuzzleFluentClassifier
