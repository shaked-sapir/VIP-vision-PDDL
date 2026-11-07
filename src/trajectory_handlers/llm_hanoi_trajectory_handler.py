from src.fluent_classification.llm_hanoi_fluent_classifier import LLMHanoiFluentClassifier
from src.object_detection.llm_hanoi_object_detector import LLMHanoiObjectDetector
from src.trajectory_handlers.llm_external_trajectory_handler import LLMExternalImageTrajectoryHandler


class LLMHanoiImageTrajectoryHandler(LLMExternalImageTrajectoryHandler):
    """LLM-based trajectory handler for the Hanoi domain."""

    detector_class = LLMHanoiObjectDetector
    classifier_class = LLMHanoiFluentClassifier
