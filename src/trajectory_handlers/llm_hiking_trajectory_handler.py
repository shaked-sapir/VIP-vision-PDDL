from src.fluent_classification.llm_hiking_fluent_classifier import LLMHikingFluentClassifier
from src.object_detection.llm_hiking_object_detector import LLMHikingObjectDetector
from src.trajectory_handlers.llm_image_trajectory_handler import LLMImageTrajectoryHandler


class LLMHikingImageTrajectoryHandler(LLMImageTrajectoryHandler):
    """LLM-based trajectory handler for the Hiking domain."""

    detector_class = LLMHikingObjectDetector
    classifier_class = LLMHikingFluentClassifier
