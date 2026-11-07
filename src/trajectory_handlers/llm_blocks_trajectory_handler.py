from src.fluent_classification.llm_blocks_fluent_classifier import LLMBlocksFluentClassifier
from src.object_detection.llm_blocks_object_detector import LLMBlocksObjectDetector
from src.trajectory_handlers.llm_external_trajectory_handler import LLMExternalImageTrajectoryHandler


class LLMBlocksImageTrajectoryHandler(LLMExternalImageTrajectoryHandler):
    """LLM-based trajectory handler for the Blocksworld domain."""

    detector_class = LLMBlocksObjectDetector
    classifier_class = LLMBlocksFluentClassifier
