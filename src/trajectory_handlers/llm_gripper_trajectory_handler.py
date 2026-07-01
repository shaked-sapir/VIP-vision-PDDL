"""LLM-based trajectory handler for the Gripper domain."""

from pathlib import Path

from src.fluent_classification.llm_gripper_fluent_classifier import LLMGripperFluentClassifier
from src.object_detection.llm_gripper_object_detector import LLMGripperObjectDetector
from src.trajectory_handlers.llm_external_trajectory_handler import LLMExternalImageTrajectoryHandler
from src.utils.pddl_trajectory import ensure_trajectory_json


class LLMGripperImageTrajectoryHandler(LLMExternalImageTrajectoryHandler):
    """LLM-based trajectory handler for the Gripper domain."""

    detector_class = LLMGripperObjectDetector
    classifier_class = LLMGripperFluentClassifier

    def _pre_init_hook(self, init_state_image_path: Path) -> None:
        """Auto-convert .trajectory to _trajectory.json if needed."""
        ensure_trajectory_json(init_state_image_path.parent)
