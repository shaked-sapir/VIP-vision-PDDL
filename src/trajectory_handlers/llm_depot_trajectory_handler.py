"""LLM-based trajectory handler for the Depot domain."""

from pathlib import Path

from src.fluent_classification.llm_depot_fluent_classifier import LLMDepotFluentClassifier
from src.object_detection.llm_depot_object_detector import LLMDepotObjectDetector
from src.trajectory_handlers.llm_external_trajectory_handler import LLMExternalImageTrajectoryHandler
from src.utils.pddl_trajectory import ensure_trajectory_json


class LLMDepotImageTrajectoryHandler(LLMExternalImageTrajectoryHandler):
    """LLM-based trajectory handler for the Depot domain."""

    detector_class = LLMDepotObjectDetector
    classifier_class = LLMDepotFluentClassifier

    def _pre_init_hook(self, init_state_image_path: Path) -> None:
        """Auto-convert .trajectory to _trajectory.json if needed."""
        ensure_trajectory_json(init_state_image_path.parent)
