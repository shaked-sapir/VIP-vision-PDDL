"""LLM + External-source trajectory handler — combines LLM visual components with external data."""

from src.trajectory_handlers.external_trajectory_handler import ExternalImageTrajectoryHandler
from src.trajectory_handlers.llm_visual_components_mixin import LLMVisualComponentsMixin


class LLMExternalImageTrajectoryHandler(LLMVisualComponentsMixin, ExternalImageTrajectoryHandler):
    """LLM-based trajectory handler for externally-sourced image sequences.

    Subclasses set detector_class and classifier_class, and optionally override
    _rename_ground_action or _pre_init_hook.
    """
    pass
