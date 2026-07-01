"""LLM + PDDLGym trajectory handler — combines LLM visual components with gym data generation."""

from src.trajectory_handlers.llm_visual_components_mixin import LLMVisualComponentsMixin
from src.trajectory_handlers.pddlgym_trajectory_handler import PDDLGymImageTrajectoryHandler


class LLMImageTrajectoryHandler(LLMVisualComponentsMixin, PDDLGymImageTrajectoryHandler):
    """LLM-based trajectory handler backed by PDDLGym.

    Subclasses set detector_class and classifier_class, and optionally override
    _rename_ground_action, _manipulate_trajectory_json, or _pre_init_hook.
    """
    pass
