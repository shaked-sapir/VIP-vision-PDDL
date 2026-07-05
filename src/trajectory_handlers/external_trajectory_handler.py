"""External-source trajectory handler — for domains whose images come from outside PDDLGym."""

from pathlib import Path
from typing import List

from src.trajectory_handlers.image_trajectory_handler import ImageTrajectoryHandler
from src.utils.pddl_trajectory import ensure_trajectory_json, extract_actions_from_trajectory_json


class ExternalImageTrajectoryHandler(ImageTrajectoryHandler):
    """Trajectory handler for externally-sourced image sequences.

    Expects a directory containing:
        - state_*.png images (any padding: state_0.png or state_0000.png).
        - A .trajectory file (GT trajectory in PDDL format).
        - A problem .pddl file (for object typing).

    Pipeline flow (run_pipeline):
        1. Convert .trajectory + .pddl → _trajectory.json if not already present.
        2. Extract ground action strings from the trajectory JSON.
        3. Auto-detect image filename padding from existing files.
        4. Run visual inference on the images (construct_trajectory_from_images).
        5. Write inferred .trajectory + .masking_info files.

    To add a new externally-sourced domain:
        - For LLM-based detection: subclass LLMExternalImageTrajectoryHandler (which
          combines LLMVisualComponentsMixin with this class) and set
          detector_class + classifier_class.
        - Override _pre_init_hook if the domain needs preparation before visual
          component init (e.g. ensuring trajectory JSON exists for few-shot examples).
        - Override _rename_ground_action if action names in the trajectory differ
          from your domain's expected format.
    """

    def run_pipeline(self, problem_name: str, images_path: Path, **kwargs) -> List[dict]:
        """Run the full external-source pipeline.

        Converts GT files if needed, extracts actions, detects image format,
        then runs inference to produce .trajectory + .masking_info.

        Args:
            problem_name: PDDL problem identifier (e.g. "problem0").
            images_path: Directory containing the pre-existing images and GT files.
            **kwargs: Reserved for future use.

        Returns:
            The inferred trajectory as a list of step dicts.
        """
        ensure_trajectory_json(images_path)
        actions = extract_actions_from_trajectory_json(images_path)
        self._set_seq_idx_format(images_path)
        return self.create_trajectory_and_masks(problem_name, actions, images_path)
