"""LLM-based object detector for the Gripper domain."""

import re
from pathlib import Path
from typing import Dict, List, Union

from src.fluent_classification.image_llm_backend_protocol import ImageLLMBackend
from src.llms.domains.gripper.prompts import object_detection_system_prompt
from src.object_detection.llm_object_detector import LLMObjectDetector


class LLMGripperObjectDetector(LLMObjectDetector):
    """LLM-based object detector for the Gripper domain.

    The LLM returns objects as "rooma:room", "ball1:ball", "left:gripper", etc.

    Name mappings from image labels to PDDL names:
    - Room labels: "ROOMA" → "rooma", "ROOMB" → "roomb" (lowercase)
    - Ball numbers: "1" → "ball1", "2" → "ball2" (prefixed)
    - Gripper labels: "L" → "left", "R" → "right"

    The detect() method normalizes room names to lowercase after detection.
    """

    # Map image gripper labels to PDDL names.
    GRIPPER_NAME_MAP = {"r": "right", "l": "left"}

    def __init__(
        self,
        llm_backend: ImageLLMBackend,
        init_state_image_path: Path,
        temperature: float = None,
        inference_mode: bool = False,
    ):
        super().__init__(
            llm_backend=llm_backend,
            init_state_image_path=init_state_image_path,
            temperature=temperature,
            inference_mode=inference_mode,
        )

        # Identity mapping — gripper prompt already instructs the LLM
        # to output PDDL-style names (rooma, ball1, left, right).
        self.imaged_obj_to_gym_obj_name = {}

        gym_objects = self.extract_objects_from_gt_state()
        self.fewshot_examples = [(init_state_image_path, gym_objects)]

    def _get_system_prompt(self) -> str:
        """Returns the system prompt for Gripper object detection."""
        return object_detection_system_prompt

    @staticmethod
    def _get_result_regex() -> str:
        """Returns regex for gripper object detection output.

        Gripper names include lowercase letters and digits (rooma, ball1, left, right).
        """
        return r"\b[A-Za-z]+\d*:[a-z]+\b"

    def detect(self, image: Union[Path, str], *args, **kwargs) -> Dict[str, List[str]]:
        """Detect objects and normalize names.

        - Room names: ROOMA → rooma (lowercase)
        - Gripper names should already be 'left'/'right' from the prompt,
          but normalize R/L → right/left as fallback.
        """
        detected_objects = super().detect(image, *args, **kwargs)

        # Normalize room names to lowercase
        if "room" in detected_objects:
            detected_objects["room"] = [
                name.lower() for name in detected_objects["room"]
            ]

        # Normalize gripper names (R→right, L→left) as fallback
        if "gripper" in detected_objects:
            detected_objects["gripper"] = [
                self.GRIPPER_NAME_MAP.get(name.lower(), name.lower())
                for name in detected_objects["gripper"]
            ]

        return detected_objects
