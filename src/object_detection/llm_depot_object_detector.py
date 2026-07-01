"""LLM-based object detector for the Depot domain."""

from pathlib import Path
from typing import Dict, List, Union

from src.fluent_classification.image_llm_backend_protocol import ImageLLMBackend
from src.llms.domains.depot.prompts import object_detection_system_prompt
from src.object_detection.llm_object_detector import LLMObjectDetector


class LLMDepotObjectDetector(LLMObjectDetector):
    """LLM-based object detector for the Depot domain.

    The LLM returns objects as "D1:depot", "t1:truck", "c1:crane",
    "pile1:pile", "p1:package", etc.

    Images label depots as "D1", "D2" (uppercase), but PDDL uses "d1", "d2"
    (lowercase). The detect() method normalizes this after detection.
    """

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

        # Identity mapping — depot image labels match PDDL object names.
        self.imaged_obj_to_gym_obj_name = {}

        gym_objects = self.extract_objects_from_gt_state()
        self.fewshot_examples = [(init_state_image_path, gym_objects)]

    def _get_system_prompt(self) -> str:
        """Returns the system prompt for Depot object detection."""
        return object_detection_system_prompt

    @staticmethod
    def _get_result_regex() -> str:
        """Returns regex for depot object detection output.

        Depot labels include uppercase letters and digits (D1, t1, pile1, p1),
        so the pattern is broader than the default.
        """
        return r"\b[A-Za-z]+\d*:[a-z]+\b"

    def detect(self, image: Union[Path, str], *args, **kwargs) -> Dict[str, List[str]]:
        """Detect objects and normalize depot names from D1 → d1."""
        detected_objects = super().detect(image, *args, **kwargs)

        if "depot" in detected_objects:
            detected_objects["depot"] = [
                name.lower() for name in detected_objects["depot"]
            ]

        return detected_objects
