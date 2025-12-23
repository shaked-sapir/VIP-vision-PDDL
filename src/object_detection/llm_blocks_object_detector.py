from pathlib import Path

from src.fluent_classification.image_llm_backend_protocol import ImageLLMBackend
from src.llms.domains.blocks.prompts import object_detection_system_prompt
from src.object_detection.llm_object_detector import LLMObjectDetector


class LLMBlocksObjectDetector(LLMObjectDetector):
    def __init__(
            self,
            llm_backend: ImageLLMBackend,
            init_state_image_path: Path,
            temperature: float = None,
            inference_mode: bool = False
    ):
        super().__init__(
            llm_backend=llm_backend,
            init_state_image_path=init_state_image_path,
            temperature=temperature,
            inference_mode=inference_mode
        )

        self.imaged_obj_to_gym_obj_name = {
            "red": "a",
            "cyan": "b",
            "blue": "c",
            "green": "d",
            "yellow": "e",
            "pink": "f",
            "gripper": "robot"
        }

        self.fewshot_examples = [(init_state_image_path, self.extract_objects_from_gt_state())]

    def _get_system_prompt(self) -> str:
        return object_detection_system_prompt  # TODO: find a proper way to handle the objects, not hardcoding from the .prompts file
