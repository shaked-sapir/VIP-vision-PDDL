from pathlib import Path
from typing import Dict, List

from src.fluent_classification.llm_blocks_fluent_classifier import LLMBlocksFluentClassifier
from src.object_detection.llm_blocks_object_detector import LLMBlocksObjectDetector
from src.trajectory_handlers import ImageTrajectoryHandler


class LLMBlocksImageTrajectoryHandler(ImageTrajectoryHandler):
    """
   LLM-based trajectory handler for the Blocksworld domain.
   Uses LLMBlocksObjectDetector and LLMBlocksFluentClassifier.
   """
    def __init__(self,
                 domain_name,
                 openai_apikey: str,
                 object_detector_model: str = "gpt-4o",
                 object_detection_temperature: float = 1.0,
                 fluent_classifier_model: str = "gpt-4o",
                 fluent_classification_temperature: float = 1.0):
        super().__init__(domain_name=domain_name)
        self.openai_apikey = openai_apikey
        self.object_detector_model = object_detector_model
        self.object_detector_temperature = object_detection_temperature
        self.fluent_classifier_model = fluent_classifier_model
        self.fluent_classification_temperature = fluent_classification_temperature

    def init_visual_components(self, init_state_image_path: Path) -> None:
        """
        In this class, this method should only be called after initializing a specific
        blocksworld problem, because the object detection module depends on blocksworld colors which
        are determined only at problem initialization time - and they are extracted from the initial state image.
        """

        self.object_detector = LLMBlocksObjectDetector(
            openai_apikey=self.openai_apikey,
            model=self.object_detector_model,
            temperature=self.object_detector_temperature
        )
        detected_objects_by_type: Dict[str, List[str]] = self.object_detector.detect(str(init_state_image_path))

        self.fluent_classifier = LLMBlocksFluentClassifier(
            openai_apikey=self.openai_apikey,
            type_to_objects=detected_objects_by_type,
            model=self.fluent_classifier_model,
            temperature=self.fluent_classification_temperature
        )

        print(f"Initialized LLMBlocksImageTrajectoryHandler with detected objects: {detected_objects_by_type}")

    @staticmethod
    def _rename_ground_action(action_str: str) -> str:
        """
        in the pddlgym, the "blocks" domain actions are with "-" instead of "_" (in amlgym format).
        :param action_str: action to transform from gym format to our format
        :return:
        """
        return action_str.replace('pick-up', 'pick_up').replace('put-down', 'put_down')
