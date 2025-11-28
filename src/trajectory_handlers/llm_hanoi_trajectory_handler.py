from pathlib import Path
from typing import Dict, List

from src.fluent_classification.llm_hanoi_fluent_classifier import LLMHanoiFluentClassifier
from src.object_detection.llm_hanoi_object_detector import LLMHanoiObjectDetector
from src.trajectory_handlers import ImageTrajectoryHandler


class LLMHanoiImageTrajectoryHandler(ImageTrajectoryHandler):
    """
    LLM-based trajectory handler for the Hanoi domain.
    Uses LLMHanoiObjectDetector and LLMHanoiFluentClassifier.
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
        blocks problem, because the object detection module depends on blocks colors which
        are determined only at problem initialization time - and they are extracted from the initial state image.
        """

        self.object_detector = LLMHanoiObjectDetector(
            openai_apikey=self.openai_apikey,
            model=self.object_detector_model,
            temperature=self.object_detector_temperature
        )
        detected_objects_by_type: Dict[str, List[str]] = self.object_detector.detect(str(init_state_image_path))

        self.fluent_classifier = LLMHanoiFluentClassifier(
            openai_apikey=self.openai_apikey,
            type_to_objects=detected_objects_by_type,
            model=self.fluent_classifier_model,
            temperature=self.fluent_classification_temperature
        )

        print(f"Initialized LLMHanoiImageTrajectoryHandler with detected objects: {detected_objects_by_type}")

    @staticmethod
    def _rename_ground_action(action_str: str) -> str:
        """
        in the pddlgym, the "hanoi" domain is typeless so we need to rename the actions to include types.
        :param action_str: action to transform from gym format to our format
        :return:
        """
        # split into: original_name, "(args)"
        name_end: int = action_str.index('(')
        args_str: str = action_str[name_end:]  # includes parentheses

        # extract argument names
        args: list[str] = args_str[1:-1].split(',')
        names: list[str] = [a.split(':')[0].strip() for a in args]

        # classify 2nd + 3rd args
        c2 = "peg" if names[1].startswith("peg") else "disc"
        c3 = "peg" if names[2].startswith("peg") else "disc"

        # new action name
        new_name = f"move-{c2}-{c3}"

        # return new name + original arguments
        return f"{new_name}{args_str}"
