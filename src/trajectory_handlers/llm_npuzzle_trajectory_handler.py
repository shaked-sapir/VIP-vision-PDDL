from pathlib import Path
from typing import Dict, List

from src.fluent_classification.llm_hanoi_fluent_classifier import LLMHanoiFluentClassifier
from src.fluent_classification.llm_npuzzle_fluent_classifier import LLMNpuzzleFluentClassifier
from src.object_detection.llm_hanoi_object_detector import LLMHanoiObjectDetector
from src.object_detection.llm_npuzzle_object_detector import LLMNpuzzleObjectDetector
from src.trajectory_handlers import ImageTrajectoryHandler


class LLMNpuzzleImageTrajectoryHandler(ImageTrajectoryHandler):
    """
    LLM-based trajectory handler for the Hanoi domain.
    Uses LLMNpuzzleObjectDetector and LLMNpuzzleFluentClassifier.
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

        self.object_detector = LLMNpuzzleObjectDetector(
            openai_apikey=self.openai_apikey,
            model=self.object_detector_model,
            temperature=self.object_detector_temperature
        )
        detected_objects_by_type: Dict[str, List[str]] = self.object_detector.detect(str(init_state_image_path))

        self.fluent_classifier = LLMNpuzzleFluentClassifier(
            openai_apikey=self.openai_apikey,
            type_to_objects=detected_objects_by_type,
            model=self.fluent_classifier_model,
            temperature=self.fluent_classification_temperature
        )

        print(f"Initialized LLMNpuzzleImageTrajectoryHandler with detected objects: {detected_objects_by_type}")

    @staticmethod
    def _rename_ground_action(action_str: str) -> str:
        """
        in the pddlgym, the "move-X" (tile, X, Y, X\Y new) means to move the BLANK to position X,Y
        by sliding the tile at position X,Y into the blank position.
        :param action_str: action to transform from gym format to our format
        :return:
        """
        # split into: original_name, "(args)"
        gym_action_name, args_part = action_str.split("(", 1)
        args_str = args_part.rstrip(")")

        # extract argument names
        arg_names = [a.split(":", 1)[0].strip() for a in args_str.split(",")]

        # extract coordinates
        tile_raw, gym_from_x_cord, gym_from_y_cord, gym_shift_cord = arg_names

        target_position_from = f"p_{gym_from_x_cord[1]}_{gym_from_y_cord[1]}"
        target_tile = f"{tile_raw[0]}_{tile_raw[1]}"

        if gym_action_name in ["move-down", "move-up"]:
            target_position_to = f"p_{gym_from_x_cord[1]}_{gym_shift_cord[1]}"
        elif gym_action_name in ["move-left", "move-right"]:
            target_position_to = f"p_{gym_shift_cord[1]}_{gym_from_y_cord[1]}"

        return f"move({target_tile}:tile, {target_position_from}:position, {target_position_to}:position)"
