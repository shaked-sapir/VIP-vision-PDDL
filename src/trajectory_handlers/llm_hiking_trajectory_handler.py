from pathlib import Path
from typing import Dict, List

from pddl_plus_parser.lisp_parsers import DomainParser

from src.action_model.gym2SAM_parser import parse_grounded_predicates
from src.fluent_classification.llm_fluent_classifier import LLMFluentClassifier
from src.fluent_classification.llm_hiking_fluent_classifier import LLMHikingFluentClassifier
from src.fluent_classification.llm_npuzzle_fluent_classifier import LLMNpuzzleFluentClassifier
from src.object_detection.llm_hiking_object_detector import LLMHikingObjectDetector
from src.object_detection.llm_npuzzle_object_detector import LLMNpuzzleObjectDetector
from src.trajectory_handlers import ImageTrajectoryHandler
from src.utils.masking import save_masking_info


class LLMHikingImageTrajectoryHandler(ImageTrajectoryHandler):
    """
    LLM-based trajectory handler for the Hanoi domain.
    Uses LLMNpuzzleObjectDetector and LLMNpuzzleFluentClassifier.
    """

    def __init__(self,
                 domain_name,
                 pddl_domain_file: Path,
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
        self.domain = DomainParser(pddl_domain_file, partial_parsing=True).parse_domain()

    def init_visual_components(self, init_state_image_path: Path) -> None:
        """
        In this class, this method should only be called after initializing a specific
        blocksworld problem, because the object detection module depends on blocksworld colors which
        are determined only at problem initialization time - and they are extracted from the initial state image.
        """

        self.object_detector = LLMHikingObjectDetector(
            openai_apikey=self.openai_apikey,
            model=self.object_detector_model,
            temperature=self.object_detector_temperature
        )
        detected_objects_by_type: Dict[str, List[str]] = self.object_detector.detect(str(init_state_image_path))

        self.fluent_classifier = LLMHikingFluentClassifier(
            openai_apikey=self.openai_apikey,
            type_to_objects=detected_objects_by_type,
            model=self.fluent_classifier_model,
            temperature=self.fluent_classification_temperature
        )

        print(f"Initialized LLMHikingImageTrajectoryHandler with detected objects: {detected_objects_by_type}")

    @staticmethod
    def _rename_ground_action(action_str: str) -> str:
        return action_str

    def create_masking_info(self, problem_name: str, imaged_trajectory: list[dict], trajectory_path: Path) -> None:
        trajectory_masking_info = (
                [parse_grounded_predicates(imaged_trajectory[0]['current_state']['unknown'], self.domain)] +
                [parse_grounded_predicates(step['next_state']['unknown'], self.domain)
                 for step in imaged_trajectory]
        )

        # Save to working directory
        save_masking_info(trajectory_path, problem_name, trajectory_masking_info)

    def create_trajectory_and_masks(self, problem_name: str, actions: List[str], images_path: Path) -> List[dict]:
        """
        Creates trajectory and masking info files from images.

        This method:
        1. Initializes visual components (object detection) if not already done
        2. Runs fluent classification on all images
        3. Saves trajectory file (problem_name.trajectory)
        4. Saves masking info file (problem_name.masking_info)

        Returns:
            imaged_trajectory: List of dicts containing predicted states for each step
        """
        imaged_trajectory = super().image_trajectory_pipeline(problem_name, actions, images_path)

        self.create_masking_info(problem_name, imaged_trajectory, images_path)

        return imaged_trajectory
