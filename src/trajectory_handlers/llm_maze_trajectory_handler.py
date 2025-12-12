from pathlib import Path
from typing import Dict, List

from pddl_plus_parser.lisp_parsers import DomainParser

from src.action_model.gym2SAM_parser import parse_grounded_predicates
from src.fluent_classification.image_llm_backend_factory import ImageLLMBackendFactory
from src.fluent_classification.llm_maze_fluent_classifier import LLMMazeFluentClassifier
from src.object_detection.llm_maze_object_detector import LLMMazeObjectDetector
from src.trajectory_handlers import ImageTrajectoryHandler
from src.utils.masking import save_masking_info


class LLMMazeImageTrajectoryHandler(ImageTrajectoryHandler):
    """
    LLM-based trajectory handler for the Hanoi domain.
    Uses LLMHanoiObjectDetector and LLMHanoiFluentClassifier.
    """

    def __init__(self,
                 domain_name,
                 pddl_domain_file: Path,
                 api_key: str,
                 vendor: str = "openai",
                 object_detector_model: str = "gpt-4o",
                 object_detection_temperature: float = 1.0,
                 fluent_classifier_model: str = "gpt-4o",
                 fluent_classification_temperature: float = 1.0):
        super().__init__(domain_name=domain_name)
        self.api_key = api_key
        self.vendor = vendor
        self.object_detector_model = object_detector_model
        self.object_detector_temperature = object_detection_temperature
        self.fluent_classifier_model = fluent_classifier_model
        self.fluent_classification_temperature = fluent_classification_temperature
        self.domain = DomainParser(pddl_domain_file, partial_parsing=True).parse_domain()

    def init_visual_components(self, init_state_image_path: Path) -> None:
        """
        Initialize visual components for the maze domain.

        Args:
            init_state_image_path: Path to the initial state image for object detection
        """

        self.object_detector = LLMMazeObjectDetector(
            llm_backend=ImageLLMBackendFactory.create(
                vendor=self.vendor,
                model_type="object_detection"
            ),
            init_state_image_path=init_state_image_path
        )
        detected_objects_by_type: Dict[str, List[str]] = self.object_detector.detect(str(init_state_image_path))

        self.fluent_classifier = LLMMazeFluentClassifier(
            llm_backend=ImageLLMBackendFactory.create(
                vendor=self.vendor,
                model_type="fluent_classification"
            ),
            type_to_objects=detected_objects_by_type,
            init_state_image_path=init_state_image_path
        )

        print(f"Initialized LLMMazeImageTrajectoryHandler with detected objects: {detected_objects_by_type}")

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
