"""LLM-based trajectory handler for the Gripper domain."""

from pathlib import Path
from typing import Dict, List

from pddl_plus_parser.lisp_parsers import DomainParser

from src.fluent_classification.image_llm_backend_factory import ImageLLMBackendFactory
from src.fluent_classification.llm_gripper_fluent_classifier import LLMGripperFluentClassifier
from src.object_detection.llm_gripper_object_detector import LLMGripperObjectDetector
from src.trajectory_handlers import ImageTrajectoryHandler
from src.utils.trajectory_json_converter import convert_trajectory_to_json


class LLMGripperImageTrajectoryHandler(ImageTrajectoryHandler):
    """LLM-based trajectory handler for the Gripper domain.

    Uses LLMGripperObjectDetector and LLMGripperFluentClassifier.
    If a trajectory.json does not exist but a .trajectory + problem.pddl do,
    the converter is called automatically during init.
    """

    def __init__(
        self,
        domain_name: str,
        pddl_domain_file: Path,
        api_key: str,
        vendor: str = "openai",
        object_detector_model: str = "gpt-4o",
        object_detection_temperature: float = 1.0,
        fluent_classifier_model: str = "gpt-4o",
        fluent_classification_temperature: float = 1.0,
    ):
        super().__init__(domain_name=domain_name)
        self.api_key = api_key
        self.vendor = vendor
        self.object_detector_model = object_detector_model
        self.object_detector_temperature = object_detection_temperature
        self.fluent_classifier_model = fluent_classifier_model
        self.fluent_classification_temperature = fluent_classification_temperature
        self.domain = DomainParser(pddl_domain_file, partial_parsing=True).parse_domain()

    @staticmethod
    def _ensure_trajectory_json(images_dir: Path) -> None:
        """If trajectory.json is missing, convert from .trajectory + problem.pddl."""
        json_files = list(images_dir.glob("*_trajectory.json"))
        if json_files:
            return  # Already exists

        trajectory_files = list(images_dir.glob("*.trajectory"))
        problem_files = list(images_dir.glob("*.pddl"))

        if not trajectory_files or not problem_files:
            return  # Nothing to convert — let the base class handle the error

        convert_trajectory_to_json(
            trajectory_path=trajectory_files[0],
            problem_path=problem_files[0],
        )

    def init_visual_components(self, init_state_image_path: Path) -> None:
        """Initialize the LLM-based object detector and fluent classifier.

        Automatically converts .trajectory → trajectory.json if needed.
        """
        images_dir = init_state_image_path.parent
        self._ensure_trajectory_json(images_dir)

        self.object_detector = LLMGripperObjectDetector(
            llm_backend=ImageLLMBackendFactory.create(
                vendor=self.vendor,
                model_type="object_detection",
            ),
            init_state_image_path=init_state_image_path,
        )
        detected_objects_by_type: Dict[str, List[str]] = self.object_detector.detect(
            str(init_state_image_path)
        )

        self.fluent_classifier = LLMGripperFluentClassifier(
            llm_backend=ImageLLMBackendFactory.create(
                vendor=self.vendor,
                model_type="fluent_classification",
            ),
            type_to_objects=detected_objects_by_type,
            init_state_image_path=init_state_image_path,
        )

        print(
            f"Initialized LLMGripperImageTrajectoryHandler with "
            f"detected objects: {detected_objects_by_type}"
        )

