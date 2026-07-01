"""LLM-based trajectory handler for the Depot domain."""

from pathlib import Path
from typing import Dict, List

from pddl_plus_parser.lisp_parsers import DomainParser

from src.action_model.gym2SAM_parser import parse_grounded_predicates
from src.fluent_classification.image_llm_backend_factory import ImageLLMBackendFactory
from src.fluent_classification.llm_depot_fluent_classifier import LLMDepotFluentClassifier
from src.object_detection.llm_depot_object_detector import LLMDepotObjectDetector
from src.trajectory_handlers import ImageTrajectoryHandler
from src.utils.masking import save_masking_info
from src.utils.trajectory_json_converter import convert_trajectory_to_json


class LLMDepotImageTrajectoryHandler(ImageTrajectoryHandler):
    """LLM-based trajectory handler for the Depot domain.

    Uses LLMDepotObjectDetector and LLMDepotFluentClassifier.
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

    def get_image_path_by_index(self, image_dir: Path, image_sequential_index: int) -> str:
        """Override to use unpadded image names (state_0.png instead of state_0000.png).

        Depot images come from an external source that doesn't zero-pad indices.
        """
        image_name = f"state_{image_sequential_index}.png"
        return self.get_image_full_path(image_dir, image_name)

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

        self.object_detector = LLMDepotObjectDetector(
            llm_backend=ImageLLMBackendFactory.create(
                vendor=self.vendor,
                model_type="object_detection",
            ),
            init_state_image_path=init_state_image_path,
        )
        detected_objects_by_type: Dict[str, List[str]] = self.object_detector.detect(
            str(init_state_image_path)
        )

        self.fluent_classifier = LLMDepotFluentClassifier(
            llm_backend=ImageLLMBackendFactory.create(
                vendor=self.vendor,
                model_type="fluent_classification",
            ),
            type_to_objects=detected_objects_by_type,
            init_state_image_path=init_state_image_path,
        )

        print(
            f"Initialized LLMDepotImageTrajectoryHandler with "
            f"detected objects: {detected_objects_by_type}"
        )

    def create_masking_info(
        self,
        problem_name: str,
        imaged_trajectory: list[dict],
        trajectory_path: Path,
    ) -> None:
        """Create and save masking info from the imaged trajectory."""
        trajectory_masking_info = (
            [parse_grounded_predicates(
                imaged_trajectory[0]["current_state"]["unknown"], self.domain
            )]
            + [
                parse_grounded_predicates(step["next_state"]["unknown"], self.domain)
                for step in imaged_trajectory
            ]
        )
        save_masking_info(trajectory_path, problem_name, trajectory_masking_info)

    def create_trajectory_and_masks(
        self,
        problem_name: str,
        actions: List[str],
        images_path: Path,
    ) -> List[dict]:
        """Create trajectory and masking info files from images.

        1. Initializes visual components (object detection) if not already done
        2. Runs fluent classification on all images
        3. Saves trajectory file (problem_name.trajectory)
        4. Saves masking info file (problem_name.masking_info)

        Returns:
            imaged_trajectory: List of dicts containing predicted states per step.
        """
        imaged_trajectory = super().image_trajectory_pipeline(
            problem_name, actions, images_path
        )
        self.create_masking_info(problem_name, imaged_trajectory, images_path)
        return imaged_trajectory
