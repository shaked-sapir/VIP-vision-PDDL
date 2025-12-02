import json
from pathlib import Path
from typing import Dict, List, Set

from pddl_plus_parser.lisp_parsers import DomainParser

from src.action_model.gym2SAM_parser import parse_grounded_predicates
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
        self.const_predicate_names = {"move-dir-up", "move-dir-down", "move-dir-left", "move-dir-right"}

    def _generate_const_predicates(self, images_path: Path) -> Set[str]:
        """
        Extract constant predicates from the ground truth trajectory.json file.

        Args:
            images_path: Path to the directory containing images and trajectory.json

        Returns:
            Set of constant predicate strings (e.g., "move-dir-up(loc-1-1:location,loc-0-1:location)")
        """
        # Find the trajectory.json file in the images directory
        trajectory_file = None
        for filename in images_path.iterdir():
            if filename.name.endswith("trajectory.json"):
                trajectory_file = filename
                break

        if trajectory_file is None:
            print(f"Warning: No trajectory.json file found in {images_path}")
            return set()

        # Load the ground truth trajectory
        with open(trajectory_file, 'r') as f:
            trajectory_data = json.load(f)

        # Get initial state literals
        if not trajectory_data or len(trajectory_data) == 0:
            print(f"Warning: Empty trajectory data in {trajectory_file}")
            return set()

        # initial_state_literals = trajectory_data[0]['current_state']['literals']
        initial_state_literals = trajectory_data["states"][0]['literals'] #debug only

        # Filter predicates by the specified predicate names
        const_predicates = set()
        for literal in initial_state_literals:
            # Check if this literal matches one of our constant predicate names
            for pred_name in self.const_predicate_names:
                if literal.startswith(f"{pred_name}("):
                    const_predicates.add(literal)
                    break

        print(f"Extracted {len(const_predicates)} constant predicates from {trajectory_file.name}")
        return const_predicates
        
    def init_visual_components(self, init_state_image_path: Path) -> None:
        """
        Initialize visual components for the maze domain.

        Args:
            init_state_image_path: Path to the initial state image for object detection
            images_path: Optional path to the images directory (to extract constant predicates)
        """

        self.object_detector = LLMMazeObjectDetector(
            openai_apikey=self.openai_apikey,
            model=self.object_detector_model,
            temperature=self.object_detector_temperature
        )
        detected_objects_by_type: Dict[str, List[str]] = self.object_detector.detect(str(init_state_image_path))

        # Extract constant predicates if images_path is provided
        images_path: Path = init_state_image_path.parent
        const_predicates: Set[str] = self._generate_const_predicates(images_path)

        self.fluent_classifier = LLMMazeFluentClassifier(
            openai_apikey=self.openai_apikey,
            type_to_objects=detected_objects_by_type,
            model=self.fluent_classifier_model,
            temperature=self.fluent_classification_temperature,
            const_predicates=const_predicates
        )

        print(f"Initialized LLMMazeImageTrajectoryHandler with detected objects: {detected_objects_by_type}")
        if const_predicates:
            print(f"  Loaded {len(const_predicates)} constant predicates")

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
