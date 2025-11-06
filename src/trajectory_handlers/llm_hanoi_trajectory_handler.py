"""
LLM-based trajectory handler for Hanoi domain.

Uses GPT-4 Vision for object detection and fluent classification.
"""

from src.fluent_classification.llm_hanoi_fluent_classifier import LLMHanoiFluentClassifier
from src.object_detection.llm_hanoi_object_detector import LLMHanoiObjectDetector
from src.trajectory_handlers import ImageTrajectoryHandler


class LLMHanoiImageTrajectoryHandler(ImageTrajectoryHandler):
    """
    LLM-based trajectory handler for the Hanoi domain.
    Uses LLMHanoiObjectDetector and LLMHanoiFluentClassifier (both GPT-4 Vision-based).
    """

    def __init__(self, domain_name: str, openai_apikey: str, trajectory_size_limit: int = 1000):
        """
        Initialize the LLM-based Hanoi trajectory handler.

        :param domain_name: Name of the PDDL gym domain (e.g., 'PDDLEnvHanoi-v0')
        :param openai_apikey: OpenAI API key for GPT-4 Vision
        :param trajectory_size_limit: Maximum trajectory size
        """
        # Initialize object detector and fluent classifier
        object_detector = LLMHanoiObjectDetector(openai_apikey=openai_apikey)
        fluent_classifier = LLMHanoiFluentClassifier(openai_apikey=openai_apikey)

        super().__init__(
            domain_name=domain_name,
            trajectory_size_limit=trajectory_size_limit,
            object_detector=object_detector,
            fluent_classifier=fluent_classifier
        )

        self.openai_apikey = openai_apikey

    def init_visual_components(self, init_state_image_path=None) -> None:
        """
        Initialize the LLM-based object detector and fluent classifier for Hanoi domain.

        For LLM-based Hanoi, we need to set the type_to_objects mapping for the fluent classifier.

        :param init_state_image_path: Path to initial state image (used to detect objects)
        """
        if init_state_image_path is None:
            raise ValueError("init_state_image_path is required for LLM-based Hanoi trajectory handler")

        # Detect objects in the initial state to determine which discs and pegs are present
        detected_objects = self.object_detector.detect_objects(init_state_image_path)

        # Extract object names by type
        discs = []
        pegs = []

        for obj_label in detected_objects.keys():
            obj_str = str(obj_label)
            obj_name = obj_str.split(":")[0]  # e.g., "d1" from "d1:disc"
            obj_type = obj_str.split(":")[1]  # e.g., "disc" from "d1:disc"

            if obj_type == "disc":
                discs.append(obj_name)
            elif obj_type == "peg":
                pegs.append(obj_name)

        # Sort for consistency
        discs.sort()
        pegs.sort()

        # Set type_to_objects for the fluent classifier
        type_to_objects = {
            "disc": discs,
            "peg": pegs
        }

        self.fluent_classifier.set_type_to_objects(type_to_objects)

        print(f"LLM-based Hanoi visual components initialized")
        print(f"Detected discs: {discs}")
        print(f"Detected pegs: {pegs}")
        print(f"Object detector: LLMHanoiObjectDetector (GPT-4 Vision)")
        print(f"Fluent classifier: LLMHanoiFluentClassifier (GPT-4 Vision)")
