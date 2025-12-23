from pathlib import Path
from typing import Dict, List

from pddl_plus_parser.lisp_parsers import DomainParser

from src.action_model.gym2SAM_parser import parse_grounded_predicates
from src.fluent_classification.image_llm_backend_factory import ImageLLMBackendFactory
from src.fluent_classification.llm_blocks_fluent_classifier import LLMBlocksFluentClassifier
from src.object_detection.llm_blocks_object_detector import LLMBlocksObjectDetector
from src.trajectory_handlers import ImageTrajectoryHandler
from src.utils.masking import save_masking_info


class LLMBlocksImageTrajectoryHandler(ImageTrajectoryHandler):
    """
   LLM-based trajectory handler for the Blocksworld domain.
   Uses LLMBlocksObjectDetector and LLMBlocksFluentClassifier.
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
        In this class, this method should only be called after initializing a specific
        blocksworld problem, because the object detection module depends on blocksworld colors which
        are determined only at problem initialization time - and they are extracted from the initial state image.
        """

        self.object_detector = LLMBlocksObjectDetector(
            llm_backend=ImageLLMBackendFactory.create(
                vendor=self.vendor,
                model_type="object_detection"
            ),
            init_state_image_path=init_state_image_path
        )
        detected_objects_by_type: Dict[str, List[str]] = self.object_detector.detect(str(init_state_image_path))

        self.fluent_classifier = LLMBlocksFluentClassifier(
            llm_backend=ImageLLMBackendFactory.create(
                vendor=self.vendor,
                model_type="fluent_classification"
            ),
            type_to_objects=detected_objects_by_type,
            init_state_image_path=init_state_image_path
        )

        print(f"Initialized LLMBlocksImageTrajectoryHandler with detected objects: {detected_objects_by_type}")

    @staticmethod
    def _rename_ground_action(action_str: str) -> str:
        """
        in the pddlgym, the "blocks" domain actions are with "-" instead of "_" (in amlgym format).
        :param action_str: action to transform from gym format to our format
        :return:
        """
        return (action_str.replace('pick-up', 'pick_up')
                .replace('put-down', 'put_down')
                .replace(', robot:robot', '')
                )

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

    def _manipulate_trajectory_json(self, gt_trajectory_json: list) -> list:
        """
        Override to apply blocksworld-specific transformations to the trajectory JSON.

        Example transformations:
        - Modify action names (e.g., pick-up → pick_up, put-down → put_down)
        - Simplify predicates (e.g., handempty(robot:robot) → handempty())
        - Remove redundant robot parameters from actions

        Args:
            gt_trajectory_json: List of trajectory steps

        Returns:
            Modified trajectory JSON with blocksworld-specific transformations
        """
        for step in gt_trajectory_json:
            # Modify literals in current_state and next_state
            for state_key in ['current_state', 'next_state']:
                if state_key in step and 'literals' in step[state_key]:
                    literals = step[state_key]['literals']
                    new_literals = []
                    for lit in literals:
                        # Example: Change handempty(robot:robot) to handempty()
                        if lit == "handempty(robot:robot)":
                            new_literals.append("handempty()")
                        # Example: Remove handfull(robot:robot)
                        elif lit == "handfull(robot:robot)":
                            continue
                        else:
                            new_literals.append(lit)
                    step[state_key]['literals'] = new_literals

            # Modify ground_action
            if 'ground_action' in step:
                action = step['ground_action']
                # Example: pick-up(b:block, robot:robot) → pick_up(b:block)
                import re
                action = re.sub(r'pick-up\(([^,]+):block,\s*robot:robot\)', r'pick_up(\1:block)', action)
                action = re.sub(r'put-down\(([^,]+):block,\s*robot:robot\)', r'put_down(\1:block)', action)
                step['ground_action'] = action

        return gt_trajectory_json
