from pathlib import Path
from typing import Dict, List

from pddl_plus_parser.lisp_parsers import DomainParser

from src.action_model.gym2SAM_parser import parse_grounded_predicates
from src.fluent_classification.image_llm_backend_factory import ImageLLMBackendFactory
from src.fluent_classification.llm_hanoi_fluent_classifier import LLMHanoiFluentClassifier
from src.object_detection.llm_hanoi_object_detector import LLMHanoiObjectDetector
from src.trajectory_handlers import ImageTrajectoryHandler
from src.utils.masking import save_masking_info


class LLMHanoiImageTrajectoryHandler(ImageTrajectoryHandler):
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
        In this class, this method should only be called after initializing a specific
        blocksworld problem, because the object detection module depends on blocksworld colors which
        are determined only at problem initialization time - and they are extracted from the initial state image.
        """

        self.object_detector = LLMHanoiObjectDetector(
            llm_backend=ImageLLMBackendFactory.create(
                vendor=self.vendor,
                model_type="object_detection"
            ),
            init_state_image_path=init_state_image_path
        )
        detected_objects_by_type: Dict[str, List[str]] = self.object_detector.detect(str(init_state_image_path))

        self.fluent_classifier = LLMHanoiFluentClassifier(
            llm_backend=ImageLLMBackendFactory.create(
                vendor=self.vendor,
                model_type="fluent_classification"
            ),
            type_to_objects=detected_objects_by_type,
            init_state_image_path=init_state_image_path
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
        new_name = f"move_{c2}_{c3}"

        # return new name + original arguments
        return f"{new_name}{args_str}"

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
        Transform hanoi trajectory from pddlgym format to typed format.

        Transformations:
        1. smaller(...) → smaller-peg(...) or smaller-disc(...) based on peg presence
        2. on(...) → on-peg(...) or on-disc(...) based on peg presence
        3. clear(...) → clear-peg(...) or clear-disc(...) based on peg presence
        4. Change :default type to :peg or :disc based on variable names
        5. Transform move(...) actions using _rename_ground_action

        Args:
            gt_trajectory_json: List of trajectory steps in pddlgym format

        Returns:
            Modified trajectory JSON in typed hanoi format
        """
        import re

        def transform_object_type(obj: str) -> str:
            """Transform object from :default to :peg or :disc based on name."""
            if ':default' not in obj:
                return obj

            name = obj.split(':')[0]
            if name.startswith('peg'):
                return f"{name}:peg"
            elif name.startswith('d'):
                return f"{name}:disc"
            return obj

        def contains_peg(literal: str) -> bool:
            """Check if a literal contains any peg argument."""
            # Extract arguments from literal
            match = re.match(r'\w+\((.*)\)', literal)
            if match:
                args_str = match.group(1)
                # Check if any argument starts with 'peg'
                args = [arg.split(':')[0].strip() for arg in args_str.split(',')]
                return any(arg.startswith('peg') for arg in args)
            return False

        def transform_literal(lit: str) -> str:
            """Transform a single literal."""
            # Transform smaller(...) → smaller-peg(...) or smaller-disc(...)
            if lit.startswith('smaller('):
                suffix = 'peg' if contains_peg(lit) else 'disc'
                lit = lit.replace('smaller(', f'smaller-{suffix}(')

            # Transform on(...) → on-peg(...) or on-disc(...)
            elif lit.startswith('on('):
                suffix = 'peg' if contains_peg(lit) else 'disc'
                lit = lit.replace('on(', f'on-{suffix}(')

            # Transform clear(...) → clear-peg(...) or clear-disc(...)
            elif lit.startswith('clear('):
                suffix = 'peg' if contains_peg(lit) else 'disc'
                lit = lit.replace('clear(', f'clear-{suffix}(')

            # Transform :default to :peg or :disc
            lit = re.sub(r'(peg\d+):default', r'\1:peg', lit)
            lit = re.sub(r'(d\d+):default', r'\1:disc', lit)

            return lit

        # Process each step
        for step in gt_trajectory_json:
            # Transform literals in current_state and next_state
            for state_key in ['current_state', 'next_state']:
                if state_key in step and 'literals' in step[state_key]:
                    literals = step[state_key]['literals']
                    new_literals = [transform_literal(lit) for lit in literals]
                    step[state_key]['literals'] = new_literals

                # Transform objects
                if state_key in step and 'objects' in step[state_key]:
                    objects = step[state_key]['objects']
                    new_objects = [transform_object_type(obj) for obj in objects]
                    step[state_key]['objects'] = new_objects

                # Transform goal literals if present
                if state_key in step and 'goal' in step[state_key]:
                    goal_literals = step[state_key]['goal']
                    new_goal_literals = [transform_literal(lit) for lit in goal_literals]
                    step[state_key]['goal'] = new_goal_literals

            # Transform ground_action using existing method
            if 'ground_action' in step:
                original_action = step['ground_action']
                try:
                    transformed_action = self._rename_ground_action(original_action)
                    # Also transform :default to :peg or :disc in the action
                    transformed_action = re.sub(r'(peg\d+):default', r'\1:peg', transformed_action)
                    transformed_action = re.sub(r'(d\d+):default', r'\1:disc', transformed_action)
                    step['ground_action'] = transformed_action
                except Exception as e:
                    print(f"Warning: Failed to transform action '{original_action}': {e}")

        return gt_trajectory_json
