"""Base class for image trajectory handlers — inference pipeline only.

This module contains only the shared inference logic (classify images → .trajectory + .masking_info).
PDDLGym-specific logic lives in pddlgym_trajectory_handler.py.
External-source logic lives in external_trajectory_handler.py.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from pddl_plus_parser.models import Domain

from src.action_model.gym2SAM_parser import parse_grounded_predicates
from src.action_model.pddl2gym_parser import (
    parse_image_predicate_to_gym, is_positive_gym_predicate, is_unknown_gym_predicate,
)
from src.fluent_classification.base_fluent_classifier import FluentClassifier
from src.object_detection.base_object_detector import ObjectDetector
from src.utils.masking import save_masking_info
from src.utils.pddl_gym import multi_replace_predicate
from src.utils.pddl_trajectory import build_trajectory_file


class ImageTrajectoryHandler(ABC):
    """Abstract base for all image trajectory handlers.

    Provides the shared inference pipeline: classify images → build .trajectory + .masking_info.

    Subclasses must implement:
        init_visual_components — set up self.object_detector and self.fluent_classifier.
        run_pipeline — single entry point that produces .trajectory + .masking_info
                        from whatever source the handler uses (gym, external files, etc.).

    The two concrete subclass families are:
        PDDLGymImageTrajectoryHandler — generates images via PDDLGym, then runs inference.
        ExternalImageTrajectoryHandler — reads pre-existing images + GT files, then runs inference.

    To add a new domain:
        1. Choose the right base (PDDLGym or External).
        2. Implement init_visual_components (or use LLMVisualComponentsMixin).
        3. Override hooks as needed (_rename_ground_action, _pre_init_hook, etc.).
        4. run_pipeline is inherited — no need to override unless the domain
           has a non-standard orchestration flow.
    """

    object_detector: ObjectDetector
    fluent_classifier: FluentClassifier
    domain: Domain

    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self.object_detector = None
        self.fluent_classifier = None
        self.seq_idx_format = '04d'

    @abstractmethod
    def init_visual_components(self, *args, **kwargs) -> None:
        """Initialize the object detector and fluent classifier."""
        pass

    @abstractmethod
    def run_pipeline(self, problem_name: str, images_path: Path, **kwargs) -> List[dict]:
        """Run the full end-to-end pipeline for this handler.

        This is the single entry point callers should use. Each subclass family
        defines what "end-to-end" means:

        - **PDDLGym handlers**: generate images from the gym environment, produce a
          GT trajectory JSON, then run LLM/deterministic inference to create
          .trajectory + .masking_info files.
        - **External handlers**: read pre-existing images and GT files from disk,
          auto-detect the image naming format, then run inference to create
          .trajectory + .masking_info files.

        Args:
            problem_name: PDDL problem identifier (e.g. "problem0").
            images_path: Directory for images — written to (gym) or read from (external).
            **kwargs: Handler-specific options (e.g. num_steps, planner for gym).

        Returns:
            The inferred trajectory as a list of step dicts.
        """
        pass

    # ── Image path helpers ──────────────────────────────────────────────

    @staticmethod
    def get_image_full_path(image_dir: Path, image_name: str) -> str:
        return os.path.join(image_dir, image_name)

    def get_image_path_by_index(self, image_dir: Path, image_sequential_index: int) -> str:
        return self.get_image_full_path(
            image_dir, f"state_{image_sequential_index:{self.seq_idx_format}}.png")

    def _set_seq_idx_format(self, images_path: Path) -> None:
        """Auto-detect image filename padding from existing files.

        Inspects the first state_0*.png file to determine the padding format.
        E.g. state_0.png → 'd' (unpadded), state_0000.png → '04d'.
        """
        first = next(Path(images_path).glob("state_0*.png"), None)
        if first is None:
            raise FileNotFoundError(f"No state_0*.png found in {images_path}")
        index_part = first.stem.split("_", 1)[1]
        self.seq_idx_format = f'0{len(index_part)}d' if len(index_part) > 1 else 'd'

    # ── Hook for domain-specific action renaming ────────────────────────

    @staticmethod
    def _rename_ground_action(ground_action: str) -> str:
        return ground_action

    # ── Core inference pipeline ─────────────────────────────────────────

    def construct_trajectory_from_images(self, images_path: Path,
                                         ground_actions: List[str]) -> List[dict]:
        """Classify each image frame and build an inferred trajectory.

        First state uses few-shot GT examples; subsequent states are LLM-classified.
        """
        imaged_trajectory = []
        current_state_predicates = {}

        for i, action in enumerate(ground_actions):
            next_image_path = self.get_image_path_by_index(images_path, i + 1)
            next_state_predicates = self.fluent_classifier.classify(next_image_path)

            if i == 0:
                current_literals = [
                    multi_replace_predicate(p, self.fluent_classifier.imaged_obj_to_gym_obj_name)
                    for p in self.fluent_classifier.fewshot_examples[0][1]]
            else:
                current_literals = [
                    parse_image_predicate_to_gym(pred, val)
                    for pred, val in current_state_predicates.items()]

            next_literals = [parse_image_predicate_to_gym(pred, val)
                             for pred, val in next_state_predicates.items()]

            action = self._rename_ground_action(action)

            imaged_trajectory.append({
                "step": i + 1,
                "current_state": {
                    "literals": [p for p in current_literals if is_positive_gym_predicate(p)],
                    "unknown": [p for p in current_literals if is_unknown_gym_predicate(p)],
                },
                "ground_action": action,
                "next_state": {
                    "literals": [p for p in next_literals if is_positive_gym_predicate(p)],
                    "unknown": [p for p in next_literals if is_unknown_gym_predicate(p)],
                },
            })
            current_state_predicates = next_state_predicates

        return imaged_trajectory

    def image_trajectory_pipeline(self, problem_name: str, actions: List[str],
                                   images_path: Path) -> List[dict]:
        """Init visual components if needed, classify images, write .trajectory file."""
        if not self.object_detector or not self.fluent_classifier:
            self.init_visual_components(
                Path(self.get_image_path_by_index(images_path, 0)))

        imaged_trajectory = self.construct_trajectory_from_images(images_path, actions)
        build_trajectory_file(imaged_trajectory, problem_name, images_path)
        return imaged_trajectory

    def create_masking_info(self, problem_name: str, imaged_trajectory: list[dict],
                            trajectory_path: Path) -> None:
        """Extract and save masking info from the imaged trajectory's unknown predicates."""
        trajectory_masking_info = (
            [parse_grounded_predicates(imaged_trajectory[0]['current_state']['unknown'], self.domain)] +
            [parse_grounded_predicates(step['next_state']['unknown'], self.domain)
             for step in imaged_trajectory]
        )
        save_masking_info(trajectory_path, problem_name, trajectory_masking_info)

    def create_trajectory_and_masks(self, problem_name: str, actions: List[str],
                                     images_path: Path) -> List[dict]:
        """Run image_trajectory_pipeline and save masking info."""
        imaged_trajectory = self.image_trajectory_pipeline(problem_name, actions, images_path)
        self.create_masking_info(problem_name, imaged_trajectory, images_path)
        return imaged_trajectory
