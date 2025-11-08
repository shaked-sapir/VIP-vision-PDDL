"""Main inconsistency detector that coordinates all violation detectors."""

from pathlib import Path
from typing import List, Union

from pddl_plus_parser.lisp_parsers import TrajectoryParser
from pddl_plus_parser.models import Observation, Domain

from src.plan_denoising.detectors.base_detector import Transition
from src.plan_denoising.detectors.frame_axiom_detector import FrameAxiomDetector, FrameAxiomViolation
from src.plan_denoising.detectors.effects_detector import EffectsDetector, EffectsViolation
from src.plan_denoising.transition_extractor import TransitionExtractor
from src.utils.masking import load_masked_observation


class InconsistencyDetector:
    """
    Main coordinator for detecting all types of inconsistencies in PDDL trajectories.

    This class coordinates multiple specialized detectors:
    - FrameAxiomDetector: Detects frame axiom violations
    - DeterminismDetector: Detects determinism violations (is_effect conflicts)

    Each detector is modular and can be used independently or through this coordinator.
    """

    def __init__(self, domain: Domain):
        """
        Initialize the inconsistency detector.

        :param domain: PDDL domain containing action definitions and effects
        """
        self.domain = domain
        self.trajectory_parser = TrajectoryParser(partial_domain=domain)
        self.transition_extractor = TransitionExtractor(domain)

        # Initialize specialized detectors
        self.frame_axiom_detector = FrameAxiomDetector(domain)
        self.effects_detector = EffectsDetector(domain)

    def load_trajectory(self, trajectory_path: Path) -> Observation:
        """
        Load a PDDL trajectory file.

        :param trajectory_path: Path to the .trajectory file
        :return: Observation object containing the trajectory
        """
        return self.trajectory_parser.parse_trajectory(trajectory_path)

    def load_masked_observation(self, trajectory_path: Path, trajectory_masking_path: Path) -> Observation:
        return load_masked_observation(trajectory_path, trajectory_masking_path, self.domain)

    def extract_transitions(self, observation: Observation) -> List[Transition]:
        """
        Extract transitions from an observation.

        :param observation: Observation object from trajectory parser
        :return: List of transitions
        """
        return self.transition_extractor.extract_transitions(observation)

    # ==================== Frame Axiom Violations ====================

    def detect_frame_axiom_violations(
        self,
        transitions: List[Transition]
    ) -> List[FrameAxiomViolation]:
        """
        Detect frame axiom violations in the given transitions.

        :param transitions: List of transitions from a trajectory
        :return: List of detected frame axiom violations
        """
        return self.frame_axiom_detector.detect(transitions)

    def detect_frame_axiom_violations_from_observation(
        self,
        observation: Observation
    ) -> List[FrameAxiomViolation]:
        """
        Detect frame axiom violations directly from an Observation object.

        :param observation: Observation object
        :return: List of detected frame axiom violations
        """
        transitions = self.extract_transitions(observation)
        return self.detect_frame_axiom_violations(transitions)

    def detect_frame_axiom_violations_in_trajectory(
        self,
        trajectory_path: Path,
        trajectory_masking_path: Path
    ) -> List[FrameAxiomViolation]:
        """
        Load a trajectory and detect all frame axiom violations.

        :param trajectory_path: Path to the .trajectory file
        :param trajectory_masking_path: Path to the trajectory masking file
        :return: List of detected frame axiom violations
        """
        observation = self.load_masked_observation(trajectory_path, trajectory_masking_path)
        return self.detect_frame_axiom_violations_from_observation(observation)

    # ==================== Effects Violations ====================

    def detect_effects_violations(
        self,
        transitions: List[Transition]
    ) -> List[EffectsViolation]:
        """
        Detect determinism violations in the given transitions.

        :param transitions: List of transitions from a trajectory
        :return: List of detected determinism violations
        """
        return self.effects_detector.detect(transitions)

    def detect_effects_violations_from_observation(
        self,
        observation: Observation
    ) -> List[EffectsViolation]:
        """
        Detect determinism violations directly from an Observation object.

        :param observation: Observation object
        :return: List of detected determinism violations
        """
        transitions = self.extract_transitions(observation)
        return self.detect_effects_violations(transitions)

    def detect_effects_violations_in_trajectory(
        self,
        trajectory_path: Path,
        trajectory_masking_path: Path
    ) -> List[EffectsViolation]:
        """
        Load a trajectory and detect all determinism violations.

        :param trajectory_path: Path to the .trajectory file
        :param trajectory_masking_path: Path to the trajectory masking file
        :return: List of detected determinism violations
        """
        observation = self.load_masked_observation(trajectory_path, trajectory_masking_path)
        return self.detect_effects_violations_from_observation(observation)

    # ==================== Unified Detection Methods ====================

    def detect_all_violations(
        self,
        transitions: List[Transition]
    ) -> dict[str, List[Union[FrameAxiomViolation, EffectsViolation]]]:
        """
        Detect all types of violations in the given transitions.

        :param transitions: List of transitions from a trajectory
        :return: Dictionary with violation types as keys and lists of violations as values
        """
        return {
            'frame_axiom': self.detect_frame_axiom_violations(transitions),
            'determinism': self.detect_effects_violations(transitions),
        }

    def detect_all_violations_from_observation(
        self,
        observation: Observation
    ) -> dict[str, List[Union[FrameAxiomViolation, EffectsViolation]]]:
        """
        Detect all types of violations directly from an Observation object.

        :param observation: Observation object
        :return: Dictionary with violation types as keys and lists of violations as values
        """
        transitions = self.extract_transitions(observation)
        return self.detect_all_violations(transitions)

    def detect_all_violations_in_trajectory(
        self,
        trajectory_path: Path
    ) -> dict[str, List[Union[FrameAxiomViolation, EffectsViolation]]]:
        """
        Load a trajectory and detect all types of violations.

        :param trajectory_path: Path to the .trajectory file
        :return: Dictionary with violation types as keys and lists of violations as values
        """
        observation = self.load_trajectory(trajectory_path)
        return self.detect_all_violations_from_observation(observation)

    # ==================== Print Methods ====================

    def print_frame_axiom_violations(self, violations: List[FrameAxiomViolation]) -> None:
        """Print frame axiom violations."""
        self.frame_axiom_detector.print_violations(violations)

    def print_effects_violations(self, violations: List[EffectsViolation]) -> None:
        """Print determinism violations."""
        self.effects_detector.print_violations(violations)

    def print_all_violations(
        self,
        violations: dict[str, List[Union[FrameAxiomViolation, EffectsViolation]]]
    ) -> None:
        """
        Print all types of violations.

        :param violations: Dictionary with violation types and their lists
        """
        print("\n" + "="*60)
        print("TRAJECTORY VIOLATION REPORT")
        print("="*60)

        # Print frame axiom violations
        if 'frame_axiom' in violations:
            print("\n--- Frame Axiom Violations ---")
            self.print_frame_axiom_violations(violations['frame_axiom'])

        # Print determinism violations
        if 'determinism' in violations:
            print("\n--- Determinism Violations ---")
            self.print_effects_violations(violations['determinism'])

        # Summary
        total_violations = sum(len(v) for v in violations.values())
        print("="*60)
        print(f"Total violations: {total_violations}")
        print("="*60)
