"""Example usage of the modular inconsistency detection system."""

from pathlib import Path
from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain

from src.plan_denoising import (
    InconsistencyDetector,
    FrameAxiomDetector,
    EffectsDetector,
    TransitionExtractor,
)
from src.utils.config import load_config


def example_main_detector(pddl_domain_file: Path, trajectory_path: Path):
    """Example: Using the main InconsistencyDetector coordinator."""
    # Load domain
    domain: Domain = DomainParser(pddl_domain_file).parse_domain()

    # Create main detector
    detector = InconsistencyDetector(domain)

    # Load and detect all violations
    violations = detector.detect_all_violations_in_trajectory(trajectory_path)

    # Print all violations
    detector.print_all_violations(violations)

    # Access specific violation types
    frame_axiom_violations = violations['frame_axiom']
    determinism_violations = violations['determinism']

    print(f"\nFound {len(frame_axiom_violations)} frame axiom violations")
    print(f"Found {len(determinism_violations)} determinism violations")


def example_individual_detectors():
    """Example: Using individual detectors directly."""
    # Load domain
    domain_parser = DomainParser()
    domain = domain_parser.parse_domain(Path("path/to/domain.pddl"))

    # Create transition extractor
    extractor = TransitionExtractor(domain)

    # Load trajectory
    from pddl_plus_parser.lisp_parsers import TrajectoryParser
    trajectory_parser = TrajectoryParser(partial_domain=domain)
    observation = trajectory_parser.parse_trajectory(Path("path/to/trajectory.pddl+"))

    # Extract transitions
    transitions = extractor.extract_transitions(observation)

    # Create and use individual detectors
    frame_detector = FrameAxiomDetector(domain)
    determinism_detector = EffectsDetector(domain)

    # Detect specific types of violations
    frame_violations = frame_detector.detect(transitions)
    determinism_violations = determinism_detector.detect(transitions)

    # Print results
    print("\n--- Frame Axiom Violations ---")
    frame_detector.print_violations(frame_violations)

    print("\n--- Determinism Violations ---")
    determinism_detector.print_violations(determinism_violations)


def example_frame_axiom_only(pddl_domain_file: Path, trajectory_path: Path, trajectory_masking_path: Path):
    """Example: Detecting only frame axiom violations."""
    # Load domain
    domain: Domain = DomainParser(pddl_domain_file).parse_domain()

    # Create detector
    detector = InconsistencyDetector(domain)

    # Detect only frame axiom violations
    violations = detector.detect_frame_axiom_violations_in_trajectory(trajectory_path, trajectory_masking_path)

    # Print violations
    detector.print_frame_axiom_violations(violations)

    # Process each violation
    for violation in violations:
        print(f"Transition {violation.transition_index}: {violation.fluent}")
        print(f"  Type: {violation.violation_type}")
        print(f"  Action: {violation.action_name}")


def example_effects_only(pddl_domain_file: Path, trajectory_path: Path, trajectory_masking_path: Path):
    """Example: Detecting only determinism violations."""
    domain: Domain = DomainParser(pddl_domain_file).parse_domain()

    # Create detector
    detector = InconsistencyDetector(domain)

    # Detect inconsistencies
    inconsistencies = detector.detect_effects_violations_in_trajectory(trajectory_path, trajectory_masking_path)

    # Print results
    detector.print_effects_violations(inconsistencies)


def example_custom_detector():
    """Example: Creating a custom detector for a new violation type."""
    from src.plan_denoising.detectors.base_detector import BaseDetector, Transition
    from dataclasses import dataclass
    from typing import List

    @dataclass
    class CustomViolation:
        """Custom violation type."""
        transition_index: int
        description: str

        def __str__(self):
            return f"CustomViolation at transition {self.transition_index}: {self.description}"

    class CustomDetector(BaseDetector):
        """Custom detector for a specific type of violation."""

        def detect(self, transitions: List[Transition]) -> List[CustomViolation]:
            """Detect custom violations."""
            violations = []
            # Custom detection logic here
            for trans in transitions:
                # Example: detect if more than 5 fluents changed
                changes = len(trans.next_state.symmetric_difference(trans.prev_state))
                if changes > 5:
                    violations.append(CustomViolation(
                        transition_index=trans.index,
                        description=f"Too many changes: {changes} fluents"
                    ))
            return violations

        def print_violations(self, violations: List[CustomViolation]) -> None:
            """Print custom violations."""
            if not violations:
                print("No custom violations detected!")
                return
            print(f"\nDetected {len(violations)} custom violations:\n")
            for idx, violation in enumerate(violations, 1):
                print(f"{idx}. {violation}")

    # Use the custom detector
    domain_parser = DomainParser()
    domain = domain_parser.parse_domain(Path("path/to/domain.pddl"))

    custom_detector = CustomDetector(domain)
    extractor = TransitionExtractor(domain)

    from pddl_plus_parser.lisp_parsers import TrajectoryParser
    trajectory_parser = TrajectoryParser(partial_domain=domain)
    observation = trajectory_parser.parse_trajectory(Path("path/to/trajectory.pddl+"))

    transitions = extractor.extract_transitions(observation)
    violations = custom_detector.detect(transitions)
    custom_detector.print_violations(violations)


if __name__ == "__main__":
    print("Modular Inconsistency Detection Examples")
    print("=" * 60)

    config = load_config()
    domain = "blocks"  # or "hanoi", etc.
    pddl_domain_file = Path(config['domains'][domain]['domain_file'])
    curr_experiment = "llm_cv_test__PDDLEnvBlocks-v0__gpt-4o__steps=25__01-11-2025T16:01:36"
    trajectory_path = Path(f"{config['paths']['experiments_dir']}/{curr_experiment}/problem1.trajectory")
    trajectory_masking_path = Path(f"{config['paths']['experiments_dir']}/{curr_experiment}/problem1.masking_info")
    # Uncomment to run examples:
    # example_main_detector(pddl_domain_file, trajectory_path)
    # example_individual_detectors()
    example_frame_axiom_only(pddl_domain_file, trajectory_path, trajectory_masking_path)
    # example_effects_only(pddl_domain_file, trajectory_path, trajectory_masking_path)
    # example_custom_detector()
