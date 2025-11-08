"""
Example usage of the DataRepairer for fixing trajectory violations.

This demonstrates how to use the LLM-based repairer to fix effects violations
(determinism violations) in PDDL trajectories.
"""

from pathlib import Path
from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser
from pddl_plus_parser.models import Domain

from src.plan_denoising import (
    InconsistencyDetector,
    DataRepairer,
    EffectsViolation
)


def example_basic_repair():
    """
    Example: Detect and repair an effects violation.

    This shows the complete workflow:
    1. Load domain and trajectory
    2. Detect effects violations
    3. Use LLM to verify which state is correct
    4. Repair the incorrect state
    """
    # Configuration
    domain_file = Path("path/to/domain.pddl")
    trajectory_file = Path("path/to/trajectory.pddl+")
    images_dir = Path("path/to/images")
    openai_api_key = "your-api-key"
    domain_name = "blocks"  # or "hanoi", etc.

    # Step 1: Load the domain
    print("Step 1: Loading domain...")
    domain: Domain = DomainParser(domain_file).parse_domain()

    # Step 2: Detect violations in the trajectory
    print("\nStep 2: Detecting violations...")
    detector = InconsistencyDetector(domain)
    violations = detector.detect_effects_violations_in_trajectory(trajectory_file)

    if not violations:
        print("No violations detected! Trajectory is consistent.")
        return

    print(f"Found {len(violations)} effects violations")
    detector.print_effects_violations(violations)

    # Step 3: Initialize the repairer
    print("\nStep 3: Initializing data repairer...")
    repairer = DataRepairer(openai_apikey=openai_api_key, model="gpt-4o")

    # Step 4: Repair the first violation
    violation = violations[0]
    print(f"\nStep 4: Repairing violation: {violation}")

    # Get image paths for the conflicting transitions
    # Convention: images are named "state_{index}.png"
    image1_path = images_dir / f"state_{violation.transition1_index + 1}.png"
    image2_path = images_dir / f"state_{violation.transition2_index + 1}.png"

    # Load the observation
    trajectory_parser = TrajectoryParser(partial_domain=domain)
    observation = trajectory_parser.parse_trajectory(trajectory_file)

    # Repair the violation using LLM verification
    repaired_obs, repair_op, repair_choice = repairer.repair_violation(
        observation=observation,
        violation=violation,
        image1_path=image1_path,
        image2_path=image2_path,
        domain_name=domain_name
    )

    print(f"\nRepair completed!")
    print(f"  Repaired transition: {repair_op.transition_index}")
    print(f"  Fluent: {repair_op.fluent_changed}")
    print(f"  Changed from: {repair_op.old_value} → {repair_op.new_value}")


def example_stepwise_repair():
    """
    Example: Manual step-by-step repair process.

    This shows how to use individual DataRepairer methods for more control.
    """
    # Configuration
    openai_api_key = "your-api-key"
    domain_name = "blocks"

    # Initialize repairer
    repairer = DataRepairer(openai_apikey=openai_api_key)

    # Assume we have a violation and images
    violation = None  # Would come from detector
    image1_path = Path("path/to/image1.png")
    image2_path = Path("path/to/image2.png")
    observation = None  # Would come from trajectory parser

    # Step 1: Verify with LLM which image is correct
    print("Step 1: Verifying fluent presence with LLM...")
    llm_result = repairer.verify_fluent_with_llm(
        image1_path=image1_path,
        image2_path=image2_path,
        fluent=violation.conflicting_fluent,
        domain_name=domain_name,
        temperature=0.2  # Low temperature for deterministic results
    )
    print(f"LLM verification result: {llm_result}")

    # Step 2: Determine repair strategy
    print("\nStep 2: Determining repair strategy...")
    repair_choice, fluent_should_be_present = repairer.determine_repair_choice(
        violation=violation,
        image1_path=image1_path,
        image2_path=image2_path,
        domain_name=domain_name
    )
    print(f"Repair choice: {repair_choice.value}")
    print(f"Fluent should be present: {fluent_should_be_present}")

    # Step 3: Apply the repair
    print("\nStep 3: Applying repair to observation...")
    repaired_obs, repair_op = repairer.repair_observation(
        observation=observation,
        violation=violation,
        repair_choice=repair_choice,
        fluent_should_be_present=fluent_should_be_present
    )
    print(f"Repair operation: {repair_op}")


def example_repair_multiple_violations():
    """
    Example: Repair multiple violations in sequence.

    This shows how to handle trajectories with multiple inconsistencies.
    """
    # Configuration
    domain_file = Path("path/to/domain.pddl")
    trajectory_file = Path("path/to/trajectory.pddl+")
    images_dir = Path("path/to/images")
    openai_api_key = "your-api-key"
    domain_name = "blocks"

    # Load domain and trajectory
    domain: Domain = DomainParser(domain_file).parse_domain()
    trajectory_parser = TrajectoryParser(partial_domain=domain)
    observation = trajectory_parser.parse_trajectory(trajectory_file)

    # Detect all violations
    detector = InconsistencyDetector(domain)
    transitions = detector.extract_transitions(observation)
    violations = detector.detect_effects_violations(transitions)

    if not violations:
        print("No violations to repair!")
        return

    # Initialize repairer
    repairer = DataRepairer(openai_apikey=openai_api_key)

    # Repair each violation
    print(f"Repairing {len(violations)} violations...\n")

    for idx, violation in enumerate(violations, 1):
        print(f"=== Repairing violation {idx}/{len(violations)} ===")
        print(f"Violation: {violation}")

        # Get images for this violation
        image1_path = images_dir / f"state_{violation.transition1_index + 1}.png"
        image2_path = images_dir / f"state_{violation.transition2_index + 1}.png"

        # Repair
        observation, repair_op, repair_choice = repairer.repair_violation(
            observation=observation,
            violation=violation,
            image1_path=image1_path,
            image2_path=image2_path,
            domain_name=domain_name
        )

        print(f"Repaired! Choice: {repair_choice.value}\n")

    # Verify all violations are fixed
    print("=== Verification ===")
    transitions = detector.extract_transitions(observation)
    remaining_violations = detector.detect_effects_violations(transitions)

    if remaining_violations:
        print(f"Warning: {len(remaining_violations)} violations remain")
    else:
        print("Success! All violations repaired.")


def example_integration_with_workflow():
    """
    Example: Complete workflow integrating detection and repair.

    This shows the typical end-to-end process of trajectory denoising.
    """
    # Configuration
    config = {
        "domain_file": Path("path/to/domain.pddl"),
        "trajectory_file": Path("path/to/trajectory.pddl+"),
        "images_dir": Path("path/to/images"),
        "openai_api_key": "your-api-key",
        "domain_name": "blocks",
    }

    # Initialize components
    print("Initializing trajectory denoising workflow...")
    domain = DomainParser(config["domain_file"]).parse_domain()
    detector = InconsistencyDetector(domain)
    repairer = DataRepairer(openai_apikey=config["openai_api_key"])

    # Load trajectory
    print("\nLoading trajectory...")
    trajectory_parser = TrajectoryParser(partial_domain=domain)
    observation = trajectory_parser.parse_trajectory(config["trajectory_file"])

    # Detect violations
    print("\nDetecting violations...")
    transitions = detector.extract_transitions(observation)
    violations = detector.detect_effects_violations(transitions)
    print(f"Found {len(violations)} violations")

    # Iterative repair loop
    max_iterations = 10
    iteration = 0

    while violations and iteration < max_iterations:
        iteration += 1
        print(f"\n=== Repair Iteration {iteration} ===")

        # Repair first violation
        violation = violations[0]
        print(f"Repairing: {violation}")

        # Get images
        img1 = config["images_dir"] / f"state_{violation.transition1_index + 1}.png"
        img2 = config["images_dir"] / f"state_{violation.transition2_index + 1}.png"

        # Apply repair
        observation, repair_op, _ = repairer.repair_violation(
            observation=observation,
            violation=violation,
            image1_path=img1,
            image2_path=img2,
            domain_name=config["domain_name"]
        )

        # Re-detect violations
        transitions = detector.extract_transitions(observation)
        violations = detector.detect_effects_violations(transitions)
        print(f"Remaining violations: {len(violations)}")

    # Final status
    print("\n" + "="*60)
    if violations:
        print(f"Warning: {len(violations)} violations remain after {iteration} iterations")
    else:
        print(f"Success! All violations repaired in {iteration} iterations")
    print("="*60)


if __name__ == "__main__":
    print("DataRepairer Usage Examples")
    print("="*60)

    # Uncomment to run examples:
    # example_basic_repair()
    # example_stepwise_repair()
    # example_repair_multiple_violations()
    # example_integration_with_workflow()
