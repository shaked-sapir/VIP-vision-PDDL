"""Example usage of the plan denoising system."""

from pathlib import Path
from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain

from src.plan_denoising import (
    PlanDenoiser,
    InconsistencyDetector
)
from src.simulator import load_config


def example_detect_inconsistencies(pddl_domain_file: Path, trajectory_path: Path):
    """Example: Detect inconsistencies in a trajectory."""
    domain: Domain = DomainParser(pddl_domain_file).parse_domain()

    # Create detector
    detector = InconsistencyDetector(domain)

    # Detect inconsistencies
    inconsistencies = detector.detect_effects_inconsistencies_in_trajectory(trajectory_path)

    # Print results
    detector.print_inconsistencies(inconsistencies)


def example_denoise_trajectory():
    """Example: Denoise a trajectory using PI-SAM and LLM verification."""
    # Paths
    domain_path = Path("path/to/domain.pddl")
    trajectory_path = Path("path/to/trajectory.trajectory")
    image_directory = Path("path/to/images/")
    openai_apikey = "your-openai-api-key"

    # Parse domain
    domain_parser = DomainParser(domain_path=domain_path, partial_parsing=True)
    domain = domain_parser.parse_domain()

    # Create denoiser
    denoiser = PlanDenoiser(
        domain=domain,
        openai_apikey=openai_apikey,
        image_directory=image_directory,
        domain_name="blocks",  # or "hanoi", etc.
        max_iterations=50,
        max_backtracks=5
    )

    # Denoise the trajectory
    denoised_observation, learned_domain, conflict_tree = denoiser.denoise_from_trajectory_file(
        trajectory_path,
        use_llm_verification=True
    )

    # Print results
    print("\n=== Denoising Results ===")
    print(f"Final learned domain: {learned_domain}")
    print(f"\nConflict tree:")
    print(conflict_tree)

    # Access repair history
    repairs = conflict_tree.get_current_repairs()
    print(f"\nTotal repairs made: {len(repairs)}")
    for i, repair in enumerate(repairs, 1):
        print(f"  {i}. {repair}")


def example_manual_repair():
    """Example: Manually repair a specific inconsistency."""
    from src.plan_denoising import DataRepairer
    from src.plan_denoising.conflict_tree import Inconsistency
    from pddl_plus_parser.lisp_parsers import TrajectoryParser

    # Load trajectory
    trajectory_path = Path("path/to/trajectory.trajectory")
    parser = TrajectoryParser()
    observation = parser.parse_trajectory(trajectory_path)

    # Define an inconsistency (normally detected automatically)
    inconsistency = Inconsistency(
        transition1_index=5,
        transition2_index=12,
        action_name="pickup(block1)",
        conflicting_fluent="holding(block1)",
        fluent_in_trans1_next=True,
        fluent_in_trans2_next=False
    )

    # Paths to images
    image1_path = Path("path/to/state_0006.png")  # After transition1
    image2_path = Path("path/to/state_0013.png")  # After transition2

    # Create repairer
    repairer = DataRepairer(openai_apikey="your-api-key")

    # Repair the inconsistency
    repaired_obs, repair_op, repair_choice = repairer.repair_inconsistency(
        observation,
        inconsistency,
        image1_path,
        image2_path,
        domain_name="blocks"
    )

    print(f"Repair choice: {repair_choice.value}")
    print(f"Repair operation: {repair_op}")


if __name__ == "__main__":
    print("Plan Denoising Examples")
    print("=" * 50)

    config = load_config()
    domain = "blocks"  # or "hanoi", etc.
    pddl_domain_file = Path(config['domains'][domain]['domain_file'])
    trajectory_path = Path(f"{config['paths']['experiments_dir']}/llm_cv_test__PDDLEnvBlocks-v0__gpt-4o__steps=25__01-11-2025T16:01:36/problem1.trajectory")

    # Uncomment to run examples:
    example_detect_inconsistencies(pddl_domain_file, trajectory_path)
    example_denoise_trajectory()
    example_manual_repair()
