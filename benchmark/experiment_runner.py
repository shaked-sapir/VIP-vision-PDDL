"""
Experiment Runner for Benchmark System

Runs PI-SAM, Noisy PI-SAM, and ROSAME on test problems and collects metrics.
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser
from pddl_plus_parser.models import Domain, Observation

from src.utils.masking import load_masked_observation
from src.pi_sam.pi_sam_learning import PiSAMLearner
from src.utils.config import load_config


def select_test_problems(training_problem: str = "problem1.pddl") -> List[Tuple[str, str]]:
    """
    Select test problems for evaluation.

    Returns:
        List of (problem_name, problem_source) tuples
        where problem_source is one of: 'blocks', 'blocks_test', 'blocks_medium'
    """
    test_problems = []

    # From pddl/blocks (4 problems - exclude training problem)
    blocks_problems = ['problem3.pddl', 'problem5.pddl', 'problem7.pddl', 'problem9.pddl']
    for prob in blocks_problems:
        if prob != training_problem:
            test_problems.append((prob, 'blocks'))

    # From pddl/blocks_test (5 problems)
    blocks_test_problems = ['problem2.pddl', 'problem4.pddl', 'problem6.pddl',
                            'problem8.pddl', 'problem10.pddl']
    for prob in blocks_test_problems:
        test_problems.append((prob, 'blocks_test'))

    # From pddl/blocks_medium (5 problems)
    blocks_medium_problems = ['problem0.pddl', 'problem2.pddl', 'problem3.pddl',
                              'problem4.pddl', 'problem5.pddl']
    for prob in blocks_medium_problems:
        test_problems.append((prob, 'blocks_medium'))

    return test_problems


def run_pisam_experiment(
    training_traces: List[Path],
    domain_file: Path,
    test_problems: List[Tuple[str, str]],
    output_dir: Path
) -> Dict:
    """
    Run PI-SAM learning and evaluation.

    Args:
        training_traces: List of paths to training trace directories
        domain_file: Path to domain PDDL file
        test_problems: List of test problems to evaluate
        output_dir: Directory to save results

    Returns:
        Dictionary with experiment results
    """
    print("="*80)
    print("RUNNING PI-SAM EXPERIMENT")
    print("="*80)
    print()

    # Parse domain
    print(f"Loading domain from: {domain_file}")
    domain = DomainParser(domain_file, partial_parsing=True).parse_domain()

    # Load all training observations
    print(f"\nLoading {len(training_traces)} training traces...")
    training_observations = []

    for trace_dir in training_traces:
        trajectory_file = trace_dir / "problem1.trajectory"
        masking_file = trace_dir / "problem1.masking_info"

        if not trajectory_file.exists() or not masking_file.exists():
            print(f"  Warning: Missing files in {trace_dir.name}, skipping")
            continue

        print(f"  Loading {trace_dir.name}...")
        obs = load_masked_observation(trajectory_file, masking_file, domain)
        training_observations.append(obs)

    print(f"\n✓ Loaded {len(training_observations)} training observations")

    # Run PI-SAM learning
    print("\nRunning PI-SAM learning...")
    start_time = time.time()

    learner = PiSAMLearner(domain)
    learned_model = learner.learn_action_model(training_observations)

    learning_time = time.time() - start_time
    print(f"✓ Learning completed in {learning_time:.2f} seconds")

    # Save learned model
    model_output_dir = output_dir / "pisam"
    model_output_dir.mkdir(parents=True, exist_ok=True)

    learned_model_file = model_output_dir / "learned_model.pddl"
    with open(learned_model_file, 'w') as f:
        f.write(str(learned_model))

    print(f"✓ Saved learned model to: {learned_model_file}")

    # TODO: Evaluate on test problems
    # This will be implemented in the next phase

    results = {
        "algorithm": "PI-SAM",
        "training_traces": len(training_observations),
        "learning_time_seconds": learning_time,
        "learned_model_file": str(learned_model_file),
        "test_results": []  # To be filled in evaluation phase
    }

    # Save results
    results_file = model_output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Saved results to: {results_file}")
    print()

    return results


def run_noisy_pisam_experiment(
    training_traces: List[Path],
    domain_file: Path,
    test_problems: List[Tuple[str, str]],
    output_dir: Path
) -> Dict:
    """
    Run Noisy PI-SAM (Conflict-driven patch search) experiment.

    Args:
        training_traces: List of paths to training trace directories
        domain_file: Path to domain PDDL file
        test_problems: List of test problems to evaluate
        output_dir: Directory to save results

    Returns:
        Dictionary with experiment results
    """
    print("="*80)
    print("RUNNING NOISY PI-SAM EXPERIMENT")
    print("="*80)
    print()

    # TODO: Implement Noisy PI-SAM learning
    # This will integrate the conflict-driven patch search

    print("Noisy PI-SAM implementation pending...")
    print("  This will use ConflictDrivenPatchSearch")
    print("  And SimpleNoisyPisamLearner")
    print()

    results = {
        "algorithm": "Noisy PI-SAM",
        "training_traces": len(training_traces),
        "learning_time_seconds": 0.0,
        "test_results": []
    }

    return results


def run_rosame_experiment(
    rosame_trace_dir: Path,
    domain_file: Path,
    test_problems: List[Tuple[str, str]],
    output_dir: Path
) -> Dict:
    """
    Prepare data and run ROSAME experiment.

    Args:
        rosame_trace_dir: Path to ROSAME training trace directory
        domain_file: Path to domain PDDL file
        test_problems: List of test problems to evaluate
        output_dir: Directory to save results

    Returns:
        Dictionary with experiment results
    """
    print("="*80)
    print("RUNNING ROSAME EXPERIMENT")
    print("="*80)
    print()

    # TODO: Implement ROSAME data preparation and execution
    # This will use the probability observations we generated

    print("ROSAME implementation pending...")
    print(f"  Will use data from: {rosame_trace_dir}")
    print("  Will use: rosame_probability_observations.json")
    print()

    results = {
        "algorithm": "ROSAME",
        "training_data": str(rosame_trace_dir),
        "learning_time_seconds": 0.0,
        "test_results": []
    }

    return results


def run_benchmark_experiments(
    data_dir: Path,
    domain_file: Path,
    output_dir: Path
) -> Dict:
    """
    Run all benchmark experiments.

    Args:
        data_dir: Directory containing training data
        domain_file: Path to equalized domain file
        output_dir: Directory to save experiment results

    Returns:
        Dictionary with all results
    """
    print("="*80)
    print("BENCHMARK EXPERIMENTS - BLOCKS DOMAIN")
    print("="*80)
    print()

    # Select test problems
    print("Selecting test problems...")
    test_problems = select_test_problems()
    print(f"✓ Selected {len(test_problems)} test problems:")
    for prob, source in test_problems:
        print(f"  - {prob} (from {source})")
    print()

    # Get training traces
    our_traces_dir = data_dir / "our_algorithms_traces"
    training_traces = sorted([d for d in our_traces_dir.iterdir() if d.is_dir()])

    rosame_trace_dir = data_dir / "rosame_trace"

    print(f"Training data:")
    print(f"  Our algorithms: {len(training_traces)} traces")
    print(f"  ROSAME: {rosame_trace_dir}")
    print()

    # Run experiments
    all_results = {
        "domain": "blocks",
        "training_problem": "problem1.pddl",
        "num_training_traces": len(training_traces),
        "num_test_problems": len(test_problems),
        "test_problems": [{"name": p, "source": s} for p, s in test_problems],
        "results": {}
    }

    # PI-SAM
    pisam_results = run_pisam_experiment(
        training_traces, domain_file, test_problems, output_dir
    )
    all_results["results"]["pisam"] = pisam_results

    # Noisy PI-SAM
    noisy_pisam_results = run_noisy_pisam_experiment(
        training_traces, domain_file, test_problems, output_dir
    )
    all_results["results"]["noisy_pisam"] = noisy_pisam_results

    # ROSAME
    rosame_results = run_rosame_experiment(
        rosame_trace_dir, domain_file, test_problems, output_dir
    )
    all_results["results"]["rosame"] = rosame_results

    # Save overall results
    overall_results_file = output_dir / "benchmark_results.json"
    with open(overall_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("="*80)
    print("BENCHMARK EXPERIMENTS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  Overall: {overall_results_file.name}")
    print()

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run benchmark experiments comparing PI-SAM, Noisy PI-SAM, and ROSAME"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="blocks",
        help="Domain to run experiments for (default: blocks)"
    )

    args = parser.parse_args()

    # Setup paths
    benchmark_dir = Path(__file__).parent
    data_dir = benchmark_dir / "data" / args.domain / "training"
    domain_file = benchmark_dir / "domains" / args.domain / f"{args.domain}_no_handfull.pddl"
    output_dir = benchmark_dir / "results" / args.domain

    # Verify training data exists
    if not data_dir.exists():
        print(f"Error: Training data not found at {data_dir}")
        print("Please run data_generator.py and noise_generator.py first")
        sys.exit(1)

    # Check if noise generation is complete
    rosame_trajectory = data_dir / "rosame_trace" / "problem1.trajectory"
    if not rosame_trajectory.exists():
        print("Warning: Noise generation may not be complete")
        print(f"  Missing: {rosame_trajectory}")
        print("\nPlease wait for noise_generator.py to complete")
        sys.exit(1)

    # Run experiments
    run_benchmark_experiments(data_dir, domain_file, output_dir)
