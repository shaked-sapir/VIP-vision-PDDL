"""
Data Generator for Benchmark System

Generates training trajectories for comparing:
- PI-SAM
- Noisy PI-SAM (Conflict-driven patch search)
- ROSAME

Supported domains:
- blocksworld
- npuzzle
- hanoi
- hiking

For each domain, this generates:
1. A long visual trace (images + ground truth)
2. Full trace for ROSAME (with LLM predictions)
3. Multiple non-overlapping shorter traces for our algorithms (split from ROSAME trace)

Key efficiency: LLM vision pipeline runs ONCE on full trace, then results are split

Output structure:
benchmark/data/<domain>/experiment_<timestamp>__steps=<num_steps>/training/
    ├── rosame_trace/
    └── pi_sam_traces/

Each run creates a new experiment folder, allowing multiple experiments to coexist.
"""

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pddlgym

from benchmark.domains.blocksworld import AmlgymLLMBlocksImageTrajectoryHandler
from src.trajectory_handlers.llm_npuzzle_trajectory_handler import LLMNpuzzleImageTrajectoryHandler
from src.trajectory_handlers.llm_hanoi_trajectory_handler import LLMHanoiImageTrajectoryHandler
from src.trajectory_handlers.llm_hiking_trajectory_handler import LLMHikingImageTrajectoryHandler
from src.utils.config import load_config
from src.utils.masking import save_masking_info, load_masking_info
from src.utils.pddl import build_trajectory_file


def _generate_training_data_generic(
    domain_display_name: str,
    domain_config_key: str,
    amlgym_domain_name: str,
    trajectory_handler_class,
    benchmark_domain_path: Path,
    output_base_dir: Path,
    num_steps: int,
    problem_name: str,
    trace_length: int
) -> Tuple[Path, List[Path]]:
    """
    Generic function to generate training data for any domain.

    Args:
        domain_display_name: Display name for logging (e.g., "BLOCKS", "N-PUZZLE")
        domain_config_key: Config key for domain (e.g., "blocks", "npuzzle")
        trajectory_handler_class: Class to instantiate for trajectory handling
        benchmark_domain_path: Path to benchmark PDDL domain file
        output_base_dir: Base directory for all benchmark data
        num_steps: Total number of steps to generate
        problem_name: Problem name to use (without .pddl extension)
        trace_length: Length of each trace for our algorithms

    Returns:
        Tuple of (rosame_trace_dir, list of our_algorithm_trace_dirs)
    """
    # Generate experiment name first for display
    timestamp = datetime.now().strftime("%d-%m-%YT%H:%M:%S")
    experiment_name = f"experiment_{timestamp}__steps={num_steps}"

    print("="*80)
    print(f"GENERATING {domain_display_name} TRAINING DATA")
    print(f"Experiment: {experiment_name}")
    print("="*80)
    print()

    # Load configuration
    config = load_config()
    openai_apikey = config['openai']['api_key']
    gym_domain_name = config['domains'][domain_config_key]['gym_domain_name']
    object_detection_model = config['domains'][domain_config_key]['object_detection']['model_name']
    object_detection_temp = config['domains'][domain_config_key]['object_detection']['temperature']
    fluent_classification_model = config['domains'][domain_config_key]['fluent_classification']['model_name']
    fluent_classification_temp = config['domains'][domain_config_key]['fluent_classification']['temperature']
    problems_dir = Path(config['domains'][domain_config_key]['problems_dir'])

    # Setup output directories with experiment timestamp (already generated above)
    benchmark_domain_dir = output_base_dir / amlgym_domain_name
    benchmark_domain_dir.mkdir(parents=True, exist_ok=True)

    experiment_dir = output_base_dir / amlgym_domain_name / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    domain_data_dir = experiment_dir / "training"
    domain_data_dir.mkdir(parents=True, exist_ok=True)

    rosame_trace_dir = domain_data_dir / "rosame_trace"
    rosame_trace_dir.mkdir(exist_ok=True)

    our_traces_base_dir = domain_data_dir / "pi_sam_traces"
    our_traces_base_dir.mkdir(exist_ok=True)

    # Setup trajectory handler
    print(f"Setting up trajectory handler...")
    print(f"  Gym environment: {gym_domain_name}")
    print(f"  Problem: {problem_name}")
    print(f"  Total steps: {num_steps}")
    print()

    trajectory_handler = trajectory_handler_class(
        domain_name=gym_domain_name,
        pddl_domain_file=benchmark_domain_path,
        openai_apikey=openai_apikey,
        object_detector_model=object_detection_model,
        object_detection_temperature=object_detection_temp,
        fluent_classifier_model=fluent_classification_model,
        fluent_classification_temperature=fluent_classification_temp
    )

    # Generate trajectory
    print(f"Generating {num_steps}-step trajectory...")
    rosame_images_dir = rosame_trace_dir / f"{problem_name}_images"
    rosame_images_dir.mkdir()

    ground_actions = trajectory_handler.create_trajectory_from_gym(
        problem_name=problem_name,
        images_output_path=rosame_images_dir,
        num_steps=num_steps
    )

    print(f"✓ Generated {len(ground_actions)} steps")
    print(f"  Sample actions: {ground_actions[:3]}...")
    print()

    # Load ground truth trajectory (for comparison purposes)
    gt_trajectory_file = rosame_images_dir / f"{problem_name}_trajectory.json"
    with open(gt_trajectory_file, 'r') as f:
        gt_trajectory = json.load(f)

    # Run LLM vision pipeline on full trajectory (object detection once, fluent classification per image)
    print(f"Running LLM vision pipeline on {num_steps}-step trajectory...")
    print(f"  This will perform object detection once and fluent classification {num_steps + 1} times")

    imaged_trajectory = trajectory_handler.create_trajectory_and_masks(
        problem_name=problem_name,
        actions=ground_actions,
        images_path=rosame_images_dir
    )

    print(f"✓ LLM vision pipeline complete")
    print(f"  Saved trajectory: {problem_name}.trajectory")
    print(f"  Saved masking info: {problem_name}.masking_info")
    print()

    # Load the generated trajectory masking info for splitting
    trajectory_masking_info = load_masking_info(
        Path(rosame_images_dir) / f"{problem_name}.masking_info",
        trajectory_handler.domain
    )

    # Copy problem file to ROSAME directory for amlgym_models compatibility
    problem_file_path = problems_dir / f"{problem_name}.pddl"
    shutil.copy(problem_file_path, rosame_images_dir)
    transform_problems_pddlgym_to_amlgym(domain_config_key, rosame_images_dir)

    print(f"✓ Saved ROSAME trace to: {rosame_trace_dir}")
    print(f"  Images: {rosame_images_dir}")
    print(f"  Ground truth: {gt_trajectory_file.name}")
    print(f"  LLM trajectory: {problem_name}.trajectory")
    print(f"  LLM masking: {problem_name}.masking_info")
    print(f"  Problem file: {problem_name}.pddl")
    print()

    # Cut into non-overlapping traces of specified length
    print(f"Cutting into {num_steps // trace_length} traces of {trace_length} steps...")
    print(f"  Note: Splitting LLM predictions (no re-running of vision pipeline)")
    our_trace_dirs = []

    for trace_idx in range(num_steps // trace_length):
        start_step = trace_idx * trace_length
        end_step = start_step + trace_length

        trace_dir = our_traces_base_dir / f"trace_{trace_idx}"
        trace_dir.mkdir()
        trace_images_dir = trace_dir / "images"
        trace_images_dir.mkdir()

        # Copy images for this trace (including initial state)
        for step in range(start_step, end_step + 1):
            src_image = rosame_images_dir / f"state_{step:04d}.png"
            dst_image = trace_images_dir / f"state_{step - start_step:04d}.png"
            shutil.copy(src_image, dst_image)

        # Extract actions for this trace
        trace_actions = ground_actions[start_step:end_step]

        # Extract ground truth steps for this trace (for comparison)
        trace_gt_trajectory = gt_trajectory[start_step:end_step]

        # Extract LLM predictions for this trace (without re-running LLM)
        trace_imaged_trajectory = imaged_trajectory[start_step:end_step]
        trace_masking_info = trajectory_masking_info[start_step:end_step + 1]  # +1 because we need initial state too

        # Save actions
        trace_actions_file = trace_dir / "actions.json"
        with open(trace_actions_file, 'w') as f:
            json.dump({
                "problem_name": problem_name,
                "trace_index": trace_idx,
                "start_step": start_step,
                "end_step": end_step,
                "num_steps": len(trace_actions),
                "actions": trace_actions
            }, f, indent=2)

        # Save ground truth trajectory (for comparison)
        trace_gt_file = trace_dir / f"{problem_name}_trace_{trace_idx}_trajectory.json"
        with open(trace_gt_file, 'w') as f:
            json.dump(trace_gt_trajectory, f, indent=2)

        # Save LLM trajectory file (PDDL format)
        trace_problem_name = f"{problem_name}_trace_{trace_idx}"
        build_trajectory_file(trace_imaged_trajectory, trace_problem_name, trace_dir)

        # Save LLM masking info
        save_masking_info(trace_dir, trace_problem_name, trace_masking_info)

        # Copy problem file to trace directory (required by amlgym_models)
        trace_problem_file = trace_dir / f"{trace_problem_name}.pddl"
        shutil.copy(problem_file_path, trace_problem_file)
        transform_problems_pddlgym_to_amlgym(domain_config_key, trace_dir)

        # Save metadata about which states this trace contains
        trace_metadata_file = trace_dir / "trace_metadata.json"
        with open(trace_metadata_file, 'w') as f:
            json.dump({
                "trace_index": trace_idx,
                "start_state": start_step,
                "end_state": end_step,
                "num_states": end_step - start_step + 1,
                "note": f"Contains states {start_step}-{end_step} from the full ROSAME trace",
                "files": {
                    "trajectory": f"{trace_problem_name}.trajectory",
                    "masking_info": f"{trace_problem_name}.masking_info",
                    "problem": f"{trace_problem_name}.pddl",
                    "actions": "actions.json",
                    "ground_truth": f"{problem_name}_trace_{trace_idx}_trajectory.json"
                }
            }, f, indent=2)

        our_trace_dirs.append(trace_dir)
        print(f"  ✓ Trace {trace_idx}: states {start_step}-{end_step} → {trace_dir.name}")

    print()
    print("="*80)
    print("TRAINING DATA GENERATION COMPLETE")
    print("="*80)
    print(f"\nExperiment saved to: {experiment_dir}")
    print(f"  Experiment name: {experiment_name}")
    print(f"  Data directory: {domain_data_dir}")
    print(f"  ROSAME trace: {rosame_trace_dir.name}")
    print(f"  Our traces: {our_traces_base_dir.name} ({len(our_trace_dirs)} traces)")
    print()

    return rosame_trace_dir, our_trace_dirs


def generate_blocks_training_data(
    output_base_dir: Path,
    num_steps: int = 100,
    problem_name: str = "problem7",
    trace_length: int = 15
) -> Tuple[Path, List[Path]]:
    """
    Generate training data for blocksworld domain.

    Args:
        output_base_dir: Base directory for all benchmark data
        num_steps: Total number of steps to generate (default: 100)
        problem_name: Problem name to use (without .pddl extension)
        trace_length: Length of each trace for our algorithms (default: 15)

    Returns:
        Tuple of (rosame_trace_dir, list of our_algorithm_trace_dirs)
    """
    benchmark_domain_path = Path(project_root) / "benchmark" / "domains" / "blocksworld" / "blocksworld.pddl"

    return _generate_training_data_generic(
        domain_display_name="BLOCKSWORLD",
        domain_config_key="blocks",
        amlgym_domain_name="blocksworld",
        trajectory_handler_class=AmlgymLLMBlocksImageTrajectoryHandler,
        benchmark_domain_path=benchmark_domain_path,
        output_base_dir=output_base_dir,
        num_steps=num_steps,
        problem_name=problem_name,
        trace_length=trace_length
    )


def generate_npuzzle_training_data(
    output_base_dir: Path,
    num_steps: int = 100,
    problem_name: str = "problem1",
    trace_length: int = 15
) -> Tuple[Path, List[Path]]:
    """
    Generate training data for n-puzzle domain.

    Args:
        output_base_dir: Base directory for all benchmark data
        num_steps: Total number of steps to generate (default: 100)
        problem_name: Problem name to use (without .pddl extension)
        trace_length: Length of each trace for our algorithms (default: 15)

    Returns:
        Tuple of (rosame_trace_dir, list of our_algorithm_trace_dirs)
    """
    benchmark_domain_path = Path(project_root) / "benchmark" / "domains" / "n_puzzle" / "n_puzzle.pddl"

    return _generate_training_data_generic(
        domain_display_name="N-PUZZLE",
        domain_config_key="n_puzzle",
        amlgym_domain_name="n_puzzle_typed",
        trajectory_handler_class=LLMNpuzzleImageTrajectoryHandler,
        benchmark_domain_path=benchmark_domain_path,
        output_base_dir=output_base_dir,
        num_steps=num_steps,
        problem_name=problem_name,
        trace_length=trace_length
    )


def generate_hanoi_training_data(
    output_base_dir: Path,
    num_steps: int = 100,
    problem_name: str = "problem0",
    trace_length: int = 15
) -> Tuple[Path, List[Path]]:
    """
    Generate training data for hanoi domain.

    Args:
        output_base_dir: Base directory for all benchmark data
        num_steps: Total number of steps to generate (default: 100)
        problem_name: Problem name to use (without .pddl extension)
        trace_length: Length of each trace for our algorithms (default: 15)

    Returns:
        Tuple of (rosame_trace_dir, list of our_algorithm_trace_dirs)
    """
    benchmark_domain_path = Path(project_root) / "benchmark" / "domains" / "hanoi" / "hanoi.pddl"

    return _generate_training_data_generic(
        domain_display_name="HANOI",
        domain_config_key="hanoi",
        amlgym_domain_name="hanoi",
        trajectory_handler_class=LLMHanoiImageTrajectoryHandler,
        benchmark_domain_path=benchmark_domain_path,
        output_base_dir=output_base_dir,
        num_steps=num_steps,
        problem_name=problem_name,
        trace_length=trace_length
    )


def generate_hiking_training_data(
    output_base_dir: Path,
    num_steps: int = 100,
    problem_name: str = "problem2",
    trace_length: int = 15
) -> Tuple[Path, List[Path]]:
    """
    Generate training data for hiking domain.

    Args:
        output_base_dir: Base directory for all benchmark data
        num_steps: Total number of steps to generate (default: 100)
        problem_name: Problem name to use (without .pddl extension)
        trace_length: Length of each trace for our algorithms (default: 15)

    Returns:
        Tuple of (rosame_trace_dir, list of our_algorithm_trace_dirs)
    """
    benchmark_domain_path = Path(project_root) / "benchmark" / "domains" / "hiking" / "hiking.pddl"

    return _generate_training_data_generic(
        domain_display_name="HIKING",
        domain_config_key="hiking",
        amlgym_domain_name="hiking",
        trajectory_handler_class=LLMHikingImageTrajectoryHandler,
        benchmark_domain_path=benchmark_domain_path,
        output_base_dir=output_base_dir,
        num_steps=num_steps,
        problem_name=problem_name,
        trace_length=trace_length
    )


def transform_problems_pddlgym_to_amlgym(domain_name: str, problems_dir: Path) -> None:
    """
    Transform all PDDLGym problem files in the specified directory to AMLGym format.

    Args:
        domain_name: Name of the domain ('blocksworld' or 'npuzzle')
        problems_dir: Directory containing PDDLGym problem files
    """
    for problem_file in problems_dir.glob("*.pddl"):
        if domain_name == "blocks":
            transform_blocks_problem_pddlgym_to_amlgym(problem_file)
        elif domain_name == "n_puzzle":
            # Currently, no transformation needed for npuzzle
            transform_npuzzle_problem_pddlgym_to_amlgym(problem_file)
        elif domain_name == "hanoi":
            return  # No transformation needed for hanoi
        elif domain_name == "hiking":
            return  # No transformation needed for hiking
        else:
            raise ValueError(f"Domain '{domain_name}' not supported for transformation.")


def transform_blocks_problem_pddlgym_to_amlgym(problem_file_path: Path) -> Path:
    """
    Transform a Blocksworld problem file from PDDLGym format to AMLGym format.

    Args:
        problem_file_path: Path to the PDDLGym problem file

    Returns:
        Path to the transformed AMLGym problem file
    """
    with open(problem_file_path, 'r') as f:
        content = f.read()

    # Rename domain
    content = content.replace('(:domain blocks)', '(:domain blocksworld)')

    # Remove robot reference fro objects
    content = content.replace('robot - robot', '')

    # Remove robot references from initial state
    content = content.replace('(handempty robot)', '(handempty)')
    content = content.replace('(handfull robot)', '')

    with open(problem_file_path, 'w') as f:
        f.write(content)

    return problem_file_path


def transform_npuzzle_problem_pddlgym_to_amlgym(problem_file_path: Path) -> Path:
    """
    Transform a Npuzzle problem file from PDDLGym format to AMLGym format.
    as we have only one problem possible here, we just copy a predefined file
    Args:
        problem_file_path: Path to the PDDLGym problem file

    Returns:
        Path to the transformed AMLGym problem file
    """
    with open(Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/benchmark/domains/n_puzzle/eight01x_amlgym.pddl"), 'r') as f:
        content = f.read()

    with open(problem_file_path, 'w') as f:
        f.write(content)

    return problem_file_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate training data for benchmark experiments"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="hiking",
        choices=["blocksworld", "npuzzle", "hanoi", "hiking"],
        help="Domain to generate data for (default: blocksworld)"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Total number of steps to generate (default: 100)"
    )
    parser.add_argument(
        "--trace-length",
        type=int,
        default=2,
        help="Length of each trace for our algorithms (default: 15)"
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="problem2",
        help="Problem name to use from PDDLGym (default: problem7 for blocksworld, problem0 for hanoi, problem2 for hiking)"
    )

    args = parser.parse_args()

    output_dir = Path(__file__).parent / "data"

    if args.domain == "blocksworld":
        rosame_dir, our_dirs = generate_blocks_training_data(
            output_base_dir=output_dir,
            num_steps=args.num_steps,
            problem_name=args.problem,
            trace_length=args.trace_length
        )
        print(f"Generated {len(our_dirs)} traces for our algorithms")
    elif args.domain == "npuzzle":
        rosame_dir, our_dirs = generate_npuzzle_training_data(
            output_base_dir=output_dir,
            num_steps=args.num_steps,
            problem_name=args.problem,
            trace_length=args.trace_length
        )
        print(f"Generated {len(our_dirs)} traces for our algorithms")
    elif args.domain == "hanoi":
        rosame_dir, our_dirs = generate_hanoi_training_data(
            output_base_dir=output_dir,
            num_steps=args.num_steps,
            problem_name=args.problem,
            trace_length=args.trace_length
        )
        print(f"Generated {len(our_dirs)} traces for our algorithms")
    elif args.domain == "hiking":
        rosame_dir, our_dirs = generate_hiking_training_data(
            output_base_dir=output_dir,
            num_steps=args.num_steps,
            problem_name=args.problem,
            trace_length=args.trace_length
        )
        print(f"Generated {len(our_dirs)} traces for our algorithms")
    else:
        print(f"Domain '{args.domain}' not yet implemented")
