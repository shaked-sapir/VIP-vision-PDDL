"""
Data Generator for Benchmark System

Generates training trajectories for comparing:
- PI-SAM
- Noisy PI-SAM (Conflict-driven patch search)
- ROSAME

For each domain, this generates:
1. A 100-step visual trace (images + ground truth)
2. Full trace for ROSAME
3. Five 15-step non-overlapping traces for our algorithms
"""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pddlgym

from src.trajectory_handlers.llm_blocks_trajectory_handler import LLMBlocksImageTrajectoryHandler
from src.utils.config import load_config


def generate_blocks_training_data(
    output_base_dir: Path,
    num_steps: int = 100,
    problem_name: str = "problem1.pddl",
    trace_length: int = 15
) -> Tuple[Path, List[Path]]:
    """
    Generate training data for blocks domain.

    Args:
        output_base_dir: Base directory for all benchmark data
        num_steps: Total number of steps to generate (default: 100)
        problem_name: Problem name to use from PDDLGym (default: "problem1.pddl", first problem)
        trace_length: Length of each trace for our algorithms (default: 15)

    Returns:
        Tuple of (rosame_trace_dir, list of our_algorithm_trace_dirs)
    """
    print("="*80)
    print("GENERATING BLOCKS TRAINING DATA")
    print("="*80)
    print()

    # Load configuration
    config = load_config()
    openai_apikey = config['openai']['api_key']
    domain = 'blocks'
    gym_domain_name = config['domains'][domain]['gym_domain_name']
    object_detection_model = config['domains'][domain]['object_detection']['model_name']
    object_detection_temp = config['domains'][domain]['object_detection']['temperature']
    fluent_classification_model = config['domains'][domain]['fluent_classification']['model_name']
    fluent_classification_temp = config['domains'][domain]['fluent_classification']['temperature']

    # Setup output directories
    blocks_data_dir = output_base_dir / "blocks" / "training"
    if blocks_data_dir.exists():
        shutil.rmtree(blocks_data_dir)
    blocks_data_dir.mkdir(parents=True)

    rosame_trace_dir = blocks_data_dir / "rosame_trace"
    rosame_trace_dir.mkdir()

    our_traces_base_dir = blocks_data_dir / "our_algorithms_traces"
    our_traces_base_dir.mkdir()

    # Setup trajectory handler
    print(f"Setting up trajectory handler...")
    print(f"  Gym environment: {gym_domain_name}")
    print(f"  Problem: {problem_name}")
    print(f"  Total steps: {num_steps}")
    print()

    trajectory_handler = LLMBlocksImageTrajectoryHandler(
        domain_name=gym_domain_name,
        openai_apikey=openai_apikey,
        object_detector_model=object_detection_model,
        object_detection_temperature=object_detection_temp,
        fluent_classifier_model=fluent_classification_model,
        fluent_classification_temperature=fluent_classification_temp
    )

    # Generate 100-step trajectory
    print(f"Generating {num_steps}-step trajectory...")
    rosame_images_dir = rosame_trace_dir / "images"
    rosame_images_dir.mkdir()

    ground_actions = trajectory_handler.create_trajectory_from_gym(
        problem_name=problem_name,
        images_output_path=rosame_images_dir,
        num_steps=num_steps
    )

    print(f"✓ Generated {len(ground_actions)} steps")
    print(f"  Sample actions: {ground_actions[:3]}...")
    print()

    # Load ground truth trajectory
    gt_trajectory_file = rosame_images_dir / f"{problem_name}_trajectory.json"
    with open(gt_trajectory_file, 'r') as f:
        gt_trajectory = json.load(f)

    # Save actions list for ROSAME
    rosame_actions_file = rosame_trace_dir / "actions.json"
    with open(rosame_actions_file, 'w') as f:
        json.dump({
            "problem_name": problem_name,
            "num_steps": len(ground_actions),
            "actions": ground_actions
        }, f, indent=2)

    print(f"✓ Saved ROSAME trace to: {rosame_trace_dir}")
    print(f"  Images: {rosame_images_dir}")
    print(f"  Actions: {rosame_actions_file.name}")
    print(f"  Ground truth: {gt_trajectory_file.name}")
    print()

    # Cut into non-overlapping traces of specified length
    print(f"Cutting into {num_steps // trace_length} traces of {trace_length} steps...")
    our_trace_dirs = []

    # First, we need to generate the full .trajectory file for ROSAME to enable splitting
    # We'll generate it using ground truth, then split it for our algorithm traces
    print(f"  Note: Ground truth .trajectory files will be generated after adding LLM noise")

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

        # Extract ground truth steps for this trace
        trace_gt_trajectory = gt_trajectory[start_step:end_step]

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

        # Save ground truth trajectory
        trace_gt_file = trace_dir / f"{problem_name.replace('.pddl', '')}_trace_{trace_idx}_trajectory.json"
        with open(trace_gt_file, 'w') as f:
            json.dump(trace_gt_trajectory, f, indent=2)

        # Save metadata about which states this trace contains
        trace_metadata_file = trace_dir / "trace_metadata.json"
        with open(trace_metadata_file, 'w') as f:
            json.dump({
                "trace_index": trace_idx,
                "start_state": start_step,
                "end_state": end_step,
                "num_states": end_step - start_step + 1,
                "note": f"Contains states {start_step}-{end_step} from the full ROSAME trace"
            }, f, indent=2)

        our_trace_dirs.append(trace_dir)
        print(f"  ✓ Trace {trace_idx}: states {start_step}-{end_step} → {trace_dir.name}")

    print()
    print("="*80)
    print("TRAINING DATA GENERATION COMPLETE")
    print("="*80)
    print(f"\nData saved to: {blocks_data_dir}")
    print(f"  ROSAME trace: {rosame_trace_dir.name}")
    print(f"  Our traces: {our_traces_base_dir.name} ({len(our_trace_dirs)} traces)")
    print()

    return rosame_trace_dir, our_trace_dirs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate training data for benchmark experiments"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="blocks",
        choices=["blocks", "hanoi", "slidetile"],
        help="Domain to generate data for (default: blocks)"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Total number of steps to generate (default: 100)"
    )
    parser.add_argument(
        "--trace-length",
        type=int,
        default=15,
        help="Length of each trace for our algorithms (default: 15)"
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="problem1.pddl",
        help="Problem name to use from PDDLGym (default: problem1.pddl)"
    )

    args = parser.parse_args()

    output_dir = Path(__file__).parent / "data"

    if args.domain == "blocks":
        rosame_dir, our_dirs = generate_blocks_training_data(
            output_base_dir=output_dir,
            num_steps=args.num_steps,
            problem_name=args.problem,
            trace_length=args.trace_length
        )
        print(f"Generated {len(our_dirs)} traces for our algorithms")
    else:
        print(f"Domain '{args.domain}' not yet implemented")
